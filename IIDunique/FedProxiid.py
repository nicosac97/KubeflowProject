    
    
from torchvision import datasets
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import argparse
from pathlib import Path 
from copy import deepcopy
import time 


def get_MNIST(type="iid", n_samples_train=200, n_samples_test=100, n_clients=3, batch_size=25, shuffle=True):

   
    def iid_split(dataset, nb_nodes, n_samples_per_node, batch_size, shuffle):
        # load and shuffle n_samples_per_node from the dataset
        loader = torch.utils.data.DataLoader(dataset,
                                            batch_size=n_samples_per_node,
                                            shuffle=shuffle)
        dataiter = iter(loader)
        
        data_splitted=list()
        for _ in range(nb_nodes):
            data_splitted.append(torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*(dataiter.next())), batch_size=batch_size, shuffle=shuffle))

        return data_splitted
        
    
    dataset_loaded_train = datasets.MNIST(
            root="mnt/mnistfed/iid/traindata",
            train=True,
            download=True,
            transform=transforms.ToTensor()
    )
    dataset_loaded_test = datasets.MNIST(
            root="mnt/mnistfed/iid/testdata",
            train=False,
            download=True,
            transform=transforms.ToTensor()
    )
    train=iid_split(dataset_loaded_train, n_clients, n_samples_train, batch_size, shuffle)
    test=iid_split(dataset_loaded_test, n_clients, n_samples_test, batch_size, shuffle)


    return train, test



#Definition of the CNN model to be used in FedProx algorithm 
mu=0

class CNN(nn.Module):

    """ConvNet -> Max_Pool -> RELU -> ConvNet -> 
    Max_Pool -> RELU -> FC -> RELU -> FC -> SOFTMAX"""
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#Example of Module definition invocation 

model_0 = CNN()


def loss_classifier(predictions,labels):
    
    m = nn.LogSoftmax(dim=1)
    loss = nn.NLLLoss(reduction="mean")
    return loss(m(predictions) ,labels.view(-1))

def loss_dataset(model, dataset, loss_f):
    """Compute the loss of `model` on `dataset`"""
    loss=0
    
    for idx,(features,labels) in enumerate(dataset):
        
        predictions= model(features)
        loss+=loss_f(predictions,labels)
    
    loss/=idx+1
    return loss


def accuracy_dataset(model, dataset):
    """Compute the accuracy of `model` on `dataset`"""
    
    correct=0
    
    for features,labels in iter(dataset):
        
        predictions= model(features)
        
        _,predicted=predictions.max(1,keepdim=True)
        
        correct+=torch.sum(predicted.view(-1,1)==labels.view(-1, 1)).item()
        
    accuracy = 100*correct/len(dataset.dataset)
        
    return accuracy

#Functioln that Train `model` on one epoch of `train_data`
def train_step(model, model_0, mu:int, optimizer, train_data, loss_f):
    """Train `model` on one epoch of `train_data`"""
    
    total_loss=0
    
    for idx, (features,labels) in enumerate(train_data):
        
        optimizer.zero_grad()
        
        predictions= model(features)
        
        loss=loss_f(predictions,labels)
        loss+=mu/2*difference_models_norm_2(model,model_0)
        total_loss+=loss
        
        loss.backward()
        optimizer.step()
        
    return total_loss/(idx+1)



def local_learning(model, mu:float, optimizer, train_data, epochs:int, loss_f):
    
    model_0=deepcopy(model)
    
    for e in range(epochs):
        local_loss=train_step(model,model_0,mu,optimizer,train_data,loss_f)
        
    return float(local_loss.detach().numpy())


def difference_models_norm_2(model_1, model_2):
    """Return the norm 2 difference between the two model parameters
    """
    
    tensor_1=list(model_1.parameters())
    tensor_2=list(model_2.parameters())
    
    norm=sum([torch.sum((tensor_1[i]-tensor_2[i])**2) 
        for i in range(len(tensor_1))])
    
    return norm


def set_to_zero_model_weights(model):
    """Set all the parameters of a model to 0"""

    for layer_weigths in model.parameters():
        layer_weigths.data.sub_(layer_weigths.data)


def average_models(model, clients_models_hist:list , weights:list):


    """Creates the new model of a given iteration with the models of the other
    clients"""
    
    new_model=deepcopy(model)
    set_to_zero_model_weights(new_model)

    for k,client_hist in enumerate(clients_models_hist):
        
        for idx, layer_weights in enumerate(new_model.parameters()):

            contribution=client_hist[idx].data*weights[k]
            layer_weights.data.add_(contribution)
            
    return new_model
           

def FedProx(model, training_sets:list, n_iter:int, testing_sets:list, mu=0, 
    file_name="test", epochs=5, lr=10**-2, decay=1):
    """ all the clients are considered in this implementation of FedProx
    Parameters:
        - `model`: common structure used by the clients and the server
        - `training_sets`: list of the training sets. At each index is the 
            training set of client "index"
        - `n_iter`: number of iterations the server will run
        - `testing_set`: list of the testing sets. If [], then the testing
            accuracy is not computed
        - `mu`: regularization term for FedProx. mu=0 for FedAvg
        - `epochs`: number of epochs each client is running
        - `lr`: learning rate of the optimizer
        - `decay`: to change the learning rate at each iteration
    
    returns :
        - `model`: the final global model 
    """
  
    loss_f=loss_classifier

    
    #Variables initialization
    K=len(training_sets) #number of clients
    n_samples=sum([len(db.dataset) for db in training_sets])
    weights=([len(db.dataset)/n_samples for db in training_sets])
    print("Clients' weights:",weights)
    
    
    loss_hist=[[float(loss_dataset(model, dl, loss_f).detach()) 
        for dl in training_sets]]
    acc_hist=[[accuracy_dataset(model, dl) for dl in testing_sets]]
    server_hist=[[tens_param.detach().numpy() 
        for tens_param in list(model.parameters())]]
    models_hist = []
    
    
    server_loss=sum([weights[i]*loss_hist[-1][i] for i in range(len(weights))])
    server_acc=sum([weights[i]*acc_hist[-1][i] for i in range(len(weights))])
    print(f'====> i: 0 Loss: {server_loss} Server Test Accuracy: {server_acc}')
    
    for i in range(n_iter):
        
        clients_params=[]
        clients_models=[]
        clients_losses=[]
        
        for k in range(K):
            
            local_model=deepcopy(model)
            local_optimizer=optim.SGD(local_model.parameters(),lr=lr)
            
            local_loss=local_learning(local_model,mu,local_optimizer,
                training_sets[k],epochs,loss_f)
            
            clients_losses.append(local_loss)
                
            #GET THE PARAMETER TENSORS OF THE MODEL
            list_params=list(local_model.parameters())
            list_params=[tens_param.detach() for tens_param in list_params]
            clients_params.append(list_params)    
            clients_models.append(deepcopy(local_model))
        
        
        #CREATE THE NEW GLOBAL MODEL
        model = average_models(deepcopy(model), clients_params, 
            weights=weights)
        models_hist.append(clients_models)
        
        #COMPUTE THE LOSS/ACCURACY OF THE DIFFERENT CLIENTS WITH THE NEW MODEL
        loss_hist+=[[float(loss_dataset(model, dl, loss_f).detach()) 
            for dl in training_sets]]
        acc_hist+=[[accuracy_dataset(model, dl) for dl in testing_sets]]

        server_loss=sum([weights[i]*loss_hist[-1][i] for i in range(len(weights))])
        server_acc=sum([weights[i]*acc_hist[-1][i] for i in range(len(weights))])

        print(f'====> i: {i+1} Loss: {server_loss} Server Test Accuracy: {server_acc}')
        

        server_hist.append([tens_param.detach().cpu().numpy() 
            for tens_param in list(model.parameters())])
        
        #DECREASING THE LEARNING RATE AT EACH SERVER ITERATION
        lr*=decay
            
    return model, loss_hist, acc_hist



if __name__ == '__main__':
    
    # This component receives differnt inputs
    # it only outpus one artifact containing the two downloaded lists of train and test `data`.
    parser = argparse.ArgumentParser()
    parser.add_argument('--nsampletrain', type=int)
    parser.add_argument('--nsampletest', type=int)
    parser.add_argument('--nclients', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--n_iter', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--lr', type =float)
    parser.add_argument('--mu', type =float)
    args = parser.parse_args()
    
    # Creating the directory where the output file will be created 
    # # (the directory may or may not exist).
    Path(args.datatrain).parent.mkdir(parents=True, exist_ok=True)
    Path(args.datatest).parent.mkdir(parents=True, exist_ok=True)


    start_dl=time.time()
    #Execution of the getMnist function to download and extract the dbs
    mnist_iid_train_dls, mnist_iid_test_dls = get_MNIST("iid",
        args.nsampletrain, args.nsampletest, args.nclients, 
        args.batch_size, shuffle =True)
    finish_dl=time.time()
    time_taken_dwl=finish_dl-start_dl
    print(f'Time taken to download the dataset ----> {time_taken_dwl}')    

    start= time.time()
    #invoke the FedProx function to start the training on the downloaded DB
    model_f, loss_hist_FA_iid, acc_hist_FA_iid = FedProx( model_0, 
        mnist_iid_train_dls,args.n_iter ,mnist_iid_test_dls,mu=args.mu, epochs=args.epochs, 
        lr=args.lr)
    finish= time.time() 
    time_taken_train=finish-start
    print(f'Time taken to train ----> {time_taken_train}')    


    
   



        
    




    
