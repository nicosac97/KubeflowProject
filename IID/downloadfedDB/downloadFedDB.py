    
    
from torchvision import datasets
from torchvision import transforms
import torch 
import argparse
from pathlib import Path 

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
            root="./data",
            train=True,
            download=True,
            transform=transforms.ToTensor()
    )
    dataset_loaded_test = datasets.MNIST(
            root="./data",
            train=False,
            download=True,
            transform=transforms.ToTensor()
    )
    train=iid_split(dataset_loaded_train, n_clients, n_samples_train, batch_size, shuffle)
    test=iid_split(dataset_loaded_test, n_clients, n_samples_test, batch_size, shuffle)


    return train, test


if __name__ == '__main__':
    
    # This component receives differnt inputs
    # it only outpus one artifact containing the two downloaded lists of train and test `data`.
    parser = argparse.ArgumentParser()
    parser.add_argument('--nsampletrain', type=int)
    parser.add_argument('--nsampletest', type=int)
    parser.add_argument('--nclients', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--datatrain', type=str)
    parser.add_argument('--datatest', type=str)
    args = parser.parse_args()
    
    # Creating the directory where the output file will be created 
    # # (the directory may or may not exist).
    Path(args.datatrain).parent.mkdir(parents=True, exist_ok=True)
    Path(args.datatest).parent.mkdir(parents=True, exist_ok=True)

    #Execution of the getMnist function to download and extract the dbs
    mnist_iid_train_dls, mnist_iid_test_dls = get_MNIST("iid",
        args.nsampletrain, args.nsampletest, args.nclients, 
        args.batch_size, shuffle =True)
    
    #save data to the two output files
    trainfw=open(args.datatrain,'wb')
    torch.save(mnist_iid_train_dls, trainfw)
    testfw=open(args.datatest,'wb')
    torch.save(mnist_iid_test_dls, testfw)
   



        
    




    