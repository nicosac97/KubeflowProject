from torchvision import datasets
from torchvision import transforms
import torch 
import argparse
from pathlib import Path 



def get_MNIST( n_samples_train=400, n_samples_test=200, n_clients=3, batch_size=25, shuffle=True):

    def non_iid_split(dataset, nb_nodes, n_samples_per_node, batch_size, shuffle, shuffle_digits=False):
        assert(nb_nodes>0 and nb_nodes<=10)

        digits=torch.arange(10) if shuffle_digits==False else torch.randperm(10, generator=torch.Generator().manual_seed(0))

        # split the digits in a fair way
        digits_split=list()
        i=0
        for n in range(nb_nodes, 0, -1):
            inc=int((10-i)/n)
            digits_split.append(digits[i:i+inc])
            i+=inc

        # load and shuffle nb_nodes*n_samples_per_node from the dataset
        loader = torch.utils.data.DataLoader(dataset,
                                            batch_size=nb_nodes*n_samples_per_node,
                                            shuffle=shuffle)
        dataiter = iter(loader)
        images_train_mnist, labels_train_mnist = dataiter.next()

        data_splitted=list()
        for i in range(nb_nodes):
            idx=torch.stack([y_ == labels_train_mnist for y_ in digits_split[i]]).sum(0).bool() # get indices for the digits
            data_splitted.append(torch.utils.data.DataLoader(torch.utils.data.TensorDataset(images_train_mnist[idx], labels_train_mnist[idx]), batch_size=batch_size, shuffle=shuffle))

        return data_splitted
        

    
    dataset_loaded_train = datasets.MNIST(
            root="mnt/mnistfed/niid/traindata",
            train=True,
            download=True,
            transform=transforms.ToTensor()
    )
    dataset_loaded_test = datasets.MNIST(
            root="mnt/mnistfed/niid/testdata",
            train=False,
            download=True,
            transform=transforms.ToTensor()
    )
    train=non_iid_split(dataset_loaded_train, n_clients, n_samples_train, batch_size, shuffle)
    test=non_iid_split(dataset_loaded_test, n_clients, n_samples_test, batch_size, shuffle)


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
    mnist_iid_train_dls, mnist_iid_test_dls = get_MNIST(
        args.nsampletrain, args.nsampletest, args.nclients, 
        args.batch_size, shuffle =True)
    
    #save data to the two output files 
    trainfw=open(args.datatrain,'wb')
    torch.save(mnist_iid_train_dls, trainfw)
    testfw=open(args.datatest,'wb')
    torch.save(mnist_iid_test_dls, testfw)
   


