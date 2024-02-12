import torch.utils.data as data
import numpy as np
import os.path
import random
import scipy
import torch
import scanpy as sc
from sklearn.feature_extraction.text import TfidfTransformer
from torchvision import transforms
random.seed(1)
np.random.seed(1)

def reconstruct_labels(rna_label_file, atac_label_file):
    # label to idx
    rna_label_file = rna_label_file
    atac_label_file = atac_label_file
    total_rna_labels=np.loadtxt(rna_label_file)
    total_atac_labels=np.loadtxt(atac_label_file)
    label_idx_mapping = {}
    unique_labels = np.unique(total_rna_labels)
    for i, name in enumerate(unique_labels):
        label_idx_mapping[name] = i
    print(label_idx_mapping)
    rna_label = np.array([label_idx_mapping.get(label, -1) for label in total_rna_labels])
    atac_label = np.array([label_idx_mapping.get(label, -1) for label in total_atac_labels])

    return rna_label, atac_label

def load_labels(label_file):  # please run parsing_label.py first to get the numerical label file (.txt)
    
    return np.loadtxt(label_file)

def npz_reader(file_name):
    print('load npz matrix:', file_name)
    data = scipy.sparse.load_npz(file_name)
    tfidf = TfidfTransformer()
    data = tfidf.fit_transform(data.toarray().astype(float)).toarray()
    sc.pp.scale(data)
    return data

def read_from_file(data_path, label_path = None):
    data_path = os.path.join(os.path.realpath('.'), data_path)
    labels = None
    data_reader = npz_reader(data_path) 
    if label_path is not None:        
        label_path = os.path.join(os.path.realpath('.'), label_path)    
        labels = load_labels(label_path)

    return data_reader, labels


def add_noise(data, noise_level=0.01):
    noise = noise_level * torch.randn_like(data)
    return data + noise



class scDataset(data.Dataset):
    def __init__(self, train = True, data_reader = None, labels = None,  label_ratio = 1, labeled = True):
        self.train = train        
        self.labeled = labeled
        self.data_reader= data_reader.astype(float)
        self.labels = labels
        self.input_size = self.data_reader.shape[1]
        self.length = int(self.data_reader.shape[0])
        self.sample_num = int( self.data_reader.shape[0] * label_ratio )
        
        ###make sure topk data includes all the classes#####################
        k = len(np.unique(self.labels))
        unique_labels, indices = np.unique(self.labels, return_index=True)
        selected_indices = np.sort(indices[:k])
        selected_label = self.labels[selected_indices]
        selected_data = self.data_reader[selected_indices]
        remaining_label = np.delete(self.labels, selected_indices)
        remaining_data = np.delete(self.data_reader, selected_indices, axis=0)
        n = remaining_data.shape[0]
        indexes = np.random.permutation(n)
        remaining_label = remaining_label[indexes]
        remaining_data = remaining_data[indexes]
        
        self.labels = np.concatenate((selected_label, remaining_label))
        self.data_reader = np.concatenate((selected_data, remaining_data), axis=0)
        ########################################################################
    
    def __getitem__(self, index):
        if self.train:
            if self.labeled:            
                rand_idx = random.randint(0, self.sample_num - 1)
                sample = np.array(self.data_reader[rand_idx])
                sample = sample.reshape((1, self.input_size))
                in_data = sample.astype(np.float)             
                in_label = self.labels[rand_idx]
    
                return in_data, in_label, index
            else:         
                rand_idx = random.randint(self.sample_num,self.length-1)
                sample = np.array(self.data_reader[rand_idx])
                sample = sample.reshape((1, self.input_size))
                in_data = sample.astype(np.float)             
                in_label = self.labels[rand_idx]
    
                return in_data, in_label, index
            
        else:
            sample = np.array(self.data_reader[index])
            sample = sample.reshape((1, self.input_size))
            in_data = sample.astype(np.float)  

            in_label = self.labels[index]
            return in_data, in_label, index


    def __len__(self):
        if self.labeled: 
            return self.sample_num
        else:
            return self.length - self.sample_num
    
class Prepare_scDataloader():
    def __init__(self, args):
        self.rna_data = args.rna_data
        self.rna_label = args.rna_label
        self.atac_data = args.atac_data
        self.atac_label = args.atac_label
        self.label_ratio = args.label_ratio
        self.batchsize = args.batch_size
        
        print("label ratio: ", self.label_ratio)
        
        rna_label, atac_label = reconstruct_labels(self.rna_label[0],self.atac_label[0])
        
        # load RNA

        for rna_path, label_path in zip(self.rna_data, self.rna_label):  
            data_reader, _ = read_from_file(rna_path, label_path)
            # train loader 
            labeled_trainset = scDataset(True, data_reader, rna_label,  self.label_ratio,labeled = True)
            labeled_train_rna_loader = data.DataLoader(labeled_trainset, batch_size=self.batchsize, shuffle=True,drop_last=True,num_workers=10)
            if self.label_ratio < 1:       
                unlabeled_trainset = scDataset(True, data_reader, rna_label,  self.label_ratio,labeled = False)
                unlabeled_train_rna_loader = data.DataLoader(unlabeled_trainset, batch_size=self.batchsize, shuffle=True,drop_last=True,num_workers=10)    
            else:                 
                unlabeled_train_rna_loader = None
            # test loader 
            testset = scDataset(False, data_reader, rna_label)
            test_rna_loader = data.DataLoader(testset, batch_size=256, shuffle=False,num_workers=10)                        
          
        # load ATAC
        for atac_path in self.atac_data:   
            data_reader, _ = read_from_file(atac_path)
            # train loader
            trainset = scDataset(True, data_reader, atac_label)
            train_atac_loader = data.DataLoader(trainset, batch_size= self.batchsize , shuffle = True,drop_last=True,num_workers=10)                        
            # test loader
            testset = scDataset(False, data_reader, atac_label)
            test_atac_loader = data.DataLoader(testset, batch_size= 256, shuffle = False,num_workers=10 )                        
                            
        self.labeled_train_rna_loader = labeled_train_rna_loader
        self.unlabeled_train_rna_loader = unlabeled_train_rna_loader
        self.test_rna_loader = test_rna_loader
        self.train_atac_loader = train_atac_loader
        self.test_atac_loader = test_atac_loader
                    
        
        self.gene_size = data_reader.shape[1]
        self.type_num = torch.unique(torch.from_numpy(rna_label)).shape[0] 
    
    def getloader(self):
        return self.labeled_train_rna_loader, self.unlabeled_train_rna_loader,self.test_rna_loader,\
            self.train_atac_loader, self.test_atac_loader, self.gene_size, self.type_num
    
    
class scNCL_Dataloader():
    def __init__(self, args):
        self.rna_data = args.rna_data
        self.rna_label = args.rna_label
        self.atac_data = args.atac_data
        self.atac_label = args.atac_label
        self.label_ratio = args.label_ratio
        self.batchsize = args.batch_size

        rna_label, atac_label = reconstruct_labels(self.rna_label[0],self.atac_label[0])
        
        # load RNA

        for rna_path, label_path in zip(self.rna_data, self.rna_label):  
            data_reader, _ = read_from_file(rna_path, label_path)
            # train loader 
            trainset = scDataset(True, data_reader, rna_label,  self.label_ratio)
            train_rna_loader = data.DataLoader(trainset, batch_size=self.batchsize, shuffle=True,drop_last=True,num_workers=10)                        
            # test loader 
            testset = scDataset(False, data_reader, rna_label)
            test_rna_loader = data.DataLoader(testset, batch_size=256, shuffle=False,num_workers=10)                        
          
        # load ATAC
        for atac_path in self.atac_data:   
            data_reader, _ = read_from_file(atac_path)
            # train loader
            trainset = scNCL_Dataset(True, data_reader, atac_label)
            train_atac_loader = data.DataLoader(trainset, batch_size= self.batchsize , shuffle = True,drop_last=True,num_workers=10)                        
            # test loader
            testset = scDataset(False, data_reader, atac_label)
            test_atac_loader = data.DataLoader(testset, batch_size= 256, shuffle = False,num_workers=10 )                        
            self.atac_dataset = data_reader.toarray().astype(float)
            
        self.train_rna_loader = train_rna_loader
        self.test_rna_loader = test_rna_loader
        self.train_atac_loader = train_atac_loader
        self.test_atac_loader = test_atac_loader
                    
        
        self.gene_size = data_reader.shape[1]
        self.type_num = torch.unique(torch.from_numpy(rna_label)).shape[0] 
    
    def getloader(self):
        return self.train_rna_loader, self.test_rna_loader, self.train_atac_loader, self.test_atac_loader, self.gene_size, self.type_num, self.atac_dataset
    
class scNCL_Dataset(data.Dataset):
    def __init__(self, train = True, data_reader = None, labels = None,  label_ratio = 1):
        self.train = train        
        self.data_reader= data_reader.toarray().astype(float)
        self.labels = labels
        self.input_size = self.data_reader.shape[1]
        self.sample_num = int( self.data_reader.shape[0] * label_ratio )
        
        ###make sure topk data includes all the classes#####################
        k = len(np.unique(self.labels))
        unique_labels, indices = np.unique(self.labels, return_index=True)
        selected_indices = np.sort(indices[:k])
        selected_label = self.labels[selected_indices]
        selected_data = self.data_reader[selected_indices]
        remaining_label = np.delete(self.labels, selected_indices)
        remaining_data = np.delete(self.data_reader, selected_indices, axis=0)
        n = remaining_data.shape[0]
        indexes = np.random.permutation(n)
        remaining_label = remaining_label[indexes]
        remaining_data = remaining_data[indexes]
        
        self.labels = np.concatenate((selected_label, remaining_label))
        self.data_reader = np.concatenate((selected_data, remaining_data), axis=0)
        ########################################################################
    
    def __getitem__(self, index):
        if self.train:
            # get atac data            
            rand_idx = random.randint(0, self.sample_num - 1)
            sample = np.array(self.data_reader[rand_idx])
            sample = sample.reshape((1, self.input_size))
            in_data = (sample>0).astype(np.float)  # binarize data            
            in_label = self.labels[rand_idx]
 
            return in_data, rand_idx

        else:
            sample = np.array(self.data_reader[index])
            sample = sample.reshape((1, self.input_size))
            in_data = (sample>0).astype(np.float)  # binarize data

            in_label = self.labels[index]
            return in_data, in_label


    def __len__(self):
        return self.sample_num