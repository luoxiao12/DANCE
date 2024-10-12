import torch
import torch.nn as nn
from typing import List, Dict

class sc_model(nn.Module):
    def __init__(self, input_size,num_of_class):
        super(sc_model, self).__init__()
        self.input_size = input_size
        self.k = 64
        self.f = 64

        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, 64)
        )
        self.cell = nn.Sequential(
            nn.Linear(64, num_of_class)
        )
    def forward(self, data):
        data = data.float().view(-1, self.input_size)
        embedding = self.encoder(data)
        cell_prediction = self.cell(embedding)
        return embedding,cell_prediction

class sc_classifier(nn.Module):
    def __init__(self, num_of_class):
        super(sc_classifier, self).__init__()
        self.cell = nn.Sequential(
            nn.Linear(64, num_of_class)
        )

    def forward(self, embedding):
        cell_prediction = self.cell(embedding)

        return cell_prediction

class sc_net(nn.Module):
    def __init__(self, input_size, num_of_class):
        super(sc_net, self).__init__()
        self.input_size = input_size
        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 64),
            nn.InstanceNorm1d(64),
        )
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(64, num_of_class)
        )
        
        self.cluster_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(64, num_of_class)
        )
        self.eqinv_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(64, num_of_class)
        )
    def forward(self, data):
        data = data.float().view(-1, self.input_size)
        embedding = self.encoder(data)
        cell_prediction = self.classifier(embedding)
        cluster_prediction = self.cluster_head(embedding)
        eqinv_prediction = self.eqinv_head(embedding)
        return embedding,cell_prediction,cluster_prediction,eqinv_prediction
    
    def get_parameters(self,base_lr=0.003) -> List[Dict]:
        return [{"params": self.parameters(), "lr": base_lr}]