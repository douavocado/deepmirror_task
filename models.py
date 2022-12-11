# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 11:24:53 2022

@author: xusem
"""
import torch
from torch.nn import Linear, ReLU
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric.nn import TransformerConv
from torch_geometric.nn import TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


embed_dim = 39

class GNN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = GCNConv(embed_dim, 64)
        self.pool1 = TopKPooling(64, ratio=0.8)
        self.conv2 = GCNConv(64, 64)
        self.pool2 = TopKPooling(64, ratio=0.8)
        self.conv3 = GCNConv(64, 64)
        self.pool3 = TopKPooling(64, ratio=0.8)
        
        self.lin1 = Linear(128, 64)
        self.lin2 = Linear(64, 32)
        self.lin3 = Linear(32, 1)
        self.act1 = ReLU()
        self.act2 = ReLU()        
  
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = x.squeeze(1)   
        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
     
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))

        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3
        
        x = self.lin1(x)
        x = self.act1(x)
        x = self.lin2(x)
        x = self.act2(x)      
        x = F.dropout(x, p=0.5, training=self.training)

        x = torch.sigmoid(self.lin3(x)).squeeze(1)

        return x

class TransformerNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = TransformerConv(embed_dim, 64)
        self.pool1 = TopKPooling(64, ratio=0.8)
        self.conv2 = TransformerConv(64, 64)
        self.pool2 = TopKPooling(64, ratio=0.8)
        self.conv3 = TransformerConv(64, 64)
        self.pool3 = TopKPooling(64, ratio=0.8)
        
        self.lin1 = Linear(128, 64)
        self.lin2 = Linear(64, 32)
        self.lin3 = Linear(32, 1)
        self.act1 = ReLU()
        self.act2 = ReLU()        
  
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = x.squeeze(1)   
        
        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
     
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))

        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3
        
        x = self.lin1(x)
        x = self.act1(x)
        x = self.lin2(x)
        x = self.act2(x)      
        x = F.dropout(x, p=0.5, training=self.training)

        x = torch.sigmoid(self.lin3(x)).squeeze(1)

        return x