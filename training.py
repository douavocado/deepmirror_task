# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 12:24:03 2022

@author: xusem
"""
import torch
from torchmetrics import AveragePrecision


crit = torch.nn.BCELoss()
alt_crit = AveragePrecision(task="binary")

def train_epoch(model, dataloader, optim):  
    model.train()

    alt_loss = 0
    loss_all = 0
    for data in dataloader:
        optim.zero_grad()
        output = model(data)
        label = data.y
        loss = crit(output, label.float())
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        alt_loss += alt_crit(output, label.float()).item()
        optim.step()
    
    return loss_all / len(dataloader), alt_loss / len(dataloader)

def val_data_epoch(model, dataloader):
    
    model.eval()

    alt_loss = 0
    loss_all = 0
    for data in dataloader:
        output = model(data)
        label = data.y
        loss = crit(output, label.float())
        loss_all += data.num_graphs * loss.item()
        alt_loss += alt_crit(output, label.float()).item()
    return loss_all / len(dataloader), alt_loss / len(dataloader)
    

def train_model(model, train_loader, val_loader, epochs=300, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    val_losses = []
    auprc_losses = []
    for epoch in range(epochs):        
        bce_loss, _ = train_epoch(model, train_loader, optimizer)
        train_losses.append(bce_loss)
        val_loss, auprc_loss = val_data_epoch(model, val_loader)
        val_losses.append(val_loss)
        auprc_losses.append(auprc_loss)
        if (epoch + 1) % 5 == 0:
            print(f'epoch: {epoch+1}, training_loss: {bce_loss}, val_loss: {val_loss}, val_AUPRC: {auprc_loss}')
    
    return train_losses, val_losses, auprc_losses