import tsaug 
import numpy as np
import torch
from model import CasualConvTran
import torch.nn as nn
import torch.optim as optim
from numpy import load
import torch.nn.functional as F
from torch.autograd import Function
import torch.nn.init as init
from utils_fixmatchl import data_loading
from utils_fixmatchl import EarlyStopping
from sklearn.metrics import f1_score
from utils_fixmatchl import dropout
L2018=np.load('/home/malo/Stage/Data/data modifiées 11 classes/l2018_modif.npz',allow_pickle=True)



n_epochs=250
data = L2018 
train_dataloader1,train_dataloader2,train_dataloader3,val_dataloader,test_dataloader,dates,data_shape = data_loading(data)
data_shape=(data_shape[0],data_shape[2],data_shape[1])

config={'emb_size':64,'num_heads':8,'Data_shape':data_shape,'Fix_pos_encode':'tAPE','Rel_pos_encode':'eRPE','dropout':0.2,'dim_ff':64}
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = CasualConvTran(config, 11, dates).to(device)

learning_rate = 0.0001
loss_fn = nn.CrossEntropyLoss()


optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)
#transformation = tsaug.AddNoise(scale=0.01)
#transformation = tsaug.Quantize(n_levels=20)
#transformation = tsaug.TimeWarp(n_speed_change=5, max_speed_ratio=3)
transformation = dropout(p=0.8)
nom_transformation ="add_noise"
valid_f1 = 0.
early_stopping=EarlyStopping(patience=25)

early_stop=False
for n in range(n_epochs):
    
    print(n)
    print (f'Epoch {n+1}---------------')
    model.train()
    if early_stop==False :
        for xm_batch, y_batch in train_dataloader1:
            x_batch,m_batch = xm_batch[:,:,:2],xm_batch[:,:,2] # m_batch correspond aux mask du batch
            x_batch = x_batch.to(device)
            m_batch = m_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            pred = model(x_batch,m_batch)
            loss = loss_fn(pred,y_batch)
            loss.backward()
            optimizer.step()
        model.eval()
        
        tot_pred = []
        tot_labels = []
        for xm_batch, y_batch in val_dataloader:
            x_batch,mask_batch = xm_batch[:,:,:2],xm_batch[:,:,2]
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            mask_batch = mask_batch.to(device)
            pred = model(x_batch,mask_batch)
            pred_npy = np.argmax(pred.cpu().detach().numpy(), axis=1)
            tot_pred.append( pred_npy )
            tot_labels.append( y_batch.cpu().detach().numpy())
        tot_pred = np.concatenate(tot_pred)
        
        tot_labels = np.concatenate(tot_labels)
        fscore = f1_score(tot_pred, tot_labels, average="weighted")
        fscore_a = np.round(fscore,3)
        print(f'f_score val set {fscore_a}')
        early_stop = early_stopping(fscore,model)
    else :
        print(f'early stopping en {n} epochs')
        break
torch.save(model.state_dict(), f"model_fixmatchL{nom_transformation}.pth")
model.eval()       
tot_pred = []
tot_labels = []
for xm_batch, y_batch in test_dataloader:
    x_batch,mask_batch = xm_batch[:,:,:2],xm_batch[:,:,2]
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)
    mask_batch = mask_batch.to(device)
    pred = model(x_batch,mask_batch)
    pred_npy = np.argmax(pred.cpu().detach().numpy(), axis=1)
    
    tot_pred.append( pred_npy )
    tot_labels.append( y_batch.cpu().detach().numpy())
tot_pred = np.concatenate(tot_pred)
tot_labels = np.concatenate(tot_labels)
fscore= f1_score(tot_pred, tot_labels, average="weighted")
print(fscore)
