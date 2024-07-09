import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
import random
L2018=np.load('/home/malo/Stage/Data/data modifiées 11 classes/l2018_modif.npz',allow_pickle=True)
L2019=np.load('/home/malo/Stage/Data/data modifiées 11 classes/l2019_modif.npz',allow_pickle=True)
L2020=np.load('/home/malo/Stage/Data/data modifiées 11 classes/l2020_modif.npz',allow_pickle=True)
R2018=np.load('/home/malo/Stage/Data/data modifiées 11 classes/r2018_modif.npz',allow_pickle=True)
R2019=np.load('/home/malo/Stage/Data/data modifiées 11 classes/r2019_modif.npz',allow_pickle=True)
R2020=np.load('/home/malo/Stage/Data/data modifiées 11 classes/r2020_modif.npz',allow_pickle=True)
T2018=np.load('/home/malo/Stage/Data/data modifiées 11 classes/t2018_modif.npz',allow_pickle=True)
T2019=np.load('/home/malo/Stage/Data/data modifiées 11 classes/t2019_modif.npz',allow_pickle=True)
T2020=np.load('/home/malo/Stage/Data/data modifiées 11 classes/t2020_modif.npz',allow_pickle=True)

class dropout:                                                         # permet de faire une transformation de données en enlevant certaines valeurs ces valeurs sont remplacées par un zéros 
    def __init__(self, p):                                             # p est la probabilité de conservation des donées
        self.p = p
    def augment(self,x,mask):
        
        
        size = [x.shape[0],x.shape[1]]
        
        
       
        suppr = torch.bernoulli(self.p * torch.ones(size)).cuda()    # on va conserver les données là où il y a un 1 et supprimer celles où où il y a un 0
                                                                        #suppr fourni un masque que nous appliquons à la fois aux données et aux masques
        
        
        mask = mask.masked_fill(suppr==0,0)                        # application sur le masque
        suppr = suppr.unsqueeze(2)                                 # les deux lignes suivantes permettent d'appliquer le masque sur les 2 channels de nos données   
        suppr = suppr.repeat(1,1,2)
        
        x = x.masked_fill(suppr==0,0)                              # application sur les données
        
        return x,mask
class identité:        #transformation identité
        
        def augment(x):
          return(x)
def get_day_count(dates,ref_day='09-01'):
    # Days elapsed from 'ref_day' of the year in dates[0]
    ref = np.datetime64(f'{dates.astype("datetime64[Y]")[0]}-'+ref_day)
    days_elapsed = (dates - ref).astype('timedelta64[D]').astype(int) #(dates - ref_day).astype('timedelta64[D]').astype(int)#
    return torch.tensor(days_elapsed,dtype=torch.long)

def add_mask(values,mask): # permet d'attacher les mask aux données pour pouvoir faire les batchs sans perdre le mask
    mask=mask.unsqueeze(0).unsqueeze(-1)
    shape=values.shape
    mask=mask.expand(shape[0],-1,-1)
    values=torch.tensor(values,dtype=torch.float32)

    valuesWmask=torch.cat((values,mask),dim=-1)
    return valuesWmask

def comp (data,msk) : #permet de formater les données avec 365 points d'acquisitions
  data_r={'X_SAR':data['X_SAR'],'y':data['y'],'dates_SAR':data['dates_SAR']}
  ref=data['dates_SAR'][0]
  j_p=(data['dates_SAR']-ref).astype('timedelta64[D]').astype(int)
  année=list(range(365))

  année = [ref + np.timedelta64(j, 'D') for j in année ]
  mask = []

  for i,jour in enumerate(année):
    if jour not in data['dates_SAR']:

      mask+=[0]
      msk=np.insert(msk,i,0)
      data_r['dates_SAR']=np.insert(data_r['dates_SAR'],i,jour)
      data_r['X_SAR']=np.insert(data_r['X_SAR'],i,[0,0],axis=1)
    else:
      mask+=[1]


  mask=torch.tensor(mask,dtype=torch.float32)
  msk=torch.tensor(msk,dtype=torch.float32)
  return data_r,mask,msk


def suppr (data,ratio):
  data_r={'X_SAR':data['X_SAR'],'y':data['y'],'dates_SAR':data['dates_SAR']}
  ref=data['dates_SAR'][0]
  nbr,seq_len,channels=data['X_SAR'].shape #(nbr,seq_len,channels)
  
  nbr_indice=int(seq_len*ratio)
  indice=list(range(seq_len))
  indice=random.sample(indice,nbr_indice)
  mask=[0 if i in indice else 1 for i in range(seq_len)]
  mask=torch.tensor(mask)

  data_r['X_SAR']=torch.tensor(data_r['X_SAR'])
  data_r['X_SAR']=data_r['X_SAR'].permute(0,2,1)
  data_r['X_SAR']=data_r['X_SAR'].masked_fill(mask==0,0)
  data_r['X_SAR']=data_r['X_SAR'].permute(0,2,1)
  data_r['X_SAR']=data_r['X_SAR'].numpy()
  mask=mask.numpy()
  return data_r,mask




    
    
# preparation train-val-test pas encore de dataloader, # mise au format 365 jours + masque des données
def tvt_split(data):
  mapping={1:0,2:1,3:2,4:3,5:4,6:5,7:6,8:7,9:8,10:9,11:10}


  data,msk=suppr(data,0) # peut être utiliser si l'on souhaite diminuer la quantité de points d'acquisition dans les données
  data,_,mask=comp(data,msk) # rempli les données pour les mettre au fromat 365 j et donne le mask correspondant aux jours où on a mit un 0
  values=data['X_SAR']
  data_shape=data['X_SAR'].shape
  dates=data['dates_SAR']




  labels=data['y']
  labels=[mapping[v] if v in mapping else v for v in labels ]
                                                                # phase de normalisation
  max_values = np.percentile(values,99)
  min_values = np.percentile(values,1)
  values_norm=(values-min_values)/(max_values-min_values)
  values_norm[values_norm>1] = 1
  values_norm[values_norm<0] = 0
  values = values_norm                                      # les données sont normalisées
  values=add_mask(values,mask)   
  sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0) # split en train/validation + test
  indice = sss.split(values,labels)

  tv_index, test_index = next(indice)

  values_tv=[]
  values_test=[]
  labels_tv=[]
  labels_test=[]
  for i in tv_index :
    values_tv+=[values[i]]
    labels_tv+=[labels[i]]
  for j in test_index :
    values_test+=[values[j]]
    labels_test+=[labels[j]]


  sss2=StratifiedShuffleSplit(n_splits=1,test_size=0.25,random_state=0) # split de train/validation en train+validation
  indice2=sss2.split(values_tv,labels_tv)
  train_index,validation_index = next(indice2)

  values_train=[]
  values_validation=[]
  labels_train=[]
  labels_validation=[]

  for i in train_index :
    values_train+=[values_tv[i]]
    labels_train+=[labels_tv[i]]
  for j in validation_index :
    values_validation += [values_tv[j]]
    labels_validation += [labels_tv[j]]


  values_train=np.array(values_train)
  values_validation=np.array(values_validation)
  values_test=np.array(values_test)
  labels_train=np.array(labels_train)
  labels_validation=np.array(labels_validation)
  labels_test=np.array(labels_test)
  
  data_train = {'X_SAR':values_train, 'y':labels_train, 'dates_SAR':dates}
  data_validation = {'X_SAR': values_validation, 'y':labels_validation, 'dates_SAR':dates}
  data_test = {'X_SAR': values_test,'y':labels_test, 'dates_SAR':dates}



  



  return data_train,data_validation,data_test,dates,data_shape

# sélection des données pour le test et régularisatino et masking 
def selection(data):
    
    selected_data_l = []
    selected_labels_l = []
    selected_l = []
    selected_data_ul1 = []
    selected_labels_ul1 = []
    selected_ul1 = []
    selected_data_ul2 = []
    selected_labels_ul2 = []
    selected_ul2 = []
    values = data['X_SAR']
    labels = data['y']
    dates = data['dates_SAR']
    
    # Pour chaque label de 0 à 10
    for label in range(11):
        
        # Sélection des indices correspondant à ce label
        indices = np.where(labels == label)[0]
        
        # Vérification qu'il y a au moins 400 éléments pour ce label
        if len(indices) >= 400:
            
            # Sélection aléatoire de 400 indices parmi ceux disponibles
            indices = np.random.choice(indices, 400, replace=False)
            
            
            
            indices_l = indices[:100]                # indices des données dont on conservera les labels
            indices_ul1 = indices[100:250]            # indices de la première moitié des données dont on ne conservera pas les labels
            indices_ul2 = indices[250:400]            # de la seconde
            
            # Ajout des données et labels sélectionnés aux tableaux de résultats
            selected_data_l.append(values[indices_l])
            selected_labels_l.append(labels[indices_l])
            selected_data_ul1.append(values[indices_ul1])
            selected_labels_ul1.append(labels[indices_ul1])
            selected_data_ul2.append(values[indices_ul2])
            selected_labels_ul2.append(labels[indices_ul2])
            
        elif len(indices) == 0: 
            print(f'il n\'y a pas {label} dans les data')
        else:
            a1 = (len(indices)*3)//5
            a2 = (len(indices)*4)//5
            a3 = len(indices)
           
            
            indices_l = indices[:a1]
            indices_ul1 = indices[a1:a2]
            indices_ul2 = indices[a2:]
            selected_data_l.append(values[indices_l])
            selected_labels_l.append(labels[indices_l])
            selected_data_ul1.append(values[indices_ul1])
            selected_labels_ul1.append(labels[indices_ul1])
            selected_data_ul2.append(values[indices_ul2])
            selected_labels_ul2.append(labels[indices_ul2])
            
        
        
    selected_data_l = np.vstack(selected_data_l)
    selected_labels_l = np.hstack(selected_labels_l)
    selected_data_ul1 = np.vstack(selected_data_ul1)
    selected_labels_ul1 = np.hstack(selected_labels_ul1)
    selected_data_ul2 = np.vstack(selected_data_ul2)
    selected_labels_ul2 = np.hstack(selected_labels_ul2)
    selection_finale_l = {'X_SAR':selected_data_l,'y':selected_labels_l,'dates_SAR':dates}
    selection_finale_ul1 = {'X_SAR':selected_data_ul1,'y':[-1 for a in selected_labels_ul1],'dates_SAR':dates}
    selection_finale_ul2 = {'X_SAR':selected_data_ul2,'y':[-1 for a in selected_labels_ul2],'dates_SAR':dates}
    
    return selection_finale_l, selection_finale_ul1, selection_finale_ul2 # attention ici les données sont triées par classe
  
  
  
  
def selection_b(data): # pareil que précédemment suaf qu'ici toute les données non-labélisées sont conservées ensemble
    
    selected_data_l = []
    selected_labels_l = []
    selected_l = []
    selected_data_ul1 = []
    selected_labels_ul1 = []
    selected_ul1 = []
    selected_data_ul2 = []
    selected_labels_ul2 = []
    selected_ul2 = []
    values = data['X_SAR']
    labels = data['y']
    dates = data['dates_SAR']
    
    # Pour chaque label de 0 à 10
    for label in range(11):
        
        # Sélection des indices correspondant à ce label
        indices = np.where(labels == label)[0]
        
        # Vérification qu'il y a au moins 100 éléments pour ce label
        if len(indices) >= 400:
            
            # Sélection aléatoire de 100 indices parmi ceux disponibles
            indices = np.random.choice(indices, 400, replace=False)
            
            
            
            indices_l = indices[:100]
            indices_ul1 = indices[100:400]
            
            # Ajout des données et labels sélectionnés aux tableaux de résultats
            selected_data_l.append(values[indices_l])
            selected_labels_l.append(labels[indices_l])
            selected_data_ul1.append(values[indices_ul1])
            selected_labels_ul1.append(labels[indices_ul1])
            
            
        elif len(indices) == 0: 
            print(f'il n\'y a pas {label} dans les data')
        else:
            a1 = (len(indices)*3)//5
            a2 = (len(indices)*4)//5
            a3 = len(indices)
           
            
            indices_l = indices[:a1]
            indices_ul1 = indices[a1:]
            
            selected_data_l.append(values[indices_l])
            selected_labels_l.append(labels[indices_l])
            selected_data_ul1.append(values[indices_ul1])
            selected_labels_ul1.append(labels[indices_ul1])
            
            
        
        
    selected_data_l = np.vstack(selected_data_l)
    selected_labels_l = np.hstack(selected_labels_l)
    selected_data_ul1 = np.vstack(selected_data_ul1)
    selected_labels_ul1 = np.hstack(selected_labels_ul1)
    
    selection_finale_l = {'X_SAR':selected_data_l,'y':selected_labels_l,'dates_SAR':dates}
    selection_finale_ul1 = {'X_SAR':selected_data_ul1,'y':[-1 for a in selected_labels_ul1],'dates_SAR':dates}
    
    
    return selection_finale_l, selection_finale_ul1 # attention ici les données sont triées par classe
  
  
  

# data_laoding on va faire 3 data loader distincts pour pouvoir ajouter des données en cours de route

def data_loading(data):
        values_train = []
        labels_train = []
        data_train,data_val,data_test,dates,data_shape = tvt_split(data)
        
        value_val = data_val['X_SAR']
        labels_val = data_val['y']
        value_test = data_test['X_SAR']
        labels_test = data_test['y']
        
        data_train_l,data_train_ul1,data_train_ul2 = selection(data_train)
        value_train_l,labels_train_l = data_train_l['X_SAR'],data_train_l['y']
        value_train_ul1,labels_train_ul1= data_train_ul1['X_SAR'],data_train_ul1['y']
        value_train_ul2,labels_train_ul2= data_train_ul2['X_SAR'],data_train_ul2['y']
        x_train1, y_train1 = value_train_l,labels_train_l
        x_train2, y_train2 = np.concatenate((value_train_l,value_train_ul1)),np.concatenate((labels_train_l,labels_train_ul1))
        x_train3, y_train3 = np.concatenate((value_train_l,value_train_ul1,value_train_ul2)),np.concatenate((labels_train_l,labels_train_ul1,labels_train_ul2))
        
        x_val,y_val = torch.tensor(value_val,dtype=torch.float32),torch.tensor(labels_val,dtype=torch.int64)
        x_test,y_test = torch.tensor(value_test,dtype=torch.float32),torch.tensor(labels_test,dtype=torch.int64)
        
        x_train1, y_train1 = torch.tensor(x_train1,dtype=torch.float32),torch.tensor(y_train1,dtype=torch.int64)
        x_train2, y_train2 = torch.tensor(x_train2,dtype=torch.float32),torch.tensor(y_train2,dtype=torch.int64)
        x_train3, y_train3 = torch.tensor(x_train3,dtype=torch.float32),torch.tensor(y_train3,dtype=torch.int64)
        
        train_dataset1 = TensorDataset(x_train1, y_train1)
        train_dataset2 = TensorDataset(x_train2, y_train2)
        train_dataset3 = TensorDataset(x_train3, y_train3)
        val_dataset = TensorDataset(x_val, y_val)
        test_dataset = TensorDataset(x_test, y_test)
        
        train_dataloader1 = DataLoader(train_dataset1, shuffle=True, batch_size=64)
        train_dataloader2 = DataLoader(train_dataset2, shuffle=True, batch_size=64)
        train_dataloader3 = DataLoader(train_dataset3, shuffle=True, batch_size=64)
        val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=64)
        test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=64)
        
        return train_dataloader1,train_dataloader2,train_dataloader3,val_dataloader,test_dataloader,dates,data_shape


def data_loading_b(data):
        values_train = []
        labels_train = []
        data_train,data_val,data_test,dates,data_shape = tvt_split(data)
        
        value_val = data_val['X_SAR']
        labels_val = data_val['y']
        value_test = data_test['X_SAR']
        labels_test = data_test['y']
        
        data_train_l,data_train_ul1 = selection_b(data_train)
        value_train_l,labels_train_l = data_train_l['X_SAR'],data_train_l['y']
        value_train_ul1,labels_train_ul1= data_train_ul1['X_SAR'],data_train_ul1['y']
        
        x_train1, y_train1 = value_train_l,labels_train_l
        x_train2, y_train2 = np.concatenate((value_train_l,value_train_ul1)),np.concatenate((labels_train_l,labels_train_ul1))
        
        
        x_val,y_val = torch.tensor(value_val,dtype=torch.float32),torch.tensor(labels_val,dtype=torch.int64)
        x_test,y_test = torch.tensor(value_test,dtype=torch.float32),torch.tensor(labels_test,dtype=torch.int64)
        
        x_train1, y_train1 = torch.tensor(x_train1,dtype=torch.float32),torch.tensor(y_train1,dtype=torch.int64)
        x_train2, y_train2 = torch.tensor(x_train2,dtype=torch.float32),torch.tensor(y_train2,dtype=torch.int64)
        
        
        train_dataset1 = TensorDataset(x_train1, y_train1)
        train_dataset2 = TensorDataset(x_train2, y_train2)
        
        val_dataset = TensorDataset(x_val, y_val)
        test_dataset = TensorDataset(x_test, y_test)
        
        train_dataloader1 = DataLoader(train_dataset1, shuffle=True, batch_size=64)
        train_dataloader2 = DataLoader(train_dataset2, shuffle=True, batch_size=64)
        
        val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=64)
        test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=64)
        
        return train_dataloader1,train_dataloader2,val_dataloader,test_dataloader,dates,data_shape



# early stopping

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, checkpoint_path='best_model'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.checkpoint_path = checkpoint_path

    def __call__(self, val_loss, model):
        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(model)
        elif -val_loss > -(self.best_score - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0
            self.save_checkpoint(model)

        return self.early_stop

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.checkpoint_path)
        print("Saved new best model.")
    def reset(self):
        self.counter = 0
        self.best_score = None
        self.early_stop = False


        
        

        

_,s,_,_,_,_,_=data_loading(L2018)
i=0
for a,b in s :
  if i<2:
    print(b)
    i+=1
    l=[k  for k in range(len(b)) if b[k]==-1]
    t=[k for k in range(len(b)) if b[k]!=-1]
    al=a[t]
    au=a[l]
    break     


