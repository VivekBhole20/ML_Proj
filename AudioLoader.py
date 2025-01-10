# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 10:10:51 2024

@author: vicky
"""

import os
import pickle
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset,DataLoader,Subset,ConcatDataset
import math



class AudioLoader(Dataset):
    def __init__(self,tensor_path,data_path,label,num_samples,device):
        self.tensor_path=tensor_path+"/"
        self.device=device
        self.label=label.to(self.device)
        self.num_samples=num_samples
        with open(data_path+"mapper.pkl","rb") as fp:
            self.mapper=pickle.load(fp)
        with open(data_path+"segments.txt","rb") as fp:
            self.segments=pickle.load(fp)
        
    def _right_padding_if_req(self,signal):
        signal_len=signal.shape[2]
        if signal_len<self.num_samples:
            missing_pad=self.num_samples-signal_len
            signal=torch.nn.functional.pad(signal,(0,missing_pad))
        return signal
    
    """def _cut_if_req(self,signal):
        signal_len=signal.shape[1]
        if signal_len>self.num_samples:
            signal=signal[:,0:self.num_samples]
        return signal"""
    
    def _get_audio_file_path(self,idx):
        return self.tensor_path+os.listdir(self.tensor_path)[idx]
    
    def __len__(self):
        return self.segments
    
    def _get_audio_file_and_segment_idx(self,idx):
        prev_key=0

        for key, value in self.mapper.items():
            if(key>idx):
                return value, idx-prev_key
            prev_key=key
        raise IndexError("Index out of range")
    
    def __getitem__(self,idx):
        audio_file_idx,segment_idx=self._get_audio_file_and_segment_idx(idx)
        audio_file_path=self._get_audio_file_path(audio_file_idx)
        
        signal=torch.load(audio_file_path,map_location=self.device,weights_only=True)
        
        start_idx=segment_idx*self.num_samples
        end_idx=start_idx+self.num_samples
        signal=signal[:,:,start_idx:end_idx]
        
        
        signal=self._right_padding_if_req(signal)
        
        
        return signal, self.label
    
    
#if __name__=="__main__":
#Training Set
TENSOR_PATH_NP_OTHER="/home/vivek/Project/Training Dataset/Not_Progressive_Rock/Other_Songs_pt"
TENSOR_PATH_NP_TPOP="/home/vivek/Project/Training Dataset/Not_Progressive_Rock/Top_Of_The_Pops_pt"
TENSOR_PATH_P="/home/vivek/Project/Training Dataset/Progressive_Rock_Songs_pt"
DATA_PATH_NP_OTHER="/home/vivek/Project/Training Dataset/Not_Progressive_Rock/non_prog_other_"
DATA_PATH_NP_TPOP="/home/vivek/Project/Training Dataset/Not_Progressive_Rock/non_prog_tpop_"
DATA_PATH_P="/home/vivek/Project/Training Dataset/prog_"
#Testing Set
TESTTENSOR_PATH_OTHER="/home/vivek/Project/Test Dataset/Other_pt"
TESTTENSOR_PATH_NP="/home/vivek/Project/Test Dataset/Not_Progressive_Rock_pt"
TESTTENSOR_PATH_P="/home/vivek/Project/Test Dataset/Progressive Rock Songs_pt"
TESTDATA_PATH_OTHER="/home/vivek/Project/Test Dataset/other_"
TESTDATA_PATH_NP="/home/vivek/Project/Test Dataset/non_prog_"
TESTDATA_PATH_P="/home/vivek/Project/Test Dataset/prog_"
LABEL_NP= torch.tensor([0.0]) #np.float32(0.0) #"Not Progressive Rock"
#print(LABEL_NP.dtype)
LABEL_P=torch.tensor([1.0]) # np.float32(1.0) #"Progressive_Rock"
NUM_SAMPLES=431 #after mel spectogram transform

if torch.cuda.is_available():
    device="cuda"
else:
    device="cpu"
#device="cpu"
print(f"Using device {device}")

non_prog_other=AudioLoader(TENSOR_PATH_NP_OTHER,DATA_PATH_NP_OTHER,LABEL_NP,NUM_SAMPLES,device)
non_prog_tpop=AudioLoader(TENSOR_PATH_NP_TPOP,DATA_PATH_NP_TPOP,LABEL_NP,NUM_SAMPLES,device)
prog=AudioLoader(TENSOR_PATH_P,DATA_PATH_P,LABEL_P,NUM_SAMPLES,device)

#debuging code below
"""for signal,label in non_prog_other:
    print(signal.shape)
    print(label)

for signal,label in non_prog_tpop:
    print(signal.shape)
    print(label)

for signal,label in prog:
    print(signal.shape)
    print(label)"""

print("Dataset Built")
#train datasets
non_prog_oth_train=Subset(non_prog_other,torch.arange(2352,len(non_prog_other)))
non_prog_tpop_train=Subset(non_prog_tpop,torch.arange(640,len(non_prog_tpop)))
prog_train=Subset(prog,torch.arange(3066,len(prog)))

print("Training Dataset built")

#validation datasets
non_prog_oth_valid=Subset(non_prog_other,torch.arange(2352))
non_prog_tpop_valid=Subset(non_prog_tpop,torch.arange(640))
prog_valid=Subset(prog,torch.arange(3066))

print("Validation Dataset built")

#combine all training datasets
train_dataset=ConcatDataset([non_prog_oth_train,non_prog_tpop_train,prog_train])

#combine all training datasets
valid_dataset=ConcatDataset([non_prog_oth_valid,non_prog_tpop_valid,prog_valid])

print("Dataset Cancatenation done")

#training dataloader
torch.manual_seed(1)
train_dl=DataLoader(train_dataset,batch_size=20,shuffle=True)
valid_dl=DataLoader(valid_dataset,batch_size=20,shuffle=False)

print("Dataloaders done")

#debugging code below
"""num_nprog_o=0
num_nprog_t=0
num_prog=0
for x_batch,y_batch,z_batch in valid_dl:
    #print(z_batch[0])
    #break
    if z_batch[0]=="non_prog_other":
        num_nprog_o+=1
    if z_batch[0]=="non_prog_tpop":
        if num_nprog_t==0:
            print(num_nprog_o)
        num_nprog_t+=1
    if z_batch[0]=="prog":
        if num_prog==0:
            print(num_nprog_t)
        num_prog+=1

print(num_prog)"""

#Testing Datasets
test_other=AudioLoader(TESTTENSOR_PATH_OTHER,TESTDATA_PATH_OTHER,LABEL_NP,NUM_SAMPLES,device)
test_non_prog=AudioLoader(TESTTENSOR_PATH_NP,TESTDATA_PATH_NP,LABEL_NP,NUM_SAMPLES,device)
test_prog=AudioLoader(TESTTENSOR_PATH_P,TESTDATA_PATH_P,LABEL_P,NUM_SAMPLES,device)

"""for signal,label in test_prog:
    print(signal.shape)
    print(label)"""

test_dataset=ConcatDataset([test_non_prog,test_prog])

test_dl=DataLoader(test_dataset,batch_size=20,shuffle=False)
other_dl=DataLoader(test_other,batch_size=20,shuffle=False)

#print(test_non_prog.mapper)