import os
import pickle
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset,DataLoader,Subset,ConcatDataset
import math
import librosa

import torchaudio.prototype
import torchaudio.prototype.transforms

class TestTensorExtractor(Dataset):
    def __init__(self,audio_path,label,transformation,target_sample_rate,num_samples,device):
        self.audio_path=audio_path+"/"
        self.tensor_path=audio_path+"_pt/"
        self.device=device
        self.label=label.to(self.device)
        self.transformation=transformation.to(self.device)
        self.target_sample_rate=target_sample_rate
        self.num_samples=num_samples
        self.segments=0
        self.mapper={}
    
    def _resample_if_req(self,signal,sr):
        if sr!=self.target_sample_rate:
            resampler=torchaudio.transforms.Resample(sr,self.target_sample_rate).to(self.device)
            signal=resampler(signal)
        return signal
    
    def _mix_down_if_req(self,signal):
        if signal.shape[0]>1:
            signal=torch.mean(signal,dim=0,keepdim=True)
        return signal
    
    def _get_audio_file_path(self,idx):
        return self.audio_path+os.listdir(self.audio_path)[idx]
    
    def __len__(self):
        return len(os.listdir(self.audio_path))
    
    def _getSegments(self):
        return self.segments
    
    def _getMapper(self):
        return self.mapper
    
    def __getitem__(self,idx):
        audio_file_idx=idx
        #signal=torch.zeros(1,1,1)
        audio_file_path=self._get_audio_file_path(audio_file_idx)

        #if os.listdir(self.audio_path)[audio_file_idx][:-3]+"pt" not in os.listdir(self.tensor_path):
        signal,sr=torchaudio.load(audio_file_path)
        signal=signal.to(self.device)
        signal=self._resample_if_req(signal,sr)
        signal=self._mix_down_if_req(signal)
        signal=self.transformation(signal)
        segments=math.ceil(signal.shape[2]/self.num_samples)
        self.segments+=segments
        self.mapper[self.segments]=audio_file_idx
        torch.save(signal,self.tensor_path+os.listdir(self.audio_path)[audio_file_idx][:-3]+"pt")

        return signal, self.label, os.listdir(self.audio_path)[audio_file_idx][:-3]+"pt"

if __name__=="__main__":
    AUDIO_PATH_OTHER="/home/vivek/Project/Test Dataset/Other"
    AUDIO_PATH_NP="/home/vivek/Project/Test Dataset/Not_Progressive_Rock"
    AUDIO_PATH_P="/home/vivek/Project/Test Dataset/Progressive Rock Songs"
    LABEL_NP=torch.tensor([0.0]) #"Not Progressive Rock"
    LABEL_P=torch.tensor([1.0]) #"Progressive_Rock"
    SAMPLE_RATE=22050
    NUM_SAMPLES=431 #after mel spectogram transform
    melspectogram=torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE,n_fft=1024,hop_length=512,n_mels=128)
    if torch.cuda.is_available():
        device="cuda"
    else:
        device="cpu"
    #device="cpu"
    print(f"Using device {device}")

    other=TestTensorExtractor(AUDIO_PATH_OTHER,LABEL_NP,melspectogram,SAMPLE_RATE,NUM_SAMPLES,device)
    non_prog=TestTensorExtractor(AUDIO_PATH_NP,LABEL_NP,melspectogram,SAMPLE_RATE,NUM_SAMPLES,device)
    prog=TestTensorExtractor(AUDIO_PATH_P,LABEL_P,melspectogram,SAMPLE_RATE,NUM_SAMPLES,device)

    for signal,label,file in prog:
        print(file)
        print(signal.shape)

    #saving Mapper
    with open("/home/vivek/Project/Test Dataset/prog_mapper.pkl","wb") as fp:
        pickle.dump(prog._getMapper(),fp)
        print("Prog Mapper saved")

    #loading mapper
    with open("/home/vivek/Project/Test Dataset/prog_mapper.pkl","rb") as fp:
        mapper=pickle.load(fp)
        print("Mapper loaded")
        print(mapper)

    #saving segments
    with open("/home/vivek/Project/Test Dataset/prog_segments.txt","wb") as fp:
        pickle.dump(prog._getSegments(),fp)
        print("Prog segments saved")

    #loading segments
    with open("/home/vivek/Project/Test Dataset/prog_segments.txt","rb") as fp:
        segments=pickle.load(fp)
        print("Segments loaded")
        print(segments)