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

class TensorExtractor(Dataset):
    def __init__(self,audio_path,label,transformation1,transformation2,transformation3,target_sample_rate,num_samples,device):
        self.audio_path=audio_path+"/"
        self.tensor_path=audio_path+"_pt_new/"
        self.device=device
        self.label=label.to(self.device)
        self.transformation1=transformation1.to(self.device)
        self.transformation2=transformation2.to(self.device)
        self.transformation3=transformation3.to(self.device)
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
        #MFCC 13
        signal1=self.transformation1(signal)
        #Melspectrogram 64
        signal2=self.transformation2(signal)
        #ChromaSpectrogram 12
        signal3=self.transformation3(signal)
        #Onset strength 1
        signal_np=signal.cpu().numpy()
        onset_env=librosa.onset.onset_strength(y=signal_np,sr=22050,aggregate=np.median,fmax=8000,n_mels=64)
        signal4=torch.tensor(onset_env).view(1,1,signal3.shape[2]).to(self.device)

        signal=torch.cat((signal1,signal2,signal3,signal4),dim=1)

        segments=math.ceil(signal.shape[2]/self.num_samples)
        self.segments+=segments
        self.mapper[self.segments]=audio_file_idx
        torch.save(signal,self.tensor_path+os.listdir(self.audio_path)[audio_file_idx][:-3]+"pt")

        return signal, self.label, os.listdir(self.audio_path)[audio_file_idx][:-3]+"pt"


if __name__=="__main__":


    """
    def _mix_down_if_req(signal):
        if signal.shape[0]>1:
            signal=torch.mean(signal,dim=0,keepdim=True)
        return signal
    def _resample_if_req(signal,sr,target_sample_rate=22050):
        if sr!=target_sample_rate:
            resampler=torchaudio.transforms.Resample(sr,target_sample_rate)
            signal=resampler(signal)
        return signal
    signal,sr=torchaudio.load("/home/vivek/Project/Training Dataset/Progressive_Rock_Songs/-04- Knots.mp3")
    signal=_mix_down_if_req(signal)
    signal=_resample_if_req(signal,sr)
    #print(signal.shape)
    signal=signal[:,0:220500]
    MFCC=torchaudio.transforms.MFCC(sample_rate=22050,n_mfcc=13,melkwargs={"n_fft":1024,"hop_length":512,"n_mels":64})
    Chromagram=torchaudio.prototype.transforms.ChromaSpectrogram(sample_rate=22050,n_fft=1024,hop_length=512)
    mfcc=MFCC(signal)
    print(mfcc.shape)
    chroma=Chromagram(signal)
    print(chroma.shape)
    signal_np=signal.numpy()
    onset_env=librosa.onset.onset_strength(y=signal_np,sr=22050,aggregate=np.median,fmax=8000,n_mels=64)
    onset_env_tensor=torch.tensor(onset_env).view(1,1,431)
    print(onset_env_tensor.shape)
    melspectogram=torchaudio.transforms.MelSpectrogram(sample_rate=22050,n_fft=1024,hop_length=512,n_mels=64)
    mel=melspectogram(signal)
    print(mel.shape)

    final_tensor=torch.cat((mfcc,mel,chroma,onset_env_tensor),dim=1)
    print(final_tensor.shape)
    """

    AUDIO_PATH_NP_OTHER="/home/vivek/Project/Training Dataset/Not_Progressive_Rock/Other_Songs"
    AUDIO_PATH_NP_TPOP="/home/vivek/Project/Training Dataset/Not_Progressive_Rock/Top_Of_The_Pops"
    AUDIO_PATH_P="/home/vivek/Project/Training Dataset/Progressive_Rock_Songs"
    LABEL_NP=torch.tensor([0.0]) #"Not Progressive Rock"
    LABEL_P=torch.tensor([1.0]) #"Progressive_Rock"
    SAMPLE_RATE=22050
    NUM_SAMPLES=431 #after mel spectogram transform
    MFCC=torchaudio.transforms.MFCC(sample_rate=22050,n_mfcc=13,melkwargs={"n_fft":1024,"hop_length":512,"n_mels":64})
    melspectogram=torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE,n_fft=1024,hop_length=512,n_mels=64)
    Chromagram=torchaudio.prototype.transforms.ChromaSpectrogram(sample_rate=22050,n_fft=1024,hop_length=512)
    if torch.cuda.is_available():
        device="cuda"
    else:
        device="cpu"
    #device="cpu"
    print(f"Using device {device}")

    non_prog_other=TensorExtractor(AUDIO_PATH_NP_OTHER,LABEL_NP,MFCC,melspectogram,Chromagram,SAMPLE_RATE,NUM_SAMPLES,device)
    non_prog_tpop=TensorExtractor(AUDIO_PATH_NP_TPOP,LABEL_NP,MFCC,melspectogram,Chromagram,SAMPLE_RATE,NUM_SAMPLES,device)
    prog=TensorExtractor(AUDIO_PATH_P,LABEL_P,MFCC,melspectogram,Chromagram,SAMPLE_RATE,NUM_SAMPLES,device)

    for signal,label,file in prog:
        print(file)
        print(signal.shape)

    #saving Mapper
    with open("/home/vivek/Project/Training Dataset/Not_Progressive_Rock/prog_mapper.pkl","wb") as fp:
        pickle.dump(prog._getMapper(),fp)
        print("Prog Mapper saved")

    #loading mapper
    with open("/home/vivek/Project/Training Dataset/Not_Progressive_Rock/prog_mapper.pkl","rb") as fp:
        mapper=pickle.load(fp)
        print("Mapper loaded")
        print(mapper)

    #saving segments
    with open("/home/vivek/Project/Training Dataset/Not_Progressive_Rock/prog_segments.txt","wb") as fp:
        pickle.dump(prog._getSegments(),fp)
        print("Prog segments saved")

    #loading segments
    with open("/home/vivek/Project/Training Dataset/Not_Progressive_Rock/prog_segments.txt","rb") as fp:
        segments=pickle.load(fp)
        print("Segments loaded")
        print(segments)