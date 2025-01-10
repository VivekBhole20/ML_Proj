# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 13:15:49 2024

@author: vicky
"""
import os
import librosa
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio

"""def _resample_if_req(signal,sr,target_sample_rate):
    if sr!=target_sample_rate:
        resampler=torchaudio.transforms.Resample(sr,target_sample_rate)
        signal=resampler(signal)
    return signal

def load_audio_files(audio_path,idx):
    audio_files=[]
    #audio_path="C:/TechManhindra/GRE/UF/Course Work/Machine Learning/Project/Training Dataset/Not Progressive Rock/Not_Progressive_Rock/Other_Songs/"
    for i in range(idx,idx+20):
        file=os.listdir(audio_path)[i]
        file_path=os.path.join(audio_path,file)
        y, sr = torchaudio.load(file_path)
        y=_resample_if_req(y,sr,22050)
        audio_files.append(y)
    return audio_files
    

#mean and median calculation for signal length
audio_path="C:/TechManhindra/GRE/UF/Course Work/Machine Learning/Project/Training Dataset/Not Progressive Rock/Not_Progressive_Rock/Other_Songs/"

audio_files=load_audio_files(audio_path,80)
#print(aud)
min_aud=10e10
max_aud=0
audio_length=[]
for audio in audio_files:
    audio_length.append(audio.shape[1])
    min_aud=min(min_aud,len(audio))
    max_aud=max(max_aud,len(audio))

print(np.mean(audio_length))
print(np.median(audio_length))"""

"""
class AudioLoader:
    def __init__(self,audio_path,label,transformation,target_sample_rate,num_samples,device):
        self.audio_path=audio_path
        self.label=label
        self.device=device
        self.transformation=transformation.to(self.device)
        self.target_sample_rate=target_sample_rate
        self.num_samples=num_samples
        
    def _right_padding_if_req(self,signal):
        signal_len=signal.shape[1]
        if signal_len<self.num_samples:
            missing_pad=self.num_samples-signal_len
            signal=torch.nn.functional.pad(signal,(0,missing_pad))
        return signal
    
    def _cut_if_req(self,signal):
        signal_len=signal.shape[1]
        if signal_len>self.num_samples:
            signal=signal[:,0:self.num_samples]
        return signal
    
    def _resample_if_req(self,signal,sr):
        if sr!=self.target_sample_rate:
            resampler=torchaudio.transforms.Resample(sr,self.target_sample_rate)
            signal=resampler(signal)
        return signal
    
    def _mix_down_if_req(self,signal):
        if signal.shape[0]>1:
            signal=torch.mean(signal,dim=0,keepdim=True)
        return signal
    
    def _get_audio_file_path(self,idx):
        return os.path.join(self.audio_path,os.listdir(self.audio_path)[idx])
    
    def _create_chunks(self,signal,sr,chunk_size):
        return torch.split(signal,chunk_size*sr,dim=1)
    
    def __getitem__(self,idx):
        audio_file_path=self._get_audio_file_path(idx)
        signal,sr=torchaudio.load(audio_file_path)
        signal=signal.to(self.device)
        self.label=self.label.to(self.device)
        signal=self._resample_if_req(signal,sr)
        signal=self._mix_down_if_req(signal)
        signal=self._cut_if_req(signal)
        signal=self._right_padding_if_req(signal)
        signal=self.transformation(signal)
        
        return signal,self.label
    
    
if __name__=="__main__":
    AUDIO_PATH_NP_OTHER="C:/TechManhindra/GRE/UF/Course Work/Machine Learning/Project/Training Dataset/Not Progressive Rock/Not_Progressive_Rock/Other_Songs/"
    AUDIO_PATH_NP_TPOP="C:/TechManhindra/GRE/UF/Course Work/Machine Learning/Project/Training Dataset/Not Progressive Rock/Not_Progressive_Rock/Top_Of_The_Pops/"
    AUDIO_PATH_P="C:/TechManhindra/GRE/UF/Course Work/Machine Learning/Project/Training Dataset/Progressive Rock/Progressive_Rock_Songs/"
    LABEL_NP="Not Progressive Rock"
    LABEL_P="Progressive_Rock"
    SAMPLE_RATE=22050
    NUM_SAMPLES=6336888
    melspectogram=torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE,n_fft=1024,hop_length=512,n_mels=64)
    if torch.cuda.is_available():
        device="cuda"
    else:
        device="cpu"
    print(f"Using device {device}")
    
    non_prog_other=AudioLoader(AUDIO_PATH_NP_OTHER,LABEL_NP,melspectogram,SAMPLE_RATE,NUM_SAMPLES,device)
    non_prog_tpop=AudioLoader(AUDIO_PATH_NP_TPOP,LABEL_NP,melspectogram,SAMPLE_RATE,NUM_SAMPLES,device)
    prog=AudioLoader(AUDIO_PATH_P,LABEL_P,melspectogram,SAMPLE_RATE,NUM_SAMPLES,device)
    
    
"""
"""chunk_samples=10*sr

chunks = [y[i:i+chunk_samples] for i in range(0,len(y),chunk_samples)]

print(y,'\n',sr);

index = ["segement "+i for i in range(0,len(chunks))]

df=pd.DataFrame(chunks,index=index)

print(df)"""

#df.to_csv("C:\\TechManhindra\\GRE\\UF\\Course Work\\Machine Learning\\Project\Training Dataset\\Progressive Rock\\Progressive_Rock_Songs_CSV\\01 - Birds of Fire.csv")

#S=np.abs(librosa.stft(y);

#print(S)

#cent1=librosa.feature.spectral_centroids(y=y,sr=sr)

#S, phase=librosa.magphase(librosa.stft(y=y))

#cent2=librosa.feature.spectral_centroids(S=S)

#plt.plot(S)
y1,sr1=librosa.load("/home/vivek/Project/Training Dataset/Not_Progressive_Rock/Other_Songs/(06) [Alice Cooper] No More Mr. Nice Guy.mp3")

S=librosa.feature.melspectrogram(y=y1,sr=sr1,n_mels=128)

print(S.shape)

fig, ax = plt.subplots()
"""
S_dB = librosa.power_to_db(S, ref=np.max)
img = librosa.display.specshow(S_dB, x_axis='time',y_axis='mel', sr=sr1, ax=ax)
fig.colorbar(img, ax=ax, format='%+2.0f dB')
ax.set(title='Mel-frequency spectrogram')

plt.savefig("/home/vivek/Project/mel-spectrogram.png")

img = librosa.display.specshow(librosa.amplitude_to_db(S,ref=np.max),y_axis="linear",x_axis="m",ax=ax)
ax.set_title("Spectrogram")
fig.colorbar(img, ax=ax, format="%+2.0f dB")
plt.savefig("/home/vivek/Project/spectrogram.png")"""
"""
chromagram=librosa.feature.chroma_stft(y=y1,sr=sr1)
plt.figure(figsize=(10, 4))
librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', cmap='coolwarm')
plt.title('Chromagram')
plt.colorbar()
plt.savefig("/home/vivek/Project/chromagram.png")"""

"""D = np.abs(librosa.stft(y1))

# Convert to decibels (log scale)
spectrogram_db = librosa.amplitude_to_db(D, ref=np.max)

# Plot spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(spectrogram_db, sr=sr1, x_axis='time', y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')
plt.savefig("/home/vivek/Project/spectrogram.png")"""
"""
spectrogram = librosa.feature.melspectrogram(y=y1, sr=sr1)

# Convert to decibels (log scale)
spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

# Plot spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(spectrogram_db, x_axis='time', y_axis='mel', sr=sr1, cmap='viridis')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram')
plt.savefig("/home/vivek/Project/mel-spectrogram.png")
"""

onset_strength = librosa.onset.onset_strength(y=y1, sr=sr1)

# Plot onset strength
plt.figure(figsize=(10, 4))
plt.plot(librosa.times_like(onset_strength), onset_strength, label='Onset Strength')
plt.xlabel('Time (s)')
plt.ylabel('Onset Strength')
plt.title('Onset Strength')
plt.legend()
plt.savefig("/home/vivek/Project/onset_strength.png")