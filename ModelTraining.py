# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 12:06:04 2024

@author: vicky
"""
import time
start_time=time.time()

from AudioLoader import train_dl,valid_dl,test_dl,other_dl,non_prog_other,non_prog_tpop,prog,test_prog,test_non_prog,test_other
from CNNModel import CNNModel
import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score,roc_curve,auc
import numpy as np
import math
import copy
import os

def WTA_Other(preds,mapper):
    song_preds=torch.empty((0),dtype=torch.float32)
    list=os.listdir("/home/vivek/Project/Test Dataset/Other/")
    prev=0
    for key,value in mapper.items():
        pred=preds[prev:key,:]
        ones=(
            pred==1.0
        ).sum(dim=0)

        threshold=math.floor((key-prev)/2)
        if ones>threshold:
            song_preds=torch.cat((song_preds,torch.tensor([1.0])))
        else:
            song_preds=torch.cat((song_preds,torch.tensor([0.0])))
        
        if song_preds[value]==1.0:
            label="Progressive Rock"
        else:
            label="Not Progressive Rock"

        prev=key        
        print(f"{list[value]} : {label}")



def WTA(preds,target_preds,mappers,cutoffs,path=[]):
    num_songs=0
    song_preds=torch.empty((0),dtype=torch.float32)
    song_target=torch.empty((0),dtype=torch.float32)
    #print(song_preds)
    #print(song_target)
    for i in range(len(mappers)):
        if path != []:
            list=os.listdir(path[i])
        mapper=mappers[i]
        cutoff=cutoffs[i]
        prev=0
        target_pred=target_preds[i]
        #print(target_pred)
        for key, value in mapper.items():
            if cutoff==-1 or key<cutoff:
                pred=preds[prev:key,:]
                ones=(
                    pred==1.0
                ).sum(dim=0)

                threshold=math.floor((key-prev)/2)
                if ones>threshold:
                    song_preds=torch.cat((song_preds,torch.tensor([1.0])))
                else:
                    song_preds=torch.cat((song_preds,torch.tensor([0.0])))
                song_target=torch.cat((song_target,torch.tensor([target_pred])))
                prev=key
                num_songs+=1
                if path != []:
                    label_pred=""
                    label_target=""
                    #print(song_preds[-1])
                    if song_preds[-1]==1.0:
                        label_pred="Progressive Rock"
                    else:
                        label_pred="Not Progressive Rock"
                    #print(song_target[-1])
                    if song_target[-1]==1.0:
                        label_target="Progressive Rock"
                    else:
                        label_target="Not Progressive Rock"
                    print(f"{list[value]} :  Predicted Label: {label_pred} True Label: {label_target}")
            else:
                break


    """"
    prev=0
    for key, value in non_oth_mapper.items():
        if key<2352 :
            pred=valid_preds[prev:key,:]
            pred=pred.round()
            ones=(
                pred==1.0
            ).sum(dim=0)

            threshold=math.ceil(key/2)
            if ones>threshold:
                song_preds=torch.cat((song_preds,torch.tensor([1.0])))
            else:
                song_preds=torch.cat((song_preds,torch.tensor([0.0])))
            song_target=torch.cat((song_target,torch.tensor([0.0])))
            #print(song_preds.shape)
            #print(song_target.shape)
            prev=key
            num_songs+=1
        else:
            break
    prev=0
    for key, value in non_tpop_mapper.items():
        if key<640:
            pred=valid_preds[prev:key,:]
            ones=(
                pred==1.0
            ).sum(dim=0)

            threshold=math.ceil(key/2)
            if ones>threshold:
                song_preds=torch.cat((song_preds,torch.tensor([1.0])))
            else:
                song_preds=torch.cat((song_preds,torch.tensor([0.0])))
            song_target=torch.cat((song_target,torch.tensor([0.0])))
            prev=key
            num_songs+=1
        else:
            break
    prev=0
    for key, value in prog_mapper.items():
        if key<3066 :
            pred=valid_preds[prev:key,:]
            ones=(
                pred==1.0
            ).sum(dim=0)

            threshold=math.ceil(key/2)
            if ones>threshold:
                song_preds=torch.cat((song_preds,torch.tensor([1.0])))
            else:
                song_preds=torch.cat((song_preds,torch.tensor([0.0])))
            song_target=torch.cat((song_target,torch.tensor([1.0])))
            prev=key
            num_songs+=1
        else:
            break"""
    
    song_correct=(
        song_preds==song_target
    ).float().sum()
    #print(song_correct)

    return song_correct,num_songs,song_preds,song_target

def plotConfmat(confmat,path):
    fig,ax=plt.subplots(figsize=(7,7))
    ax.matshow(confmat,cmap=plt.cm.Blues,alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j,y=i,s=confmat[i,j],va="center",ha="center")
    ax.xaxis.set_ticks_position("bottom")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.savefig(path,bbox_inches="tight")

def ROCPlot(tpr,fpr,roc_auc,path):
    plt.figure(figsize=(6,8))
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(path,bbox_inches="tight")

def performmanceMetrics(y_pred,y_target,path):
    confmat=confusion_matrix(y_true=y_target,y_pred=y_pred)
    print(f"Confusion Matrix:  {confmat}")
    plotConfmat(confmat,path+"confusionMatrix.png")
    accuracy=accuracy_score(y_target,y_pred)
    print(f"Accuracy: {accuracy}")
    pre_val=precision_score(y_true=y_target,y_pred=y_pred)
    print(f"Presision: {pre_val}")
    rec_val=recall_score(y_true=y_target,y_pred=y_pred)
    print(f"Recall : {rec_val}")
    f1_val=f1_score(y_true=y_target,y_pred=y_pred)
    print(f"F1 Score : {f1_val}")
    fpr,tpr,thresholds=roc_curve(y_pred,y_target)
    roc_auc=auc(fpr,tpr)
    ROCPlot(tpr,fpr,roc_auc,path+"ROC.png")



def train(model,num_epochs,loss_fn,optimizer,train_dl,valid_dl):
    loss_hist_train=[0]*num_epochs
    accuracy_hist_train=[0]*num_epochs
    loss_hist_valid=[0]*num_epochs
    accuracy_hist_valid=[0]*num_epochs
    accuracy_song_valid=[0]*num_epochs
    best_accuracy=np.float32(0.0)
    best_model_weights=None
    patience=4
    best_model_song_preds=torch.empty((0,1),dtype=torch.float32).to(device)
    best_model_song_target=torch.empty((0,1),dtype=torch.float32).to(device)
    best_model_snipped_preds=torch.empty((0,1),dtype=torch.float32).to(device)
    
    for epoch in range(num_epochs):
        #training
        #print(f"Epoch {epoch} started...")
        model.train()
        for x_batch,y_batch in train_dl:
            #print("batch...")
            pred=model(x_batch)
            loss=loss_fn(pred,y_batch)
            loss.backward()
            
            #max-norm regularization
            torch.nn.utils.clip_grad_norm_(model.parameters(),4.0)

            optimizer.step()
            optimizer.zero_grad()
            loss_hist_train[epoch]+=loss.item()*y_batch.size(0)
            #pred[pred>=0.5]=1
            #pred[pred<0.5]=0
            pred=pred.round()
            #print(pred)
            is_correct=(
                pred==y_batch
                ).float()
            accuracy_hist_train[epoch]+=is_correct.cpu().sum()
        loss_hist_train[epoch]/=len(train_dl.dataset)
        accuracy_hist_train[epoch]/=len(train_dl.dataset)
        
        #inference
        
        model.eval()
        
        valid_preds=torch.empty((0,1),dtype=torch.float32).to(device)
        with torch.no_grad():
            for x_batch,y_batch in valid_dl:
                pred=model(x_batch)
                loss=loss_fn(pred,y_batch)
                loss_hist_valid[epoch] += loss.item()*y_batch.size(0)
                #pred[pred>=0.5]=1
                #pred[pred<0.5]=0
                pred=pred.round()
                valid_preds=torch.cat((valid_preds,pred),0)
                is_correct=(
                    pred==y_batch
                    ).float()
                accuracy_hist_valid[epoch]+=is_correct.cpu().sum()
        #print(valid_preds.shape)
        song_correct,num_songs,song_preds,song_target=WTA(valid_preds,[0.0,0.0,1.0],[non_prog_other.mapper,non_prog_tpop.mapper,prog.mapper],[352,640,3066])
        accuracy_song_valid[epoch]=song_correct/num_songs
        loss_hist_valid[epoch]/=len(valid_dl.dataset)
        accuracy_hist_valid[epoch]/=len(valid_dl.dataset)
        
        print(f"Epoch {epoch+1} train accuracy: {accuracy_hist_train[epoch]:.6f} valid accuracy: {accuracy_hist_valid[epoch]:.6f} valid song accuracy: {accuracy_song_valid[epoch]:.6f}")

        #early stooping
        if accuracy_hist_valid[epoch]>best_accuracy:
            best_accuracy=accuracy_hist_valid[epoch]
            best_model_weights=copy.deepcopy(model.state_dict())
            best_model_song_preds=song_preds
            best_model_song_target=song_target
            best_model_snipped_preds=valid_preds
            patience=4
        else:
            patience-=1
            if patience==0:
                break
    
    song_correct,num_songs,song_preds,song_target=WTA(best_model_snipped_preds,[0.0,0.0,1.0],[non_prog_other.mapper,non_prog_tpop.mapper,prog.mapper],[2352,640,3066],["/home/vivek/Project/Training Dataset/Not_Progressive_Rock/Other_Songs/","/home/vivek/Project/Training Dataset/Not_Progressive_Rock/Top_Of_The_Pops/","/home/vivek/Project/Training Dataset/Progressive_Rock_Songs/"])
    performmanceMetrics(best_model_song_preds,best_model_song_target,"/home/vivek/Project/valid_")
    model.load_state_dict(best_model_weights)

    #Testing

    model.eval()
    test_loss=0
    test_accuracy=0
    test_preds=torch.empty((0,1),dtype=torch.float32).to(device)
    with torch.no_grad():
        for x_batch,y_batch in test_dl:
            pred=model(x_batch)
            loss=loss_fn(pred,y_batch)
            test_loss+= loss.item()*y_batch.size(0)
            #pred[pred>=0.5]=1
            #pred[pred<0.5]=0
            pred=pred.round()
            test_preds=torch.cat((test_preds,pred),0)
            is_correct=(
                pred==y_batch
                ).float()
            test_accuracy+=is_correct.cpu().sum()
        
        test_loss/=len(test_dl.dataset)
        test_accuracy/=len(test_dl.dataset)
        song_correct,num_songs,song_preds,song_target=WTA(test_preds,[0.0,1.0],[test_non_prog.mapper,test_prog.mapper],[-1,-1],["/home/vivek/Project/Test Dataset/Not_Progressive_Rock","/home/vivek/Project/Test Dataset/Progressive Rock Songs"])
        song_accuracy=song_correct/num_songs

        performmanceMetrics(song_preds,song_target,"/home/vivek/Project/test_")
        print(f"Loss : {test_loss} accuracy : {test_accuracy} song accuracy : {song_accuracy}")

    
    #Post prog rock
    other_preds=torch.empty((0,1),dtype=torch.float32).to(device)
    with torch.no_grad():
        for x_batch,y_batch in other_dl:
            pred=model(x_batch)
            pred=pred.round()
            other_preds=torch.cat((other_preds,pred),0)

        WTA_Other(other_preds,test_other.mapper)
        
    return loss_hist_train,loss_hist_valid,accuracy_hist_train,accuracy_hist_valid,accuracy_song_valid


def visualize(hist):
    x_arr=np.arange(len(hist[0])) + 1
    fig=plt.figure(figsize=(12,4))
    ax=fig.add_subplot(1,2,1)
    ax.plot(x_arr,hist[0],'-o',label='Train loss')
    ax.plot(x_arr,hist[1],'--<',label="Validation loss")
    ax.legend(fontsize=15)
    ax.set_xlabel("Epoch",size=15)
    ax.set_ylabel("Loss",size=15)
    ax=fig.add_subplot(1,2,2)
    ax.plot(x_arr,hist[2],'-o',label='Train acc.')
    ax.plot(x_arr,hist[3],'--<',label="Validation acc.")
    ax.legend(fontsize=15)
    ax.set_xlabel("Epoch",size=15)
    ax.set_ylabel("Accuracy",size=15)
    plt.savefig("/home/vivek/Project/Plots.png")

if __name__ == "__main__":
    if torch.cuda.is_available():
        device="cuda"
    else:
        device="cpu"
    
    print(f"Using device {device}")
    
    model=CNNModel()
    model=model.to(device)
    
    print("Model built")
    
    loss_fn=nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.003,weight_decay=1e-5)
    
    torch.manual_seed(1)
    
    NUM_EPOCHS=50
    
    print("Training started...")
    hist=train(model,NUM_EPOCHS,loss_fn,optimizer,train_dl,valid_dl)

    visualize(hist)
    
    print("Time Required: ",time.time()-start_time)