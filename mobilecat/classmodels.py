# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 11:33:53 2022

@author: aratoj87
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import image
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import sys
import os



def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


       

def TrainImagestoNumpy(path,ToSave=0,Mac=0,  Setup=1,Dim=96):
    Numfiles=np.zeros(9)
    Dirs=get_immediate_subdirectories(path)
    if  Setup==1:
        Dirs=np.sort(Dirs)
    elif Setup==2: 
        Dirs=np.sort(Dirs)
    print('directories ', Dirs)
    assert len(Dirs)==len(Numfiles),' file number folder number mismatch'
    for cd,d in enumerate(Dirs):
        if Mac==1:
            files=os.listdir(path+'/'+d)
        else:
            files=os.listdir(path+'\\'+d)
            
        Numfiles[cd]=len(files)
        if ToSave==1:
            ImageArray=np.zeros((((int(Numfiles[cd]),Dim,Dim,3))),dtype='int16')
            for cf,f in enumerate(files):
                
                if Mac==1:
                    Image=plt.imread(path+'/'+d+'/'+f)
                else:
                    Image=plt.imread(path+'\\'+d+'/'+f)
                if Setup==1:
                    Image=cv2.cvtColor(Image, cv2.COLOR_BGR2RGB)  
                #image = load_img(path+'\\'+d+'/'+f)
                data = img_to_array(Image)
                ImageArray[cf,:,:,:]=data
            np.save('image_'+str(d)+'_st'+str(Setup),ImageArray)
    return Numfiles 





def TestImagestoNumpy(path,Dim=96,ToSave=0,Mac=0,  Setup=1):
    Dirs=get_immediate_subdirectories(path)
    numfiles=np.zeros(len(Dirs))
    print('directories ', Dirs)
    for cd,d in enumerate(Dirs):
        if Mac==1:
            files=os.listdir(path+'/'+d)
        else:
            files=os.listdir(path+'\\'+d)
        numfiles[cd]=len(files)
        if ToSave==1:
            ImageArray=np.zeros((((int(numfiles[cd]),Dim,Dim,3))),dtype='int16')
            for cf,f in enumerate(files):
                
                if Mac==1:
                    Image=plt.imread(path+'/'+d+'/'+f)
                else:
                    Image=plt.imread(path+'\\'+d+'/'+f)
                if Setup==1:
                    Image=cv2.cvtColor(Image, cv2.COLOR_BGR2RGB)  
                #image = load_img(path+'\\'+d+'/'+f)
                data = img_to_array(Image)
                ImageArray[cf,:,:,:]=data
            np.save('image_'+str(d)+'_test',ImageArray)
    return numfiles 




# def RunMainExport)
#   if Setup==1:
#       path='PilotSelected'       
#     elif Setup==2:
#     path='Pilot2Selected'
    
    
    
NCat=9   # number of stim categories 



def TrainTestSel(NCat,Dirs,NumTrain,NumTest,NumFiles,Dim):
    
    if len(Dirs)!=NCat:
        print(f"WARNING mismatch directory and NCat, first {NCat}. dirs are used")
    TrainX=np.zeros((((NumTrain*NCat,Dim,Dim,3))))
    TestX=np.zeros((((NumTest*NCat,Dim,Dim,3))))
    TrainY=np.zeros(NumTrain*NCat)
    TestY=np.zeros(NumTest*NCat)    
    
    for cd,d in enumerate(Dirs[0:NCat]):
        print(cd,d,'Num images',int(NumFiles[cd]))
        ImageArrayL=np.load('image_'+str(d)+'.npy')
        Rand=np.intp(np.random.permutation(np.arange(NumFiles[cd])))
        TrainIdx=Rand[0:NumTrain]
        TestIdx=Rand[NumTrain:NumTrain+NumTest]
        Count_tr_start=cd*NumTrain
        Count_te_start=cd*NumTest
        TrainX[Count_tr_start:Count_tr_start+NumTrain,:,:,:]=ImageArrayL[TrainIdx,:,:,:]
        TestX[Count_te_start:Count_te_start+NumTest,:,:,:]=ImageArrayL[TestIdx,:,:,:]
        TrainY[Count_tr_start:Count_tr_start+NumTrain]=cd
        TestY[Count_te_start:Count_tr_start+NumTest]=cd
    return TrainX,TestX,TrainY,TestY