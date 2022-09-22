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

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras import utils

# from tensorflow import keras


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


def GetDirs(path):
    ''' get subdirectories from path '''
    Dirs=get_immediate_subdirectories(path)
    print('directories ', Dirs)
    return np.sort(Dirs)
       

def TrainImagestoNumpy(path,Dirs,ToSave=0,Mac=0,  Setup=1,Dim=96,NCat=9):
    ''' load all image files from subdirectories and save as  comined numpy array, for each subfolder, excpects separte folder for each category'''

    Numfiles=np.zeros(NCat)
    assert len(Dirs)==len(Numfiles), f'file number folder number mismatch, got {len(Dirs)}'
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





def TestImagestoNumpy(path,Dirs,Dim=96,ToSave=0,Mac=0,  Setup=1):
    ''' load all image files from subdirectories and save as  comined numpy array for each subfolder'''

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
                data = img_to_array(Image)
                ImageArray[cf,:,:,:]=data
            np.save('image_'+str(d)+'_test',ImageArray)
    return numfiles 



# def RunMainExport)
#   if Setup==1:
#       path='PilotSelected'       
#     elif Setup==2:
#     path='Pilot2Selected'
    
    
   
def TrainTestSel(NCat,Dirs,NumTrain,NumTest,NumFiles,Dim,visN=3):
    ''' load numpy array for each category and randomly select training and test set,
    optionally visN examples are visualized from both training and test set '''

    if len(Dirs)!=NCat:
        print(f"WARNING mismatch directory and NCat, first {NCat}. dirs are used")
    TrainX=np.int16(np.zeros((((NumTrain*NCat,Dim,Dim,3)))))
    TestX=np.int16(np.zeros((((NumTest*NCat,Dim,Dim,3)))))
    TrainY=np.zeros(NumTrain*NCat)
    TestY=np.zeros(NumTest*NCat)    
    assert np.min(NumFiles)> NumTrain+NumTest, f"not enough files in at least one for folder, min {NumTrain+NumTest} needed"
    for cd,d in enumerate(Dirs[0:NCat]):
        if visN:
            fig,ax=plt.subplots(nrows=visN,ncols=2)
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
        for v in range(visN):
            ax[v,0].imshow(TrainX[Count_tr_start+v,:,:,:])
            ax[v,1].imshow(TestX[Count_te_start+v,:,:,:])
            ax[v,0].set_title(d+' train'+str(v+1))
            ax[v,1].set_title(d+' test'+str(v+1))
            for h in range(2):
                ax[v,h].set_xticks([])
                ax[v,h].set_yticks([])
        if visN:    
            plt.tight_layout()
        
    return TrainX,TestX,TrainY,TestY



def MakeCat(TrainY,TestY):
    return utils.to_categorical(TrainY),utils.to_categorical(TestY)


def CNN1hidden(dims_train,numcat,kernels=3,nfilt=64):
    '''  set up keras convolutional neural network with one hidden layer '''
    model_CNN = Sequential()  
    model_CNN.add(Conv2D(filters=nfilt, kernel_size=kernels, padding='same', activation='relu', input_shape=(dims_train[1],dims_train[2], dims_train[3]))) 
    model_CNN.add(MaxPooling2D(pool_size=2))
    model_CNN.add(Flatten())
    model_CNN.add(Dense(numcat, activation='softmax'))
    model_CNN.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model_CNN

def CNN2hidden(dims_train,numcat):
    '''  set up keras convolutional neural network with two hidden layers '''
    model_CNN = Sequential()  
    model_CNN.add(Conv2D(filters=32, kernel_size=5, padding='same', activation='relu', input_shape=(dims_train[1],dims_train[2], dims_train[3]))) 
    model_CNN.add(MaxPooling2D(pool_size=2))
    model_CNN.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')) 
    model_CNN.add(MaxPooling2D(pool_size=2))
    model_CNN.add(Flatten())
    model_CNN.add(Dense(numcat, activation='softmax'))
    model_CNN.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    
    return model_CNN



def CNN3hidden(dims_train,numcat):
    '''  set up keras convolutional neural network with two hidden layers '''
    model_CNN = Sequential()  
    model_CNN.add(Conv2D(filters=16, kernel_size=5, padding='same', activation='relu', input_shape=(dims_train[1],dims_train[2], dims_train[3]))) 
    model_CNN.add(MaxPooling2D(pool_size=2))
    model_CNN.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')) 
    model_CNN.add(MaxPooling2D(pool_size=2))
    model_CNN.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')) 
    model_CNN.add(MaxPooling2D(pool_size=2))
    model_CNN.add(Flatten())
    model_CNN.add(Dense(numcat, activation='softmax'))
    model_CNN.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model_CNN







def fitMod(model,X,testx, Y,testy,nepochs=1):
    'fit keras model '
    model.fit(x=X,y=Y, epochs=nepochs, validation_data=(testx,testy))
    return model


def ModPred(model,X):
    ''' make predictions with fitted model '''
    preds=model.predict(X)
    return np.argmax(preds,1)


def GetAccuracies(preds,trueY,NCat):
    ''' get prediciton accuracy for each category '''
    accurs=np.zeros(NCat)    
    Ns=np.zeros(NCat)
    for i in range(NCat):
        accurs[i]=np.sum(preds[trueY==i]==i)
        Ns[i]=np.sum(trueY==i)
    return accurs/Ns


def pipeline(model,X,testX, Y,testY,dims_train,NCat=9,nepochs=3):
    
    assert callable(model)==True,'function input expected'
    if len(np.shape(Y))==1:
        ylong,ylongtest=MakeCat(Y,testY)
        yshort,yshorttest=Y,testY
    elif len(np.shape(Y))==2:
        yshort,yshorttest=np.argmax(Y,1),np.argmax(testY,1)
        ylong,ylongtest=Y,testY
    else:
        print('Y dimensionality issue')
        
    compiled=model(dims_train,NCat)
    fitted=fitMod(compiled,X,testX, ylong,ylongtest, nepochs=nepochs)
    PredTrain=ModPred(fitted,X)
    PredTest=ModPred(fitted,testX)
    acctrain=GetAccuracies(PredTrain,yshort,NCat)
    acctest=GetAccuracies(PredTest,yshorttest,NCat)
    plt.figure()
    plt.plot(acctrain,label='training',color='olive',linewidth=3)
    plt.plot(acctest,label='test',color='salmon',linewidth=3)
    plt.ylabel('accuracy',fontsize=14)
    plt.xlabel('stimulus category',fontsize=14)
    plt.legend()
    plt.title(model.__name__+' trained for '+str(nepochs)+' epochs')
    return fitted,acctrain
    
    
    