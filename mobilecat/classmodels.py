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

from skimage.util import random_noise
from skimage import transform

import time



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
    switched=0
    assert len(Dirs)==len(Numfiles), f'file number folder number mismatch, got {len(Dirs)}'
    for cd,d in enumerate(Dirs):
        if Mac==1:
            files=os.listdir(path+'/'+d)
        else:
            files=os.listdir(path+'\\'+d)
        for f in files:
            if f.find('jpg')>-1:
                Numfiles[cd]+=1
        if ToSave==1:
            ImageArray=np.zeros((((int(Numfiles[cd]),Dim,Dim,3))),dtype='int16')
            for cf,f in enumerate(files):
                if f.find('jpg')>-1:
                    if Mac==1:
                        Image=plt.imread(path+'/'+d+'/'+f)
                    else:
                        Image=plt.imread(path+'\\'+d+'\\'+f)
                    if Setup==1 and f.find('Image')>-1:   # color switching, since some of the training images were initally saved incorrectly BGR- RGB switch
                        switched+=1  # count number of switch
                        Image=cv2.cvtColor(Image, cv2.COLOR_BGR2RGB)  
                    #image = load_img(path+'\\'+d+'/'+f)
                    data = img_to_array(Image)
                    ImageArray[cf,:,:,:]=data
            np.save(path+'image_'+str(d),ImageArray)
    print('n color switched',switched)
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
            np.save(path+'image_'+str(d)+'_test',ImageArray)
    return numfiles 



# def RunMainExport)
#   if Setup==1:
#       path='PilotSelected'       
#     elif Setup==2:
#     path='Pilot2Selected'
    
    
   
def TrainTestSel(NCat,Dirs,NumTrain,NumTest,Dim,visN=3,path=''):
    ''' load numpy array for each category and randomly select training and test set,
    optionally visN examples are visualized from both training and test set '''

    if len(Dirs)!=NCat:
        print(f"WARNING mismatch directory and NCat, first {NCat}. dirs are used")
    TrainX=np.int16(np.zeros((((NumTrain*NCat,Dim,Dim,3)))))
    TestX=np.int16(np.zeros((((NumTest*NCat,Dim,Dim,3)))))
    TrainY=np.zeros(NumTrain*NCat)
    TestY=np.zeros(NumTest*NCat)    
    f"not enough files in at least one for folder, min {NumTrain+NumTest} needed"
    for cd,d in enumerate(Dirs):
        if visN:
            fig,ax=plt.subplots(nrows=visN,ncols=2)
        print('load: ',path+'image_'+str(d)+'.npy')
        ImageArrayL=np.load(path+'image_'+str(d)+'.npy')
        Numim=np.shape(ImageArrayL)[0]
        assert Numim >= NumTrain+NumTest, f'not enough imqages, test+train= {NumTrain+NumTest}'
        
        print(cd,d,'Num images',Numim)

        Rand=np.intp(np.random.permutation(np.arange(Numim)))
        TrainIdx=Rand[0:NumTrain]
        TestIdx=Rand[NumTrain:NumTrain+NumTest]
        Count_tr_start=cd*NumTrain
        Count_te_start=cd*NumTest
        TrainX[Count_tr_start:Count_tr_start+NumTrain,:,:,:]=ImageArrayL[TrainIdx,:,:,:]
        TestX[Count_te_start:Count_te_start+NumTest,:,:,:]=ImageArrayL[TestIdx,:,:,:]
        TrainY[Count_tr_start:Count_tr_start+NumTrain]=cd
        TestY[Count_te_start:Count_te_start+NumTest]=cd
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



def TestSelRest(NCat,Dirs,NumTest,Dim,path=''):
    ''' load numpy array for each category and randomly select test data,
   the rest of files kept as traninig rate (gfrom which actual training data is randomly selected '''

    if len(Dirs)!=NCat:
        print(f"WARNING mismatch directory and NCat, first {NCat}. dirs are used")
    TestX=np.int16(np.zeros((((NumTest*NCat,Dim,Dim,3)))))
    TrainXrest=[]
    TestY=np.zeros(NumTest*NCat)    
  
    for cd,d in enumerate(Dirs):
      
        ImageArrayL=np.load(path+'image_'+str(d)+'.npy')
        print(cd,d,'Num images: ',np.shape(ImageArrayL)[0] )
        Rand=np.random.permutation(np.arange(np.shape(ImageArrayL)[0]))
        TestIdx=np.intp(Rand[0:NumTest])
        #print(TestIdx,type(TestIdx))
        TrainIdx=np.intp(Rand[NumTest:])
        Count_te_start=cd*NumTest
        #print(type(Count_te_start+NumTest))
        TestX[Count_te_start:Count_te_start+NumTest,:,:,:]=ImageArrayL[TestIdx,:,:,:]
        TestY[Count_te_start:Count_te_start+NumTest]=cd
        TrainXrest.append(ImageArrayL[TrainIdx,:,:,:])
    return TrainXrest,TestX,TestY

def SelTrain(TrainXrest,NumTrain,NCat,Dim):
    ''' load numpy array for each category and randomly select training and test set,
    optionally visN examples are visualized from both training and test set '''

    TrainX=np.int16(np.zeros((((NumTrain*NCat,Dim,Dim,3)))))
    TrainY=np.zeros(NumTrain*NCat)
    for cd in range(NCat):  
        N=len(TrainXrest[cd])
        print(cd,'num: ', N)
        Rand=np.intp(np.random.permutation(np.arange(N)))
        TrainIdx=Rand[0:NumTrain]
        Count_tr_start=cd*NumTrain
        TrainY[Count_tr_start:Count_tr_start+NumTrain]=cd
        TrainX[Count_tr_start:Count_tr_start+NumTrain]=TrainXrest[cd][TrainIdx]
    return TrainX,TrainY





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



def CNN4hidden(dims_train,numcat):
    '''  set up keras convolutional neural network with two hidden layers '''
    model_CNN = Sequential()  
    model_CNN.add(Conv2D(filters=16, kernel_size=5, padding='same', activation='relu', input_shape=(dims_train[1],dims_train[2], dims_train[3]))) 
    model_CNN.add(MaxPooling2D(pool_size=2))
    model_CNN.add(Conv2D(filters=32, kernel_size=5, padding='same', activation='relu')) 
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


def GetConfMat(preds,truey,NCat):
    MyConfMat=np.zeros((NCat,NCat))
    for i in range(NCat):
        for j in range(NCat):
            MyConfMat[j,i]=np.sum(preds[truey==i]==j)
    return MyConfMat

def VisConfMat(confmat,NCat,labels=0,title=''):
    ''' visualize confusion matrix'''

    plt.pcolor(confmat)
    plt.colorbar()
    if type(labels)!=int:
        plt.xticks(np.arange(NCat)+.5,labels[0:NCat],fontsize=12)
        plt.yticks(np.arange(NCat)+.5,labels[0:NCat],fontsize=12)
    plt.xlabel('True category',fontsize=14)
    plt.ylabel('Predicted category',fontsize=14)
    plt.title(title,fontsize=16)
    return 

def VisAccuracy(title,acctrain,acctest,NCat,nepochs):
    ''' visualize accuracy for training and test for each stimulus type '''
    plt.figure()
    plt.plot(acctrain,label='training',color='olive',linewidth=3)
    plt.plot(acctest,label='test',color='salmon',linewidth=3)
    plt.plot([0,NCat-1],[1/NCat,1/NCat],color='k',linestyle='--',linewidth=3)
    plt.plot([0,NCat-1],[1,1],color='k',linestyle='--',linewidth=3)
    plt.text(1,1/NCat+.02,'chance',fontsize=11)
    plt.ylabel('accuracy',fontsize=14)
    plt.xlabel('stimulus category',fontsize=14)
    plt.legend()
    plt.title(title+' trained for '+str(nepochs)+' epochs-accuracy test: '+str(np.round(np.mean(acctest),2)))
    plt.show()
    return 


def pipeline(model,X,testX, Y,testY,dims_train,NCat=9,nepochs=3,catnames=''):
    ''' pipeline for CNN compile, fitting, prediction, accuracy by category'''
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
    VisAccuracy(model.__name__,acctrain,acctest,NCat,nepochs)

    mattrain=GetConfMat(PredTrain,yshort,NCat)
    mattest=GetConfMat(PredTest,yshorttest,NCat)
    
    VisMats(mattrain,mattest,NCat,catnames,model.__name__)
    return fitted,acctrain,acctest
    
    
def pipelineTrainRand(model,TrainXrest,testX, testY,NTrain,dim,NCat=9,nepochs=3,catnames=''):
    ''' pipeline for CNN compile, fitting, prediction, accuracy by category
    training data is selected randomly at each epoch from all availible images'''
    assert callable(model)==True,'function input expected'
  
    
    compiled=model([0,dim,dim,3],NCat)
    
    print('model compiled')
    
  

    ax=Learnplot(model.__name__,nepochs,trainname='rand trained')

    for n in range(nepochs):
        start = time.time()
        print(f'epoch number {n}')
        X,Y=SelTrain(TrainXrest,NTrain,NCat,dim)
 
        if len(np.shape(Y))==1:
            ylong,ylongtest=MakeCat(Y,testY)
            yshort,yshorttest=Y,testY
        elif len(np.shape(Y))==2:
            yshort,yshorttest=np.argmax(Y,1),np.argmax(testY,1)
            ylong,ylongtest=Y,testY
        else:
            print('Y dimensionality issue')
        if n==0:
            fitted=fitMod(compiled,X,testX, ylong,ylongtest, nepochs=1)
        else:
            fitted=fitMod(fitted,X,testX, ylong,ylongtest, nepochs=1)
            
        PredTrain=ModPred(fitted,X)
        PredTest=ModPred(fitted,testX)
        acctrain=GetAccuracies(PredTrain,yshort,NCat)
        acctest=GetAccuracies(PredTest,yshorttest,NCat)
        ax.scatter(n+1,np.mean(acctrain),color='g')
        ax.scatter(n+1,np.mean(acctest),color='salmon')
        ax.legend(['training','test'])

        end=time.time()
        print('pipeline epoch length: ',np.round(end - start,3),' sec')

    VisAccuracy(model.__name__,acctrain,acctest,NCat,nepochs)

    mattrain=GetConfMat(PredTrain,yshort,NCat)
    mattest=GetConfMat(PredTest,yshorttest,NCat)
    
    VisMats(mattrain,mattest,NCat,catnames,model.__name__,trainingName=' random')
    return fitted,acctrain,acctest
    
def pipelineTrainAugment(model,TrainXrest,testX, testY,NTrain,dim,NCat=9,nepochs=3,catnames=''):
    ''' pipeline for CNN compile, fitting, prediction, accuracy by category
    training data is selected randomly at each epoch from all availible images'''
    assert callable(model)==True,'function input expected'
  
    
    compiled=model([0,dim,dim,3],NCat)
    
    print('model compiled')
    
   

    ax=Learnplot(model.__name__,nepochs,trainname=' augmented trained')
    for n in range(nepochs):
        print(f'epoch number {n}')
        X,Y=SelTrain(TrainXrest,NTrain,NCat,dim)
        X,Y=augmentXY(X,Y)
        if len(np.shape(Y))==1:
            ylong,ylongtest=MakeCat(Y,testY)
            yshort,yshorttest=Y,testY
        elif len(np.shape(Y))==2:
            yshort,yshorttest=np.argmax(Y,1),np.argmax(testY,1)
            ylong,ylongtest=Y,testY
        else:
            print('Y dimensionality issue')
        if n==0:
            fitted=fitMod(compiled,X,testX, ylong,ylongtest, nepochs=1)
        else:
            fitted=fitMod(fitted,X,testX, ylong,ylongtest, nepochs=1)
            
        PredTrain=ModPred(fitted,X)
        PredTest=ModPred(fitted,testX)
        acctrain=GetAccuracies(PredTrain,yshort,NCat)
        acctest=GetAccuracies(PredTest,yshorttest,NCat)
        ax.scatter(n+1,np.mean(acctrain),color='olive')
        ax.scatter(n+1,np.mean(acctest),color='salmon')
        ax.legend(['training','test'])
        
    VisAccuracy(model.__name__,acctrain,acctest,NCat,nepochs)
    mattrain=GetConfMat(PredTrain,yshort,NCat)
    mattest=GetConfMat(PredTest,yshorttest,NCat)
    VisMats(mattrain,mattest,NCat,catnames,model.__name__,trainingName=' augmented')
    return fitted,acctrain,acctest        
    
def VisMats(c1,c2,NCat,catnames,suptitle,trainingName=''):
    ''' visualize confucion matrix training and test'''
    plt.figure(figsize=(7,3))
    plt.subplot(1,2,1)
    VisConfMat(c1,NCat,labels=catnames,title='training'+trainingName)
    plt.subplot(1,2,2)
    VisConfMat(c2,NCat,labels=catnames,title='test')
    plt.suptitle(suptitle)
    plt.tight_layout()
    plt.show()


def Learnplot(modname,nepoch,trainname=''):
    ''' plot for deep learning training progress '''
    fig,ax=plt.subplots()
    ax.set_xlabel('N epochs trained')
    ax.set_ylabel('accuracy')
    ax.set_xlim([.5,nepoch+.5])
    ax.set_ylim([.3,1])
    ax.grid(True)
    ax.set_title(modname+trainname)

    return ax



def augmentXY(X,Y):
    TrainXAug1=AugmentBrightness(X,MinBr=.5,MaxBr=1.2)
    #TrainXAug2=AugmentRotation(X)
    TrainXAug2=AugmentShear(X)
    Xaug=np.concatenate((TrainXAug1,TrainXAug2),axis=0)   
    Yaug=np.concatenate((Y,Y))
    return Xaug, Yaug


def AugmentBrightness(ToAugment,MinBr=.6,MaxBr=1.2):
    NImage=np.shape(ToAugment)[0]
    Rand=np.random.uniform(MinBr,MaxBr,NImage)
    Augmented=np.zeros_like(ToAugment)
    for cim,randman in enumerate(Rand):
        Augmented[cim,:,:,:]=ToAugment[cim,:,:,:]*randman
    Augmented[Augmented<0]=0
    Augmented[Augmented>255]=255
    Augmented=np.intp(Augmented)
    return Augmented

def AugmentSaltPNoise(ToAugment,prob=.05):
    NImage=np.shape(ToAugment)[0]
    Augmented=np.zeros_like(ToAugment)
    for cim  in range(NImage):
        Augmented[cim,:,:,:]=random_noise(ToAugment[cim,:,:,:]/255,mode='s&p',amount=prob)
    Augmented=np.intp(Augmented*255)
    return Augmented
def AugmentRotation(ToAugment,MinR=-np.pi/6,MaxR=np.pi/6):
    NImage=np.shape(ToAugment)[0]
    Augmented=np.zeros_like(ToAugment)
    Rand=np.random.uniform(MinR,MaxR,NImage)
    for cim,randman in enumerate(Rand):
        tf=transform.SimilarityTransform(rotation=randman)
        Augmented[cim,:,:,:]=transform.warp(ToAugment[cim,:,:,:],tf)
    #Augmented=np.intp(Augmented*255)
    return Augmented

def AugmentShear(ToAugment,MinR=-np.pi/6,MaxR=np.pi/6):
    NImage=np.shape(ToAugment)[0]
    Augmented=np.zeros_like(ToAugment)
    Rand=np.random.uniform(MinR,MaxR,NImage)
    for cim,randman in enumerate(Rand):
        tf=transform.AffineTransform(shear=randman)
        Augmented[cim,:,:,:]=transform.warp(ToAugment[cim,:,:,:],tf)
    #Augmented=np.intp(Augmented*255)
    return Augmented


def FramePreds(frameN,preds,Vis=1):  
    ''' for an array of frame numbers and predictions;
    calcualte max and probability of max predictions'''
    frames, frameCounts=np.unique(frameN, return_counts=True)
    maxCounts=np.zeros(len(frames),dtype=int)
    maxPreds=np.zeros(len(frames),dtype=int)
    for cf,f in enumerate(frames):
        values, counts = np.unique(preds[frameN==f], return_counts=True)
        ind = np.argmax(counts)
        maxCounts[cf]=np.max(counts)
        maxPreds[cf]=values[ind]
    if Vis:           
        nc=np.unique(preds)
        plt.figure(figsize=(10,3))
        plt.scatter(frames,maxPreds,alpha=maxCounts/frameCounts )
        plt.xlabel('frame number',fontsize=15)
        plt.yticks(np.arange(len(nc)),nc)
        plt.ylabel('predicted category',fontsize=15)
    return maxPreds,maxCounts/frameCounts  


def FramePredsV(frameN,preds,alp=.2):  
    ''' for an array of frame numbers and predictions;
    calcualte max and probability of max predictions'''    
    nc=9
    plt.figure(figsize=(10,3))
    plt.scatter(frameN,preds,alpha=alp)
    plt.xlabel('frame number',fontsize=15)
    plt.yticks(np.arange(0,nc))
    plt.ylabel('predicted category',fontsize=15)
     



def VisPredFrames(images,frames,preds,nframe=6,nsamp=10,startF=0):
    '''   visualize model predictions, flexible frame number  and sample number
        figure size adapts to number of frames and samples
        images: array of images
        frames: frame number array (one/image)
        preds: predictions
        startF: starting frame number '''
    fig,ax=plt.subplots(nrows=nframe,ncols=nsamp,figsize=(nsamp*1.3,nframe*1.7))
    frameNs,counts=np.unique(frames,return_counts=True)
    for ccf,cf in enumerate(np.arange(startF,startF+nframe)):
        IdxS=int(np.nonzero(frames==frameNs[cf])[0][0])
        for cs in range(nsamp):
            if cs<=counts[cf]:
                ax[ccf,cs].imshow(images[IdxS+cs,:,:,:])
                ax[ccf,cs].set_title('pred'+str(preds[IdxS+cs]))
                ax[ccf,cs].set_xlabel('frame'+str(frameNs[cf])) 
            ax[ccf,cs].set_xticks([])
            ax[ccf,cs].set_yticks([])



def VisPredFrames2(images,frames,preds,nhor=6,ncols=10,startF=0,stepS=50):
    '''   visualize model predictions, flexible frame number  and sample number
        figure size adapts to number of frames and samples
        images: array of images
        frames: frame number array (one/image)
        preds: predictions
        startF: starting frame number '''
    fig,ax=plt.subplots(nrows=nhor,ncols=ncols,figsize=(nhor, ncols))
    frameNs,counts=np.unique(frames,return_counts=True)
    
    for ccf,cf in enumerate(np.arange(startF,startF+ncols*nhor)):
        nh=int(ccf/ncols)
        nv=np.mod(ccf,ncols)
        ax[nh,nv].imshow(images[startF+ccf,:,:,:])
        ax[nh,nv].set_title('pred'+str(preds[startF+ccf]))
        ax[nh,nv].set_xticks([])
        ax[nh,nv].set_yticks([])
