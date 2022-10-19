# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 14:24:59 2022

@author: aratoj87
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import image
import pandas as pd
import os
from os.path import isfile, join
def SetupType(Setup):
    ''' set setup type '''
    if Setup==1:
        return [0,1,0,1,0,1,0,1]  # alternating onscreen for Setup 1
    elif Setup==2:
        return [1,0,1,0,1,0,1,0]  # alternating onscreen for Setup 2


def LoadRaw(path,Vis=0, Verb=1,ax=0,maxVis=5):
    ''' load raw world video from mobile eye-tracker'''
    
    cap= cv2.VideoCapture(path+'world_raw.mp4')
    FrameN=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if Verb:
        print('num frames= ',FrameN)
    ImagesRaw=[]
    cplot=0
    for i in range(FrameN):
        ret, frame = cap.read()
        RGB_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # convert from BGR to RGB colorspace
        ImagesRaw.append(RGB_img)
        if Vis:
            if i%100==0 and cplot<maxVis:
                ax[cplot,0].imshow(RGB_img)
                ax[cplot,0].set_xticks([])
                ax[cplot,0].set_yticks([])
                cplot+=1
    cap.release()
    return ImagesRaw

def LoadGazeVid(path,Vis=0, Verb=1,ax=0,maxVis=5):
    ''' load gaze world video from mobile eye-tracker  - contains overlaid gaze location'''
    
    cap= cv2.VideoCapture(path+'world.mp4')
    FrameN=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cplot=0
    if Verb:
        print('num frames= ',FrameN)
    Images=[]
    for i in range(FrameN):
        ret, frame = cap.read()
        RGB_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # convert from BGR to RGB colorspace
        Images.append(RGB_img)
        if Vis:
            if i%100==0 and cplot<maxVis:
               
                ax[cplot,1].imshow(RGB_img)
                ax[cplot,1].set_xticks([])
                ax[cplot,1].set_yticks([])
                #plt.show()
                cplot+=1
    cap.release()
    return Images
def LoadVid(path,Vis=0, Verb=1,ax=0,maxVis=5,filename='world.mp4'):
    ''' load gaze world video from mobile eye-tracker  - contains overlaid gaze location'''
    cap= cv2.VideoCapture(path+filename)
    FrameN=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cplot=0
    if Verb:
        print('num frames= ',FrameN)
    Images=[]
    for i in range(FrameN):
        ret, frame = cap.read()
        RGB_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # convert from BGR to RGB colorspace
        Images.append(RGB_img)
        if Vis:
            if i%100==0 and cplot<maxVis:
                ax[cplot].imshow(RGB_img)
                ax[cplot].set_xticks([])
                ax[cplot].set_yticks([])
                #plt.show()
                cplot+=1
    cap.release()
    return Images


def GetGazIdx(ind,IdxConf,Gaze,xPix,yPix):
    Idx=Gaze['world_index']==ind  #Gaze['world_index']==i)
    Idx=[(Idx==True) & (IdxConf==True)][0]  #
    Idx=np.nonzero(Idx.to_numpy()==True)[0]
    if np.sum(np.isfinite(Idx))>0:
      #  print(Idx)
        X=np.round(Gaze['norm_pos_x'][Idx].to_numpy()*xPix)
        Y=np.round(Gaze['norm_pos_y'][Idx].to_numpy()*yPix)
    
       # print(i,'length',np.sum(Idx))
    else:
        X,Y=np.NAN,np.NAN
        print(ind,'not valid')
    return X,Y,np.mean(X),np.mean(Y),len(Idx)


def CutImagebyGaze(MyImage,X,Y,cutsize,xPix,yPix):
    ''' cut image around gaze location, with cutsize*2 = size of rectangle'''
    if Y>cutsize and Y<yPix-cutsize and X>cutsize and X<xPix-cutsize:
        ImageCut=MyImage[yPix-(Y+cutsize):yPix-(Y-cutsize),:][:,X-cutsize:X+cutsize]
    else:
        ImageCut=np.NAN
    return ImageCut
      
def ExportFrames(imraw, Gaze,xPix,yPix,path,filename, CutSize=48,ToSave=1,imagesgaze=[], Test=1,Each=0):
    ''' imraw: raw world video
        imagesgaze: gaze containing video
        Gaze: gaze csv file
        path: path to save files
        filename to save files (frame number added automatically)
        CutSize: output size (pixels), size around gaze location, so 48 results in output 96*96
        ToSave: whether output is saved
        Each:  each gaze sample within each frame has an image exported- much more images
        
        '''
    if len(imagesgaze)>0:
        assert np.abs(len(imraw)-len(imagesgaze))<2,'unequal videos'
    FrameN=len(imraw)
    print('n frames to exp: ',len(imraw))
    IdxConf=Gaze['confidence']>.6
    print('Confident Gaze used',np.sum(IdxConf))
    print('Uncertain Gaze unused',np.sum(IdxConf==False))
    FrameStart=Gaze['world_index'].iloc[0]
    FrameEnd=Gaze['world_index'].iloc[-1]
    FrameFixs=np.zeros(FrameN)
    
    if ToSave:
        if os.path.exists(path)==False:
            os.mkdir(path)
    for ci,i in enumerate(np.arange(FrameStart,FrameEnd)):
        X,Y,Xmean,Ymean,FrameFixs[ci]=GetGazIdx(i,IdxConf,Gaze,xPix,yPix)
        if np.isfinite(Xmean): # only if valid gaze in frame
            if Each:   # export each sample from frame
                cc=0  # counter for file name 
                for x,y in zip(X,Y):
                    cc+=1  
                    if np.isfinite(x) and np.isfinite(y):  # only valid sample
                        ImCut=CutImagebyGaze(imraw[ci],int(x),int(y),CutSize,xPix,yPix)
                        if  np.sum(np.isfinite(ImCut))>0 and ToSave:
                            if Test:
                                if i%80==0:  # every 80th frame is saved
                                    image.imsave(path+'frame'+str(i)+'_sample'+str(cc)+'_S'+filename+'.jpg', ImCut)                            
                            else:
                                image.imsave(path+'frame'+str(i)+'_sample'+str(cc)+'_S'+filename+'.jpg', ImCut)                            

                        
            else:  # export average of frame
                ImCut=CutImagebyGaze(imraw[ci],int(Xmean),int(Ymean),CutSize,xPix,yPix)  # cut image
                if np.sum(np.isfinite(ImCut))>0:         
                    if ToSave:  
                        if Test: 
                            if i%50==0:  # every 50th frame is saved
                                image.imsave(path+filename+'_frame'+str(i)+'.jpg', ImCut)                            
                        else:
                            image.imsave(path+filename+'_frame'+str(i)+'.jpg', ImCut)
                    if i%50==0:  # visualize every 50th stimulus
                        plt.figure()
                        plt.subplot(1,2,1)
                        plt.xticks([])
                        plt.yticks([])
                        plt.title('Full world')
                        if len(imagesgaze)>0:
                            plt.imshow(imagesgaze[ci])
                        else:
                            plt.imshow(imraw[ci])
                        plt.subplot(1,2,2)
                        plt.imshow(ImCut)
                        plt.xticks([])
                        plt.yticks([])
    return


def MainTrain(PathFrom,PathTo,ToSave,Vis,Setup,CutSize=48,Mac=0,nvis=5,Test=1):
    Stim=np.arange(1,9)
    OnScreen=SetupType(Setup)
    for stim,onscreen in zip(Stim,OnScreen):
        print(onscreen)
        print(stim)
        filename=str(stim)+'_'+str(CutSize)
        path=PathFrom+str(stim)   
        if Mac==0:
            pathTO=PathTo+'\\'+str(stim)
        else:
            pathTO=PathTo+'/'+str(stim)

        if onscreen:
            path+='s'
            filename+='s'
            pathTO+='s'
        if Mac==0:    
            path+='\\'
            pathTO+='\\'
        else:
            path+='/'
            pathTO+='\\'
        gaze=pd.read_csv(path+'gaze_positions.csv')
        if Vis:
            fig,ax=plt.subplots(nrows=nvis,ncols=2)
        imraw=LoadRaw(path,Vis=Vis, Verb=1,ax=ax,maxVis=nvis)
        imgaze=LoadGazeVid(path,Vis=Vis, Verb=1,ax=ax,maxVis=nvis)
        if Vis:
            plt.show()
        Dims=np.shape(imraw[0])
        xPix=Dims[1]
        yPix=Dims[0]
    
        ExportFrames(imraw, gaze,xPix,yPix,pathTO,filename, CutSize=CutSize,ToSave=ToSave,imagesgaze=imgaze, Test=Test)
    return
    


def MainTest(PathFrom,PathTo,ToSave,Vis,Setup,CutSize=48,Mac=0,nvis=5,Test=1,filename='subjx',EachSample=0):
    ''' set test to 0 to save all images, othwerwise only every 50th image is saved'''
    print('Loading from: ',PathFrom)
    print('Exportin to: ',PathTo)
    if EachSample:
        print('!!Exporting each sample')
    gaze=pd.read_csv(PathFrom+'gaze_positions.csv')
    print('Gaze Shape: ',np.shape(gaze))
    if Vis:
        fig,ax=plt.subplots(nrows=nvis)
    images=LoadVid(PathFrom,Vis=Vis, Verb=1,ax=ax,maxVis=nvis)
    print('length video: ', len(images))
    if Vis:
        plt.show()
    Dims=np.shape(images[0])
    xPix=Dims[1]
    yPix=Dims[0]

    ExportFrames(images, gaze,xPix,yPix,PathTo,filename, CutSize=CutSize,ToSave=ToSave,imagesgaze=[],Test=Test,Each=EachSample)
    return


def ReadIm(path):
    images = [f for f in os.listdir(path) if isfile(join(path, f))]
    print('files found: ',len(images))
    im1=plt.imread(path+images[0])
    np.shape(im1)
    ImArray=np.zeros(((len(images),np.shape(im1)[0],np.shape(im1)[1],np.shape(im1)[2])),dtype='uint8')
    for  ci,im in enumerate(images):
        ImArray[ci,:,:,:]=plt.imread(path+im)
    return ImArray, images


def ParseNames(impath):
  #  image.imsave(path+'frame'+str(i)+'_sample'+str(cc)+'_S'+filename+'.jpg', ImCut)                            
    FrameN=np.zeros(len(impath),dtype=int)
    for ci,im in enumerate(impath):
        FrameN[ci]=im[im.find('frame')+5:im.find('_')]
    return FrameN
    
