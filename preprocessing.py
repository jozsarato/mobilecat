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

Stim=np.arange(1,9)

def SetupType(Setup):
    ''' set setup type '''
    if Setup==1:
        return [0,1,0,1,0,1,0,1]  # alternating onscreen for Setup 1
    elif Setup==2:
        return [1,0,1,0,1,0,1,0]  # alternating onscreen for Setup 2


def LoadRaw(path,Vis=0, Verb=1):
    ''' load raw world video from mobile eye-tracker'''
    
    cap= cv2.VideoCapture(path+'/world_raw.mp4')
    FrameN=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if Verb:
        print('num frames= ',FrameN)
    ImagesRaw=[]
    for i in range(FrameN):
        ret, frame = cap.read()
        RGB_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # convert from BGR to RGB colorspace
        ImagesRaw.append(RGB_img)
        if Vis:
            if i%100==0:
                plt.imshow(RGB_img)
                plt.show()
    cap.release()
    return ImagesRaw

def LoadGazeVid(path,Vis=0, Verb=1):
    ''' load gaze world video from mobile eye-tracker  - contains overlaid gaze location'''

    cap= cv2.VideoCapture(path+'/world.mp4')
    FrameN=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if Verb:
        print('num frames= ',FrameN)
    Images=[]
    for i in range(FrameN):
        ret, frame = cap.read()
        RGB_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # convert from BGR to RGB colorspace
        Images.append(RGB_img)
        if Vis:
            if i%100==0:
                plt.imshow(RGB_img)
                plt.show()
    cap.release()
    return Images



def GetGazIdx(ind,IdxConf,Gaze,xPix,yPix):
    Idx=Gaze['world_index']==ind  #Gaze['world_index']==i)
    Idx=[(Idx==True) & (IdxConf==True)][0]  #
    Idx=np.nonzero(Idx.to_numpy()==True)[0]
    if np.sum(np.isfinite(Idx))>0:
      #  print(Idx)
        Xmean=int(np.mean(Gaze['norm_pos_x'][Idx]*xPix))
        Ymean=int(np.mean(Gaze['norm_pos_y'][Idx]*yPix))
       # print(i,'length',np.sum(Idx))
    else:
        Xmean,Ymean=np.NAN,np.NAN
        print(ind,'not valid')
        
    return Xmean,Ymean,len(Idx)

def CutImagebyGaze(MyImage,X,Y,cutsize,xPix,yPix):
    if Y>cutsize and Y<yPix-cutsize and X>cutsize and X<xPix-cutsize:
        ImageCut=MyImage[yPix-(Y+cutsize):yPix-(Y-cutsize),:][:,X-cutsize:X+cutsize]
    else:
        ImageCut=np.NAN
    return ImageCut
      
def ExportFrames(imraw, Gaze,path,filename, CutSize=48,ToSave=1,imagesgaze=[]):
    ''' imraw: raw world video
        imagesgaze: gaze containing video
        Gaze: gaze csv file
        path: path to save files
        filename to save files (frame number added automatically)
        CutSize: output size (pixels), size around gaze location, so 48 results in output 96*96
        ToSave: whether output is saved
        
        '''
    if len(imagesgaze)>0:
        assert np.abs(len(imraw)-len(imagesgaze))<2,'unequal videos'
    FrameN=len(imraw)
    IdxConf=Gaze['confidence']>.6
    FrameFixs=np.zeros(FrameN)
    for i in np.arange(FrameN):
        Xmean,Ymean,FrameFixs[i]=GetGazIdx(i,IdxConf)
        if np.isfinite(Xmean): # only if valid gaze
            ImCut=CutImagebyGaze(imraw[i],Xmean,Ymean,CutSize)  # cut image
            if np.sum(np.isfinite(ImCut))>0:         
                if ToSave:                
                    image.imsave(path+filename+str(i)+'.jpg', ImCut)
                if i%50==0:  # visualize every 50th stimulus
                    plt.figure()
                    plt.subplot(1,2,1)
                    plt.xticks([])
                    plt.yticks([])
                    plt.title('Full world')
                    if len(imagesgaze)>0:
                        plt.imshow(imagesgaze[i])
                    else:
                        plt.imshow(imraw[i])
                    plt.subplot(1,2,2)
                    plt.imshow(ImCut)
                    plt.xticks([])
                    plt.yticks([])
    return


    