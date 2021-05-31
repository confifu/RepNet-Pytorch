import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms

import cv2
import glob
from tqdm import tqdm
import random
from random import randrange, randint
import math, base64, io, os, time
from torch.utils.data import Dataset, DataLoader, ConcatDataset


"""Creates one sequence from each video"""
class BlenderDataset(Dataset):
    
    def __init__(self, parentDir, vidDir, annotDir, frame_per_vid):
        
        self.vidPath = parentDir + '/' + vidDir
        self.annotPath = parentDir + '/' + annotDir

        self.videos =  list(glob.glob(self.vidPath + '/*.mkv'))
        random.shuffle(self.videos)
        self.frame_per_vid = frame_per_vid

    def getFrames(self, path):
        """returns frames"""
    
        frames = []
        cap = cv2.VideoCapture(path)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret is False:
                break
            
            img = Image.fromarray(frame)
            frames.append(img)
        
        cap.release()
        return frames

    def __getitem__(self, index):

        parts = 64//self.frame_per_vid
        nindex = index//parts


        videoFile = self.videos[nindex]
        curFrames = self.getFrames(videoFile)

        sz = curFrames[0].size
        curFrames[0] = Image.new("RGB", sz, (0,0,0))
        curFrames[-1] = Image.new("RGB", sz, (0,0,0))

        Xlist = []
        for img in curFrames:
        
            preprocess = transforms.Compose([
            transforms.Resize((182, 182)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])])
            frameTensor = preprocess(img).unsqueeze(0)
            Xlist.append(frameTensor)
        

        ipart = nindex % parts
        X = torch.cat(Xlist[ipart*self.frame_per_vid:(ipart+1)*self.frame_per_vid])

        annot = self.annotPath + '/' + self.videos[nindex][len(self.vidPath) + 1:-4]
        labels = glob.glob(annot + '/*')

        y = np.load(labels[0])
        y[0] = 0
        y[-1] = 0
        for i in range(len(y)):
            if y[i] >= 32:
                y[i] = 0
        y = torch.FloatTensor(y[ipart*self.frame_per_vid:(ipart+1)*self.frame_per_vid]).unsqueeze(-1)
        
        assert X.shape[0] == self.frame_per_vid, str(X.shape[0]) + " "+str(self.frame_per_vid)
        assert(y.shape[0] == self.frame_per_vid)

        return X, y
        
    def __len__(self):
        return len(self.videos) * (64//self.frame_per_vid)

