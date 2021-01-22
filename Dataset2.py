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
class miniDataset(Dataset):
    
    def __init__(self, df, path_to_video):
        
        self.path = path_to_video
        self.df = df.reset_index()
        self.count = self.df.loc[0, 'count']

    def getFrames(self, path = None):
        """returns frames"""
    
        frames = []
        if path is None:
            path = self.path
        
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
        
        curFrames = self.getFrames()
        
        output_len = min(len(curFrames), randint(44, 64))
                
        newFrames = []
        for i in range(1, output_len + 1):
            newFrames.append(curFrames[i * len(curFrames)//output_len  - 1])

        a = randint(0, 64 - output_len)
        b = 64 - output_len - a
        
        
        randpath = random.choice(glob.glob('drive/MyDrive/PR_Repnet/synthvids/train*.mp4'))
        randFrames = self.getFrames(randpath)
        newRandFrames = []
        for i in range(1, a + b + 1):
            newRandFrames.append(randFrames[i * len(randFrames)//(a+b)  - 1])

        
        finalFrames = newRandFrames[:a]
        finalFrames.extend( newFrames )        
        finalFrames.extend( newRandFrames[a:] )
        
        Xlist = []
        for img in finalFrames:
        
            preprocess = transforms.Compose([
            transforms.Resize((112, 112), 2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            frameTensor = preprocess(img).unsqueeze(0)
            Xlist.append(frameTensor)
        
        X = torch.cat(Xlist)
                          
        y1 = [0 for i in range(0,a)]
        y1.extend([min(output_len//self.count, 31) if not output_len//self.count == 1 else 0 for i in range(0, output_len)])
        
        y1.extend( [ 0 for i in range(0, b)] )
        y1 = torch.LongTensor(y1)                #periodicity
        
        y2 = [0 for i in range(0, a)]
        y2.extend([1 for i in range(0, output_len)])
        y2.extend([0 for i in range(0, b)])
        y2 = torch.BoolTensor(y2).unsqueeze(-1)                #periodicity
        
        return X, y1, y2
        
    def __len__(self):
        return 1
    
class dataset_with_indices(Dataset):

    """
    Modifies the given Dataset class to return a tuple data, target, index
    instead of just data, target.
    """
    
    def __init__(self, ds):
        self.ds = ds

    def __getitem__(self, index):
        X, y1, y2 = self.ds[index]
        return X, y1, y2, index
    
    def getPeriodDist(self):
        arr = np.zeros(32,)
        
        for i in tqdm(range(self.__len__())):
            _, p,_,_ = self.__getitem__(i)
            per = max(p)
            arr[per] += 1
        return arr
    
    def __len__(self):
        return len(self.ds)


def getCombinedDataset2(dfPath, videoDir, videoPrefix):
    df = pd.read_csv(dfPath)
    path_prefix = videoDir + '/' + videoPrefix
    
    files_present = []
    for i in range(0, len(df)):
        path_to_video = path_prefix + str(i) + '.mp4'
        if os.path.exists(path_to_video):
            files_present.append(i)

    df = df.iloc[files_present]
    
    miniDatasetList = []
    for i in range(0, len(df)):
        dfi = df.iloc[[i]]
        path_to_video = path_prefix + str(dfi.index.item()) +'.mp4'
        miniDatasetList.append(miniDataset(dfi, path_to_video))
        
    megaDataset = dataset_with_indices(ConcatDataset(miniDatasetList))
    return megaDataset
