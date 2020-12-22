import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms

import cv2
import math, base64, io, os, time
from torch.utils.data import Dataset, DataLoader, ConcatDataset

class CountixDataset(Dataset):
    
    def __init__(self, df_path, videoPath, filenamePrefix, framesPerVid):
        
        self.framesPerVid = framesPerVid        
        self.df = pd.read_csv(df_path)
        self.path_prefix = videoPath + '/' + filenamePrefix
        
        files_present = []
        for i in range(0, len(self.df)):
            path_to_video = self.path_prefix + str(i) + '.mp4'
            if os.path.exists(path_to_video):
                files_present.append(i)

        self.df = self.df.iloc[files_present]
        self.df = self.df.sort_values(by=['video_id',
                                         'repetition_start',
                                         'repetition_end'])
        self.df = self.df.reset_index()


    def __getitem__(self, index):

        X= self.generateRepVid(index)
        newStart, newEnd = self.getNewRange(index)
        
        id = self.df.loc[index, 'video_id']
        count = self.df.loc[index, 'count']
        start = self.df.loc[index, 'repetition_start']
        end = self.df.loc[index, 'repetition_end']
        
        durationScaleFac = (newEnd-newStart)/self.framesPerVid
        repStartFrame = int((start- newStart)/durationScaleFac)
        repEndFrame = int((end - newStart)/durationScaleFac)
        
     
        #print(start, end, newStart, newEnd, repEndFrame, repStartFrame, count, id)
        periodLength = np.zeros((self.framesPerVid))
        for i in range(self.framesPerVid):
            if repStartFrame < i < repEndFrame:
                periodLength[i] = max(0, (repEndFrame-repStartFrame)//count - 1)      #this would mean the predicted+1 is length
                
        periodicity = np.zeros((self.framesPerVid,1)) 
        for i in range(self.framesPerVid):
            if repStartFrame < i < repEndFrame:
                periodicity[i] = True


        periodLength = torch.LongTensor(periodLength)
        periodicity = torch.FloatTensor(periodicity)
        return X, periodLength, periodicity
    
    def getNewRange(self, index):
        start = self.df.loc[index, 'repetition_start']
        end = self.df.loc[index, 'repetition_end']
        newStart =  -1
        if index - 1 >= 0 and self.df.loc[index-1, 'video_id'] == self.df.loc[index, 'video_id']:
            if int(self.df.loc[index-1, 'repetition_end']) == int(self.df.loc[index, 'repetition_start']):
                newStart=self.df.loc[index-1, 'repetition_end']

        newStart = max([start - 0.5,
                    (newStart if newStart < start else start),
                    self.df.loc[index, 'kinetics_start']])

        newEnd = np.inf   
        if self.df.loc[min(len(self)-1,index+1), 'video_id']==self.df.loc[index, 'video_id']:
            if int(self.df.loc[index, 'repetition_end'])==int(self.df.loc[min(len(self)-1,index+1), 'repetition_start']):
                newEnd=self.df.loc[min(len(self)-1,index+1), 'repetition_start']

        newEnd = min([end  + 0.5,
                      (newEnd if newEnd > end else end),
                      self.df.loc[index, 'kinetics_end']])
        
        return newStart, newEnd

    def generateRepVid(self, index):

        path_to_video = self.path_prefix + str(self.df.loc[index, 'index'])+'.mp4'
        assert os.path.exists(path_to_video), "Video file does not exist"+path_to_video
            
        frames = []
        cap = cv2.VideoCapture(path_to_video)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret is False:
                break

            img = Image.fromarray(frame)
            preprocess = transforms.Compose([
                transforms.Resize((112, 112), 2),
                transforms.ToTensor(),       
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])])
            frameTensor = preprocess(img).unsqueeze(0)
            frames.append(frameTensor)

        cap.release()
        assert len(frames) >= self.framesPerVid, "Frames not enough"
        newFrames = []
        for i in range(1, self.framesPerVid + 1):
            newFrames.append(frames[i * len(frames)//self.framesPerVid  - 1])
        
        assert len(newFrames) == self.framesPerVid, "Uniform frame pruning technique failed"
        frames = torch.cat(newFrames)
        return frames

    def __len__(self):
        return len(self.df)
