import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms

import cv2
import math, base64, io, os, time
from torch.utils.data import Dataset, DataLoader, ConcatDataset

class miniDataset(Dataset):
    
    def __init__(self, df, path_to_video):
        
        self.path = path_to_video
        self.df = df.reset_index()
        self.count = self.df.loc[0, 'count']
        self.numseq = self.count//10 if self.count%10 == 0 else self.count//10 + 1
        
        
  
    def getFrames(self, num_frames):
        """returns 50 * self.length number of frames"""
    
        frames = []
        cap = cv2.VideoCapture(self.path)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret is False:
                break
            
            img = Image.fromarray(frame)
            frames.append(img)
        
        cap.release()

        try:
            newFrames = []
            for i in range(1, num_frames + 1):
                newFrames.append(frames[i * len(frames)//num_frames  - 1])
            return newFrames
       
        except:
            print(self.path)
            print("index ", self.df.loc[0, 'index'], "total_frames", len(frames), "expected", num_frames)
            return None


    def __getitem__(self, index):
        
        #get another random number betweeen 0 and 14 'a'
        from random import randrange
        a = randrange(13) + 1
        
        #take out ith 50 groups of frames
        all_frames = self.getFrames(50 * self.numseq)
        currFrames = [all_frames[index*50] for i in range(0, a)]
        currFrames.extend(all_frames[index*50: (1 + index)*50])
        currFrames.extend([all_frames[(index + 1)*50 - 1] for i in range(0, 14-a)])
        
        
        Xlist = []
        for img in currFrames:
        
            preprocess = transforms.Compose([
            transforms.Resize((112, 112), 2),
            transforms.ToTensor()])
            frameTensor = preprocess(img).unsqueeze(0)
            Xlist.append(frameTensor)
        
        X = torch.cat(Xlist)
                          
        y1 = torch.FloatTensor([self.count//self.numseq])       #numrep
        
        y2 = [0 for i in range(0, a)]
        y2.extend([1 for i in range(0, 50)])
        y2.extend([0 for i in range(0, 14 - a)])
        y2 = torch.BoolTensor(y2).unsqueeze(-1)                #periodicity
        
        return X, y1, y2
        
    def __len__(self):
        return self.numseq
    
    

def getCombinedDataset(dfPath, videoDir, videoPrefix):
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
        
    megaDataset = ConcatDataset(miniDatasetList)
    return megaDataset



def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result


class SyntheticDataset(Dataset):
    
    def __init__(self, videoPath, filename, length):
        
        self.sourcePath = videoPath + '/' + filename
        self.length = length

    def __getitem__(self, index):

        X, beginNoRepFrames, endNoRepFrames, count = self.generateRepVid()
        
        periodicity = np.zeros((64, 1)) 
        for i in range(64):
            if beginNoRepFrames < i < 64 - endNoRepFrames:
                periodicity[i] = True

        numReps = torch.LongTensor([count])
        periodicity = torch.BoolTensor(periodicity)
        return X, numReps, periodicity

    def generateRepVid(self):

        assert os.path.exists(self.sourcePath), "Video file does not exist" + self.sourcePath
        
        from random import randrange, randint
        
        mirror = randint(0, 1)
        clipDur = randint(60, 120)
        count = randint(3, 10)
        repDur = ((mirror + 1)*count + mirror)* clipDur
        noRepDur = repDur//4
        begNoRepDur = randint(1, noRepDur - 1)
        endNoRepDur = noRepDur - begNoRepDur
        totalDur = noRepDur + repDur
        
        cap = cv2.VideoCapture(self.sourcePath)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        startFrame = randint(0, total - 2 * (clipDur + noRepDur))         #not taking risks
        cap.set(cv2.CAP_PROP_POS_FRAMES, startFrame)
        
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if ret is False or len(frames) == clipDur + noRepDur:
                break
            
            frames.append(frame)
        
        cap.release()
        begNoRepFrames = frames[:begNoRepDur]
        endNoRepFrames = frames[-endNoRepDur:]
        
        repFrames = frames[begNoRepDur : -endNoRepDur]
        
        finalFrames = begNoRepFrames
        for i in range(count):
            finalFrames.extend(repFrames)
            if mirror:
                finalFrames.extend(repFrames[::-1])
        
        finalFrames.extend(repFrames)
        finalFrames.extend(endNoRepFrames)
        
        newFrames = []
        for i in range(1, 64 + 1):
            newFrames.append(finalFrames[i * len(finalFrames)//64  - 1])
        
        angle = 0
        change = 2
        tensorList = []
        for frame in newFrames:
            if angle > 45 and change > 0:
                change = -change
            angle = angle + change
            frame = rotate_image(frame, angle)
            img = Image.fromarray(frame)
            preprocess = transforms.Compose([
            transforms.Resize((112, 112), 2),
            transforms.ToTensor()])
            img = preprocess(img).unsqueeze(0)
            tensorList.append(img)
        
        frames = torch.cat(tensorList)
        
        numBegNoRepFrames = begNoRepDur*64//totalDur
        numEndNoRepFrames = endNoRepDur*64//totalDur
        
        return frames, numBegNoRepFrames, numEndNoRepFrames, count

    def __len__(self):
        return self.length

        
