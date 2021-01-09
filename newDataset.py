import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms

import cv2
import glob
import random
from random import randrange, randint
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
        a = randrange(15)
        
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
                          
        numreps = self.count//self.numseq
        y1 = [0 for i in range(0,a)]
        y1.extend([50//numreps for i in range(0, 50)])
        y1.extend([0 for i in range(0, 14 - a)])
        y1 = torch.LongTensor(y1)                #periodicity
        
        
        
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


'''==============================================Synthetic_DS==============================================='''

#returns a 64 length array that goes low->mid->high->mid

def getRandomTransformParameter(high, mid, low, length=64):
    retarr = []
    midpos = randint(length//4, length//2)
    highpos = randint(length//2, 3*length//4)
    
    retarr = list(np.linspace(start=low, stop=mid, num=midpos))
    retarr.extend(list(np.linspace(start=mid, stop=high, num=highpos-midpos)))
    retarr.extend(list(np.linspace(start=high, stop=mid, num=length - highpos)))
    return np.array(retarr)
    
def randomTransform(frames):
    
    #resize
    scaleParams = getRandomTransformParameter(0.2, 0.1, 0.0)
    #rotate z
    zRotateParams = getRandomTransformParameter(-10, 0, 10)
    #rotate y
    yRotateParams = getRandomTransformParameter(0.2, 0.1, 0.0, 32)
    #rotate x
    xRotateParams = getRandomTransformParameter(-0.1, 0.0, 0.1, 32)
    #translate horizontally
    horizTransParam = getRandomTransformParameter(0.125, 0.0, -0.125)
    #translate vertically
    verticalTransParam = getRandomTransformParameter(-0.125, 0.0, 0.125)
    #cheap filters
    
    #rotate z
    
        
    for i, frame in enumerate(frames[:32]):
       
        #rotate x
        frames[i] = skew_x(frame, xRotateParams[i])
    
    for i, frame in enumerate(frames[32:]):
        
        #rotate y
        frames[32+i] = skew_y(frame, yRotateParams[i])

    newFrames = []
    for i, frame in enumerate(frames):
        
        frame = rotate_bound(frame, zRotateParams[i])
        frame = scale(frame, scaleParams[i])
        frame = translate(frame, horizTransParam[i], verticalTransParam[i])
        
        newFrames.append(frame)

    return newFrames
                  

                      
def skew_x(image, factor):
    h = image.shape[0]
    w = image.shape[1]
    fx = factor/2
    
    image = scale(image, 1)
    pts1 = np.float32([
                       [0, 0],
                       [0, h],
                       [w, 0],
                       [w, h]
                    ])
    pts1 += np.float32([w,h])

    pts2 = np.float32([
                       [-w * min(0,fx)      , h * np.abs(fx)],
                       [ w * max(0,fx)      , h * (1 - np.abs(fx))],
                       [ w * (1 + min(0,fx)), h * np.abs(fx)],
                       [ w * (1 - max(0,fx)), h * (1 - np.abs(fx))]
                    ])
    pts2 += np.float32([w,h])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    skewed = cv2.warpPerspective(image, M, (3*w, 3*h), flags = cv2.INTER_AREA)
    skewed = skewed[w:2*w, h:2*h,:]
    return skewed

def skew_y(image, factor):
  
    h = image.shape[0]
    w = image.shape[1]
    fy = factor/2
  
    image = scale(image, 1)
    
    pts1 = np.float32([
                       [0, 0],
                       [0, h],
                       [w, 0],
                       [w, h]
                     ])
    pts1 = pts1 + np.float32([w, h])

    pts2 = np.float32([
                       [ w * np.abs(fy)      , -h * min(0,fy)],
                       [ w * np.abs(fy)      , h * (1 + min(0,fy))],
                       [ w * (1 - np.abs(fy)), h * max(0,fy) ],
                       [ w * (1 - np.abs(fy)), h * (1 - max(0,fy))]
                    ])
    pts2 = pts2 + np.float32([w, h])


    M = cv2.getPerspectiveTransform(pts1, pts2)
    skewed = cv2.warpPerspective(image, M, (3*w, 3*h), flags = cv2.INTER_AREA)
    skewed = skewed[w:2*w, h:2*h,:]
    return skewed
    
def translate(image, factorx, factory):
    h = image.shape[0]
    w = image.shape[1]
    rows = np.any(image, axis=1)
    cols = np.any(image, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    hTrans = factorx * w
    xT = max(hTrans, -xmin) if hTrans < 0 else min(hTrans, w - xmax)

    vTrans = factory * h
    yT = max(vTrans, -ymin) if vTrans < 0 else min(vTrans, h - ymax)

    M = np.float32([[1, 0, xT],[0, 1, yT]])
    
    translated = cv2.warpAffine(image, M, (w, h), flags = cv2.INTER_AREA)
    return translated
                      
def scale(image, factor):

    sp = factor
    scaled = cv2.copyMakeBorder(image,
                               int(image.shape[0] * sp),
                               int(image.shape[0] * sp),
                               int(image.shape[1] * sp),
                               int(image.shape[1] * sp),
                               cv2.BORDER_REFLECT)
    return scaled

def rotate_bound(image, angle):
    (oh, ow) = image.shape[:2]
    
    image = scale(image, 1)
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    
    image = cv2.warpAffine(image, M, (int(nW), int(nH)), flags = cv2.INTER_AREA)
    image = image[ow:2*ow, oh:2*oh,:]
    return image

class SyntheticDataset(Dataset):
    
    def __init__(self, videoPath, filename, extension, length):
        
        self.sourcePath = videoPath + '/' + filename + '.' + extension
        self.length = length

    def __getitem__(self, index):

        X, beginNoRepFrames, endNoRepFrames, count = self.generateRepVid()
        
        periodicity = np.zeros((64, 1))
        periodLength = np.zeros((64, 1))
        for i in range(64):
            if beginNoRepFrames < i < 64 - endNoRepFrames:
                periodicity[i] = True
                periodLength[i] = max(0, (64 - beginNoRepFrames - endNoRepFrames)//count)

        periodLength = torch.LongTensor(periodLength).squeeze(1)
        periodicity = torch.BoolTensor(periodicity)
        return X, periodLength, periodicity

    def generateRepVid(self):
        
        while True:
            path = random.choice(glob.glob(self.sourcePath))
            assert os.path.exists(path), "No file with this pattern exist" + self.sourcePath

            cap = cv2.VideoCapture(path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total > 64:
                break
            else:
                os.remove(path)
        
        mirror = randint(0, 1)
        count = randint(1, 15)
        
        clipDur = randint(min(30, total//10), min(60, total//5))
        repDur = ((mirror + 1)*count)* clipDur
        noRepDur = min(total-clipDur, int(repDur *(64/(randint(4, 64//count)*count) - 1)))
        
        assert(noRepDur >= 0)
        begNoRepDur = randint(0,  noRepDur)
        endNoRepDur = noRepDur - begNoRepDur
        totalDur = noRepDur + repDur
        
        startFrame = randint(0, total - (clipDur + noRepDur))         #not taking risks
        cap.set(cv2.CAP_PROP_POS_FRAMES, startFrame)
        
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if ret is False or len(frames) == clipDur + noRepDur:
                break
            frame = cv2.resize(frame , (512, 512), interpolation = cv2.INTER_AREA)
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
        
        finalFrames.extend(endNoRepFrames)
        
        newFrames = []
        for i in range(1, 64 + 1):
            newFrames.append(finalFrames[i * len(finalFrames)//64  - 1])
            
        assert(len(newFrames) == 64)
        
        tensorList = []
        
        try:
            newFrames = randomTransform(newFrames)
        except:
            return self.generateRepVid()
            
        for frame in newFrames:
            img = Image.fromarray(frame)
            preprocess = transforms.Compose([
            transforms.Resize((112, 112), 2),
            transforms.ToTensor()])
            img = preprocess(img).unsqueeze(0)
            tensorList.append(img)
        
        frames = torch.cat(tensorList)
        
        numBegNoRepFrames = begNoRepDur*64//totalDur
        if count == 1:
            numEndNoRepFrames = 64 - numBegNoRepFrames
        else:
            numEndNoRepFrames = endNoRepDur*64//totalDur
        
        return frames, numBegNoRepFrames, numEndNoRepFrames, count

    def __len__(self):
        return self.length

        
