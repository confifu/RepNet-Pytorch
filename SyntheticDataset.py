import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms

import cv2
import glob
import random
from tqdm import tqdm
from random import randrange, randint
import math, base64, io, os, time
from torch.utils.data import Dataset, DataLoader, ConcatDataset

#returns a 64 length array that goes low->mid->high->mid

def getRandomTransformParameter(high, mid, low, length=64):
    retarr = []
    midpos = randint(length//4, length//2)
    highpos = randint(length//2, 3*length//4)
    
    retarr = list(np.linspace(start=low, stop=mid, num=midpos))
    retarr.extend(list(np.linspace(start=mid, stop=high, num=highpos-midpos)))
    retarr.extend(list(np.linspace(start=high, stop=mid, num=length - highpos)))
    
    retarr = np.array(retarr)
    retarr = retarr[::random.choice([-1, 1])]
    return retarr

def randomTransform(frames):
    
    #resize
    scaleParams = getRandomTransformParameter(0.2, 0.1, 0.0)
    #rotate z
    zRotateParams = getRandomTransformParameter(45, 0, -45)
    
    #rotate x
    xRotateParams = getRandomTransformParameter(0.2, 0.0, -0.2, 32)
    #rotate y
    yRotateParams = getRandomTransformParameter(0.2, 0.0, -0.2, 32)
    
    #translate horizontally
    horizTransParam = getRandomTransformParameter(0.1, 0.0, -0.1)
    #translate vertically
    verticalTransParam = getRandomTransformParameter(0.1, 0.0, -0.1)
    #cheap filters
    
    cbParam = [getRandomTransformParameter(1.08, 1.0, 0.92) for i in range(6)]
    dv = np.random.choice([0, 1], size=(6,))
    dv = [0, 0, 1, 1, 1, 1]
    
    #rotate z
    if dv[0]:
        for i, frame in enumerate(frames[:32]):

            #rotate x
            frames[i] = skew_x(frame, xRotateParams[i])

    if dv[1]:
        for i, frame in enumerate(frames[32:]):

            #rotate y
            frames[32+i] = skew_y(frame, yRotateParams[i])

    newFrames = []
    for i, frame in enumerate(frames):
        
        if dv[2]:
            frame = scale(frame, scaleParams[i])
        if dv[3]:
            frame = rotate_bound(frame, zRotateParams[i])
        if dv[4]:
            frame = translate(frame, horizTransParam[i], verticalTransParam[i])
        if dv[5]:
            frame = change_cb(frame,
                             cbParam[0][i],
                             cbParam[1][i],
                             cbParam[2][i],
                             cbParam[3][i],
                             cbParam[4][i],
                             cbParam[5][i])

        newFrames.append(frame)

    return newFrames

def image_stats(image):
    # compute the mean and standard deviation of each channel
    (l, a, b) = cv2.split(image)
    (lMean, lStd) = (l.mean(), l.std())
    (aMean, aStd) = (a.mean(), a.std())
    (bMean, bStd) = (b.mean(), b.std())
    # return the color statistics
    return (lMean, lStd, aMean, aStd, bMean, bStd)

def change_cb(image, lmf, lsf, amf, asf, bmf, bsf):
    # compute color statistics for the source and target images
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype("float32")
    
    (lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = image_stats(image)
    
    (lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = (lMeanTar * lmf,
                                                                 lStdTar * lsf + 1e-10, 
                                                                 aMeanTar * amf, 
                                                                 aStdTar * asf + 1e-10, 
                                                                 bMeanTar * bmf, 
                                                                 bStdTar * bsf + 1e-10)
    
    # subtract the means from the target image
    (l, a, b) = cv2.split(image)
    l -= lMeanTar
    a -= aMeanTar
    b -= bMeanTar
    # scale by the standard deviations
    l = (lStdTar / lStdSrc) * l
    a = (aStdTar / aStdSrc) * a
    b = (bStdTar / bStdSrc) * b
    # add in the source mean
    l += lMeanSrc
    a += aMeanSrc
    b += bMeanSrc
    # clip the pixel intensities to [0, 255] if they fall outside
    # this range
    l = np.clip(l, 0, 255)
    a = np.clip(a, 0, 255)
    b = np.clip(b, 0, 255)
    # merge the channels together and convert back to the RGB color
    # space, being sure to utilize the 8-bit unsigned integer data
    # type
    transfer = cv2.merge([l, a, b])
    transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2RGB)

    # return the color transferred image
    return transfer

def skew_x(image, factor):
    h = image.shape[0]
    w = image.shape[1]
    fx = factor/2
    
    #image = scale(image, 1)
    pts1 = np.float32([
                       [0, 0],
                       [0, h],
                       [w, 0],
                       [w, h]
                    ])
    #pts1 += np.float32([w,h])

    pts2 = np.float32([
                       [-w * min(0,fx)      , h * np.abs(fx)],
                       [ w * max(0,fx)      , h * (1 - np.abs(fx))],
                       [ w * (1 + min(0,fx)), h * np.abs(fx)],
                       [ w * (1 - max(0,fx)), h * (1 - np.abs(fx))]
                    ])
    #pts2 += np.float32([w,h])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    skewed = cv2.warpPerspective(image, M, (w, h), flags = cv2.INTER_AREA)
    #skewed = skewed[w:2*w, h:2*h,:]
    return skewed

def skew_y(image, factor):
  
    h = image.shape[0]
    w = image.shape[1]
    fy = factor/2
  
    #image = scale(image, 1)
    
    pts1 = np.float32([
                       [0, 0],
                       [0, h],
                       [w, 0],
                       [w, h]
                     ])
    #pts1 = pts1 + np.float32([w, h])

    pts2 = np.float32([
                       [ w * np.abs(fy)      , -h * min(0,fy)],
                       [ w * np.abs(fy)      , h * (1 + min(0,fy))],
                       [ w * (1 - np.abs(fy)), h * max(0,fy) ],
                       [ w * (1 - np.abs(fy)), h * (1 - max(0,fy))]
                    ])
    #pts2 = pts2 + np.float32([w, h])


    M = cv2.getPerspectiveTransform(pts1, pts2)
    skewed = cv2.warpPerspective(image, M, (w, h), flags = cv2.INTER_AREA)
    #skewed = skewed[w:2*w, h:2*h,:]
    return skewed
    
def translate(image, factorx, factory):
    h = image.shape[0]
    w = image.shape[1]
    rows = np.any(image, axis=1)
    cols = np.any(image, axis=0)
    try:
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
    except:
        return image
    
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
                               cv2.BORDER_CONSTANT,
                               value = (0,0,0))
    return scaled

def rotate_bound(image, angle):
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
                
                period = (64 - beginNoRepFrames - endNoRepFrames)//count
                period = period if 1< period <32 else 0
                
                periodLength[i] = period
                assert(period < 32)

        periodLength = torch.LongTensor(periodLength)
        periodicity = torch.BoolTensor(periodicity)
        return X, periodLength, periodicity, index
    
    def getPeriodDist(self, samples):
        arr = np.zeros(32,)
        
        for i in tqdm(range(samples)):
            _, p,_,_ = self.__getitem__(0)
            per = max(p)
            arr[per] += 1
        return arr
            
        
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
        period = randint(2 , 31)
        count = np.random.choice([1, randint(2, 64//(period))], p=[0.033, 0.967])
        
        clipDur = randint(min(total//(64/period - count + 1), max(period, 30)), 
                          min(total//(64/period - count + 1), 60))

        repDur = count * clipDur
        noRepDur =  int((64 / (period*count) - 1) * repDur)
         
        assert(noRepDur >= 0)
        begNoRepDur = randint(0,  noRepDur)
        endNoRepDur = noRepDur - begNoRepDur
        totalDur = noRepDur + repDur
            
        startFrame = randint(0, total - (clipDur + noRepDur))
        cap.set(cv2.CAP_PROP_POS_FRAMES, startFrame)
        
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if ret is False or len(frames) == clipDur + noRepDur:
                break
            frame = cv2.resize(frame , (256, 256), interpolation = cv2.INTER_AREA)
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
      
        newFrames = randomTransform(newFrames)      
        for frame in newFrames:
            img = Image.fromarray(frame)
            preprocess = transforms.Compose([
            transforms.Resize((112, 112), 2),
            transforms.ToTensor(),
            transforms.ColorJitter(brightness=0.02, contrast=0.05, saturation=0.05, hue=0),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
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

        
