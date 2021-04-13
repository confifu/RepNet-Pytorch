import torch
import numpy as np
import random
import cv2, os
from PIL import Image
from torchvision import transforms


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def predRep(vidPath, model, numdivs = 1):
    countbest=[]
    periodicitybest = []
    Xbest = None
    countbest = [-1]
    simsbest = []
    periodbest = []
    
    for i in range(numdivs, numdivs+1):
        frames = getFrames(vidPath, 64*i)
        periodicity = []
        periodLength = []
        sims = []
        X = []
        for j in range(i):
            x, periodLengthj, periodicityj, sim = predSmall(frames[j*64:(j+1)*64], model)
            periodicity.extend(list(periodicityj.squeeze().cpu().numpy()))
            periodLength.extend(list(periodLengthj.squeeze().cpu().numpy()))
            X.append(x)
            sims.append(sim)
        
        X = torch.cat(X)
        numofReps = 0
        count = []
        for i in range(len(periodLength)):
            if periodLength[i] == 0:
                numofReps += 0
            else:
                numofReps += max(0, periodicity[i]/(periodLength[i]))

            count.append(round(float(numofReps), 2))
        
        if count[-1] > countbest[-1]:
            countbest = count
            Xbest = X
            periodicitybest = periodicity
            simsbest = sims
            periodbest = periodLength
    
    return Xbest, countbest, periodicitybest, periodbest, simsbest

                
def getFrames(vidPath, num_frames=64):
    frames = []
    cap = cv2.VideoCapture(vidPath)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is False:
            break
        img = Image.fromarray(frame)
        frames.append(img)
    cap.release()
    
    
    newFrames = []
    for i in range(1, num_frames + 1):
        newFrames.append(frames[i * len(frames)//num_frames  - 1])
    
    return newFrames

def predSmall(frames, model):

    Xlist = []
    for img in frames:

        preprocess = transforms.Compose([
        transforms.Resize((112, 112), 2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        frameTensor = preprocess(img).unsqueeze(0)
        Xlist.append(frameTensor)

    X = torch.cat(Xlist)
    #print(X.shape)
    with torch.no_grad():
        model.eval()
        y1pred, y2pred, sim = model(X.unsqueeze(0).to(device), True)
    
    periodLength = y1pred.round().long()
    periodicity = y2pred > 0
    #print(periodLength.squeeze())
    #print(periodicity.squeeze())
    
    sim = sim[0,0,:,:]
    sim = sim.detach().cpu().numpy()
    
    return X, periodLength, periodicity, sim

def getAnim(X, countPred = None, count = None, idx = None):
    
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    plt.rcParams['animation.html'] = "jshtml"

    fig, ax = plt.subplots()
    axesimg = ax.imshow(np.zeros((100,100, 3)))
    
    def animate(i):
        title = " "
        if countPred is not None:
            title += "pred"
            title += str(countPred[i])
        if count is not None:
            title += " actual"
            title += str(count[i])
        if idx is not None:
            title += " id"
            title += str(idx)
        
        ax.set_title(title)
        img = X[i,:,:,:].transpose(0, 1).transpose(1,2).detach().cpu().numpy()
        ax.imshow(img)

    anim = FuncAnimation(fig, animate, frames=64, interval=500)
    
    return anim

def getCount(period, periodicity = None):
    period = period.round().squeeze()

    count = []

    if periodicity is None:
        periodicity = period > 2
    else :
        periodicity = periodicity.squeeze() > 0

    numofReps = 0
    for i in range(len(period)):
        if period[i] == 0:
            numofReps+=0
        else:
            numofReps += max(0, periodicity[i]/period[i])

        count.append(int(numofReps))
    return count, period, periodicity
