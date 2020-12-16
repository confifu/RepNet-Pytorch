import torch
import numpy as np
import pandas as pd
import ffmpeg
from PIL import Image
from torchvision import transforms
import cv2, youtube_dl, subprocess
import math, base64, io, os, time, cv2
from torch.utils.data import Dataset, DataLoader, ConcatDataset

class CountixDataset(Dataset):
    
    def __init__(self, df_path, framesPerVid):
        self.df = pd.read_csv(df_path)
        self.df= self.df.sort_values(by=['video_id',
                                         'repetition_start',
                                         'repetition_end'])
        self.framesPerVid = framesPerVid

    def __getitem__(self, index):

        try :
            id = self.df.loc[index, 'video_id']
            start = self.df.loc[index, 'repetition_start']
            end = self.df.loc[index, 'repetition_end']
            count = self.df.loc[index, 'count']
            X, newStart, newEnd = self.generateRepVid(start,end,id,index)
        
        except:
            index = 0
            id = self.df.loc[index, 'video_id']
            start = self.df.loc[index, 'repetition_start']
            end = self.df.loc[index, 'repetition_end']
            count = self.df.loc[index, 'count']
            
            X, newStart, newEnd = self.generateRepVid(start,end,id,index)
        
        durationScaleFac = (newEnd-newStart)/self.framesPerVid
        repStartFrame = int((start- newStart)/durationScaleFac)
        repEndFrame = int((end - newStart)/durationScaleFac)
        
        periodLength = np.zeros((self.framesPerVid))
        for i in range(self.framesPerVid):
            if repStartFrame< i <repEndFrame:
                periodLength[i] = (repEndFrame-repStartFrame)/count
                
        periodicity = np.zeros((self.framesPerVid,1)) 
        for i in range(self.framesPerVid):
            if repStartFrame< i <repEndFrame:
                periodicity[i] = True


        periodLength = torch.LongTensor(periodLength)
        periodicity = torch.FloatTensor(periodicity)
        return X, periodLength, periodicity

    def generateRepVid(self, start, end, id, index):

        newStart =  -1      
        if self.df.loc[max(0,index-1), 'video_id']== self.df.loc[index, 'video_id']:
            if int(self.df.loc[max(0,index-1), 'repetition_end']) == int(self.df.loc[index, 'repetition_start']):
                newStart=self.df.loc[max(0,index-1), 'repetition_end']

        newStart = max([start - 0.5,
                    newStart,
                    self.df.loc[index, 'kinetics_start']])

        newEnd = np.inf   
        if self.df.loc[min(len(self)-1,index+1), 'video_id']==self.df.loc[index, 'video_id']:
            if int(self.df.loc[index, 'repetition_end'])==int(self.df.loc[min(len(self)-1,index+1), 'repetition_start']):
                newEnd=self.df.loc[min(len(self)-1,index+1), 'repetition_start']

        newEnd = min([end  + 0.5,
                      newEnd,
                      self.df.loc[index, 'kinetics_end']])

        url = 'https://www.youtube.com/watch?v='+id
        path_to_video = 'videodump/video_to_train'+str(index)+'.mp4'
        #fps = self.framesPerVid/(newEnd-newStart) + 1
        #print("start and end time", newStart, newEnd, id)
        
        if os.path.exists(path_to_video):
            os.remove(path_to_video)

        opts = {'format': 'worst',
                'quiet':True,
                }
        
        while(True):
            with youtube_dl.YoutubeDL(opts) as ydl:
                result=ydl.extract_info(url, download=False)
                video=result['entries'][0] if 'entries' in result else result

            url = video['url']
            '''
            vid = ffmpeg.probe(url)
            i = vid['streams'][0]
            '''
            origDur = (end - start)
            fps = 64/origDur     
            #print("fps is ", fps)
            
            subprocess.call('ffmpeg -i "%s" -ss %s -to %s -r %s -an "%s"' % 
                                        (url, newStart, newEnd, fps, path_to_video), shell=True)

            if os.path.exists(path_to_video):
                break
        
        #path_to_video = newPath
        assert os.path.exists(path_to_video), "Video file does not exist"
            
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
        #print("in cv loop", len(frames))
        
        assert len(frames) >= 64, "frames do not exist"
        #print("number of frames are :", len(frames))
        
        if os.path.exists(path_to_video):
            os.remove(path_to_video)

        toDelete = len(frames) - 64
        period = len(frames)//toDelete
        for i in range(0, toDelete):
            del frames[len(frames) - 1 - i*period]
        
        #print("length after maneuver", len(frames))    
            
        frames = frames[:self.framesPerVid]
        frames = torch.cat(frames)
        return frames, newStart, newEnd

    def __len__(self):
        return len(self.df)
