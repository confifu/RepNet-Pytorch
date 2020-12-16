import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F

import math, base64, io, os, time, cv2
import numpy as np

#=============functions================
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

def pairwise_l2_distance(a, b):
  """Computes pairwise distances between all rows of a and b."""
  norm_a = torch.sum(torch.square(a), 1)
  norm_a = torch.reshape(norm_a, (-1, 1))
  norm_b = torch.sum(torch.square(b), 1)
  norm_b = torch.reshape(norm_b, (1, -1))
  dist = torch.maximum(norm_a - 2.0 * torch.matmul(a,torch.transpose(b,0,1)) + norm_b, torch.tensor(0.0).to(device))
  return dist

'''returns 1*num_frame*num_frame'''
def _get_sims(embs):
    """Calculates self-similarity between sequence of embeddings."""
    dist = pairwise_l2_distance(embs, embs)
    sims = -1.0 * dist
    sims = sims.unsqueeze(0)
    return sims

def get_sims(embs, temperature = 13.544):
    batch_size = embs.shape[0]
    seq_len = embs.shape[1]
    embs = torch.reshape(embs, (batch_size, seq_len, -1))

    simsarr=[]
    for i in range(batch_size):
        simsarr.append(_get_sims(embs[i,:,:]).unsqueeze(0))
    
    sims = torch.vstack(simsarr)
    sims /= temperature
    sims = F.softmax(sims, dim=-1)
    return sims
        
#============classes===================
class ResNet50Bottom(nn.Module):
    def __init__(self, original_model):
        super(ResNet50Bottom, self).__init__()
        self.rnet=nn.Sequential(*list(original_model.children())[:-4])
        self.left=nn.Sequential(*list(original_model.children())[-4][:3])
        
    def forward(self, x):
        x = self.rnet(x)
        x = self.left(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

#=============Model====================


class RepNet(nn.Module):
    def __init__(self, num_frames):
        super(RepNet, self).__init__()
        self.num_frames = num_frames
        resnetbase = torchvision.models.resnet50(pretrained=True, progress=True)
        self.resnetBase = ResNet50Bottom(resnetbase)
        

        self.Conv3D = nn.Conv3d(in_channels = 1024,
                                out_channels = 512,               #changing to 256 from 512
                                kernel_size = 3,
                                padding = 3,
                                dilation = 3)
        self.bn1 = nn.BatchNorm3d(512)
        #get_sims
        
        self.conv3x3_1 = nn.Conv2d(in_channels = 1,
                                 out_channels = 32,                    #changing to 16 from 32
                                 kernel_size = 3,
                                 padding = 1)
        
        self.conv3x3_2 = nn.Conv2d(in_channels = 32,
                                 out_channels = 512,                      #changing to 256 from 512
                                 kernel_size = 3,
                                 padding = 1)
        self.pos_encoder = PositionalEncoding(512, 0.1)                           #do
        trans_encoder_layer = nn.TransformerEncoderLayer(d_model = 512,           #do
                                                nhead = 4,
                                                dim_feedforward = 512,            #do
                                                dropout = 0.1,
                                                activation = 'relu')
        self.trans_encoder=nn.TransformerEncoder(trans_encoder_layer,1)

        #fc layers
        #period length prediction
        self.fc1_1 = nn.Linear(self.num_frames*512, 512)                             #do
        self.fc1_2 = nn.Linear(512, self.num_frames//2)                              #do
        
        #periodicity module
        self.fc2_1 = nn.Linear(self.num_frames*512, 512)                              #do
        self.fc2_2 = nn.Linear(512, 1)                                                #do

    def forward(self, x):
        batch_size = x.shape[0]
        x = torch.reshape(x, (-1, 3, x.shape[3], x.shape[4]))
        x = self.resnetBase(x)
        x = torch.reshape(x, 
                    (batch_size,-1,x.shape[1],x.shape[2],x.shape[3]))
        x = torch.transpose(x, 1, 2)
        x = self.Conv3D(x)
        x = self.bn1(x)
        x,_ = torch.max(x, 4)
        x,_ = torch.max(x, 3)
        
        final_embs = x
        
        x = torch.transpose(x, 1, 2)
        x = get_sims(x)
        x = F.relu(self.conv3x3_1(x))
        x = F.relu(self.conv3x3_2(x))
        x = torch.reshape(x, (batch_size, 512, -1))                                       #do
        x = torch.transpose(x, 1, 2)
        x = self.pos_encoder(x)
        x = self.trans_encoder(x)
        x = torch.reshape(x, (x.shape[0],self.num_frames, -1))
        
        y1 = F.relu(self.fc1_1(x))
        y1 = F.relu(self.fc1_2(y1))
        y1 = torch.transpose(y1, 1, 2)              #Cross enropy wants (minbatch*classes*dimensions)

        y2 = F.relu(self.fc2_1(x))
        y2 = F.relu(self.fc2_2(y2))
        return y1, y2, final_embs
