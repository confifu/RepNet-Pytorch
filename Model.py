import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F

import math, base64, io, os, time, cv2
import numpy as np

#=============functions================

def sim_matrix(a, b, eps=1e-8):
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

'''returns 1*num_frame*num_frame'''
def _get_sims(embs):
    """Calculates self-similarity between sequence of embeddings."""
    dist = sim_matrix(embs, embs)
    sims = dist.unsqueeze(0)
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
                                out_channels = 512,
                                kernel_size = 3,
                                padding = 3,
                                dilation = 3)
        self.bn1 = nn.BatchNorm3d(512)
        #get_sims
        
        self.conv3x3 = nn.Conv2d(in_channels = 1,
                                 out_channels = 32,                  
                                 kernel_size = 3,
                                 padding = 1)
        
        #reshape from (batch, 32, frame, frame) to  (batch, frame, (frame * 32))
        
        
        #period length prediction
        self.input_projection1 = nn.Linear(self.num_frames * 32, 512)
        self.pos_encoder1 = PositionalEncoding(512, 0.1)
        self.trans_encoder1 = nn.TransformerEncoderLayer(d_model = 512,           
                                                    nhead = 4,
                                                    dim_feedforward = 512,            
                                                    dropout = 0.1,
                                                    activation = 'relu')
        
        self.fc1_1 = nn.Linear(512, 512)
        self.fc1_2 = nn.Linear(512, self.num_frames//2)

        #periodicity prediction
        self.input_projection2 = nn.Linear(self.num_frames * 32, 512) 
        self.pos_encoder2 = PositionalEncoding(512, 0.1)
        self.trans_encoder2 =  nn.TransformerEncoderLayer(d_model = 512,           
                                                        nhead = 4,
                                                        dim_feedforward = 512,            
                                                        dropout = 0.1,
                                                        activation = 'relu')
        self.fc2_1 = nn.Linear(512, 512)
        self.fc2_2 = nn.Linear(512, 1)

    def forward(self, x):
        batch_size = x.shape[0]
        x = torch.reshape(x, (-1, 3, x.shape[3], x.shape[4]))
        x = self.resnetBase(x)
        x = torch.reshape(x, (batch_size,-1,x.shape[1],x.shape[2],x.shape[3]))
        x = torch.transpose(x, 1, 2)
        x = self.Conv3D(x)
        x = self.bn1(x)
        x,_ = torch.max(x, 4)
        x,_ = torch.max(x, 3)
        
        final_embs = x
        x = torch.transpose(x, 1, 2)
        x = get_sims(x)
        x = F.relu(self.conv3x3(x))                #batch, 32, num_frame, num_frame
        x = torch.transpose(x, 1, 2)               #batch, num_frame, 32, num_frame
        x = torch.reshape(x, (batch_size, self.num_frames, -1))               #batch, num_frame, 32*num_frame
        
        x1 = self.input_projection1(x)                          #batch, num_frame, d_model=512
        x1 = self.pos_encoder1(x1)
        
        x1 = torch.transpose(x1, 0, 1)
        x1 = self.trans_encoder1(x1)
        x1 = torch.transpose(x1, 0, 1)
        
        print(x1.shape)
        y1 = F.relu(self.fc1_1(x1))
        y1 = F.relu(self.fc1_2(y1))
        y1 = torch.transpose(y1, 1, 2)              #Cross enropy wants (minbatch*classes*dimensions)
        
        x2 = self.input_projection2(x)
        x2 = self.pos_encoder2(x2)
        
        x2 = torch.transpose(x2, 0, 1)
        x2 = self.trans_encoder2(x2)
        x2 = torch.transpose(x2, 0, 1)
        
        y2 = F.relu(self.fc2_1(x2))
        y2 = F.relu(self.fc2_2(y2))
        return y1, y2, final_embs
