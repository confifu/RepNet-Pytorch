import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F

import math, base64, io, os, time, cv2
import numpy as np

#============metrics ==================
def MAE(y, ypred) :
    """for period"""
    batch_size = y.shape[0]
    yarr = y.clone().detach().cpu().numpy()
    ypredarr = ypred.clone().detach().cpu().numpy().argmax(1)
    ae = np.sum(np.absolute(yarr - ypredarr))
    mae = ae / (yarr.shape[0]*yarr.shape[1])
    return mae

def f1score(y, ypred) :
    """for periodicity"""
    batch_size = y.shape[0]
    yarr = y.clone().detach().cpu().numpy()
    ypredarr = ypred.clone().detach().cpu().numpy().astype(bool)
    tp = np.logical_and(yarr, ypredarr).sum()
    precision = tp / (ypredarr.sum() + 1e-6)
    recall = tp / (yarr.sum() + 1e-6)
    if precision + recall == 0:
        fscore = 0
    else :
        fscore = 2*precision*recall/(precision + recall)
    return fscore

#=============functions================

def sim_matrix(a, b, eps=1e-8):
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


def pairwise_l2_distance(a, b):
  """Computes pairwise distances between all rows of a and all rows of b."""
  norm_a = torch.sum(torch.square(a), 1)
  norm_a = torch.reshape(norm_a, [-1, 1])
  norm_b = torch.sum(torch.square(b), 1)
  norm_b = torch.reshape(norm_b, [1, -1])
  dist = norm_a - 2.0 * torch.mm(a, b.transpose(0,1)) + norm_b
  dist[dist < 0] = 0
  dist = -1 * dist
  return dist


'''returns 1*num_frame*num_frame'''
def _get_sims(embs):
    """Calculates self-similarity between sequence of embeddings."""
    dist = pairwise_l2_distance(embs, embs)
    #dist = sim_matrix(embs, embs)
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
    sims = torch.log(sims + 1e-10)
    
    norm = torchvision.transforms.Normalize((0.0), (0.5))
    sims = norm(sims)
    
    simsarr = []
    for i in range(batch_size):
        sim = sims[i] - sims[i].min()
        sim = sim/sim.max()
        simsarr.append(sim.unsqueeze(0))
    
    nsims = torch.vstack(simsarr)
    return nsims
        
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
        for param in self.resnetBase.parameters():
            param.requires_grad = False
        

        self.conv3D = nn.Conv3d(in_channels = 1024,
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
        
        self.input_projection = nn.Linear(self.num_frames * 32, 512)
        self.pos_encoder = PositionalEncoding(512, 0.1, 64)
        self.trans_encoder = nn.TransformerEncoderLayer(d_model = 512,           
                                                    nhead = 4,
                                                    dim_feedforward = 512,            
                                                    dropout = 0.2,
                                                    activation = 'relu')
        
        self.dropout = nn.Dropout(0.25)
        
        #period length prediction
        self.fc1_1 = nn.Linear(512, 512)
        self.fc1_2 = nn.Linear(512, self.num_frames//2)

        #periodicity prediction
        self.fc2_1 = nn.Linear(512, 512)
        self.fc2_2 = nn.Linear(512, 1)

    def forward(self, x):
        batch_size = x.shape[0]
        x = torch.reshape(x, (-1, 3, x.shape[3], x.shape[4]))
        x = self.resnetBase(x)
        x = torch.reshape(x, (batch_size,-1,x.shape[1],x.shape[2],x.shape[3]))
        x = torch.transpose(x, 1, 2)
        x = self.conv3D(x)
        x = F.relu(self.bn1(x))
        x,_ = torch.max(x, 4)
        x,_ = torch.max(x, 3)
        
        final_embs = x
        x = torch.transpose(x, 1, 2)
        x = get_sims(x)
        
        x = F.relu(self.conv3x3(x))                              #batch, 32, num_frame, num_frame
        x = torch.transpose(x, 1, 2)                                #batch, num_frame, 32, num_frame
        x = torch.reshape(x, (batch_size, self.num_frames, -1))     #batch, num_frame, 32*num_frame
        
        x = F.relu(self.input_projection(x))                      #batch, num_frame, d_model=512
        
        x = torch.transpose(x, 0, 1)                                 #num_frame, batch, d_model=512
        x = self.pos_encoder(x)
        x = self.trans_encoder(x)
        x = torch.transpose(x, 0, 1)
        
        y = self.dropout(x)
        y1 = F.relu(self.fc1_1(y))
        y1 = F.relu(self.fc1_2(y1))
        
        y1 = torch.transpose(y1, 1, 2)              #Cross enropy wants (minbatch*classes*dimensions)
        
        y2 = F.relu(self.fc2_1(y))
        y2 = F.relu(self.fc2_2(y2))
        return y1, y2, final_embs

#====================================Symmetric Cross Entropy Loss=====================

class SCELoss(torch.nn.Module):
    def __init__(self, num_classes=10, alpha=1.0, beta=1.0):
        super(SCELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        # CCE
        try:
            ce = self.cross_entropy(pred, labels)

            # RCE
            pred = F.softmax(pred, dim=1)
            pred = torch.clamp(pred, min=1e-7, max=1.0)
            label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
            label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0).transpose(1, 2)
            rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))

            # Loss
            loss = self.alpha * ce + self.beta * rce.mean()
            return loss
        except:
            print("labels", labels)
            print("pred", pred)
