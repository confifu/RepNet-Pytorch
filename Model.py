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

#============classes===================

class Sims(nn.Module):
    def __init__(self):
        super(Sims, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.bn = nn.BatchNorm2d(1)
        
    def forward(self, x):
        '''(N, S, E)  --> (N, S, S)'''
        f = x.shape[1]
        
        I = torch.ones(1, f).to(self.device)
        xr = torch.einsum('bfe,gh->bhfe', (x, I))   #[x, x, x, x ....]
        xc = torch.einsum('bfe,gh->bfhe', (x, I))   #[x x x x ....]
        diff = xr - xc
        out = torch.einsum('bfge,bfge->bfg', (diff, diff))
        out = self.bn(out.unsqueeze(1))
        out = F.softmax(-out/13.544, dim = -1)
        return 

#---------------------------------------------------------------------------

class ResNet50Bottom(nn.Module):
    def __init__(self):
        super(ResNet50Bottom, self).__init__()
        self.original_model = torchvision.models.resnet50(pretrained=True, progress=True)
        self.activation = {}
        h = self.original_model.layer3[2].register_forward_hook(self.getActivation('comp'))
        
    def getActivation(self, name):
        def hook(model, input, output):
            self.activation[name] = output
        return hook

    def forward(self, x):
        self.original_model(x)
        output = self.activation['comp']
        return output

#---------------------------------------------------------------------------

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

#----------------------------------------------------------------------------

class TransEncoder(nn.Module):
    def __init__(self, d_model, n_head, dim_ff, dropout=0.0, num_layers = 1):
        super(TransEncoder, self).__init__()
        
        self.pos_encoder = PositionalEncoding(d_model, 0.1, 64)
        encoder_layer = nn.TransformerEncoderLayer(d_model = d_model,           
                                                    nhead = n_head,
                                                    dim_feedforward = dim_ff,
                                                    dropout = 0.0,
                                                    activation = 'relu')
        
        self.trans_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
    def forward(self, src):
        src = self.pos_encoder(src)
        e_op = self.trans_encoder(src)
        return e_op


#=============Model====================


class RepNet(nn.Module):
    def __init__(self, num_frames):
        super(RepNet, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.num_frames = num_frames
        self.resnetBase = ResNet50Bottom()
        
        self.conv3D = nn.Conv3d(in_channels = 1024,
                                out_channels = 512,
                                kernel_size = 3,
                                padding = (3,0,0),
                                dilation = (3,1,1))
        self.bn1 = nn.BatchNorm1d(512*64*5*5)
        self.pool = nn.MaxPool3d(kernel_size = (1, 5, 5))
        self.sims = Sims()
        
        self.conv3x3 = nn.Conv2d(in_channels = 1,
                                 out_channels = 32,
                                 kernel_size = 3,
                                 padding = 1)
        
        self.input_projection = nn.Linear(self.num_frames * 32, 512)
        
        self.transEncoder = TransEncoder(d_model=512, n_head=4, dim_ff=512, num_layers = 1)
        self.dropout = nn.Dropout(0.25)
        
        #period length prediction
        self.fc1_1 = nn.Linear(512, 512)
        self.fc1_2 = nn.Linear(512, self.num_frames//2)
    
    def forward(self, x):
        batch_size, _, c, h, w = x.shape
        x = x.view(-1, c, h, w)
        x = self.resnetBase(x)
        x = x.view(batch_size, self.num_frames, x.shape[1],  x.shape[2],  x.shape[3])
        x = x.transpose(1, 2)
        
        x = F.relu(self.bn1(self.conv3D(x).view(batch_size, -1)))
        x = x.view(batch_size, 512, self.num_frames, 5, 5)
        x = self.pool(x).squeeze(3).squeeze(3)
        x = x.transpose(1, 2)                           #batch, num_frame, 512
        x = x.reshape(batch_size, self.num_frames, -1)
       
        x = F.relu(self.sims(x))
        x = F.relu(self.conv3x3(x))           #batch, 32, num_frame, num_frame
        
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(batch_size, self.num_frames, -1)  #batch, num_frame, 32*num_frame
        x = F.relu(self.input_projection(x))           #batch, num_frame, d_model=512
        
        x = x.transpose(0, 1)                          #num_frame, batch, d_model=512
        x = self.transEncoder(x)
        x = x.transpose(0, 1)
        y = self.dropout(x)
        
        y = F.relu(self.fc1_1(y))
        y = F.relu(self.fc1_2(y))
        y = y.transpose(1, 2)                         #Cross enropy wants (minbatch*classes*dimensions)
        
        return y

#====================GCE=====================

class TruncatedLoss(nn.Module):

    def __init__(self, q=0.7, k=0.5, trainset_size=50000):
        super(TruncatedLoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.q = q
        self.k = k
        self.weight = torch.nn.Parameter(data=torch.ones(trainset_size, 1), requires_grad=False).to(self.device)
             
    def forward(self, logits, targets, indexes):
        p = F.softmax(logits, dim=1)
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1)).to(self.device)

        loss = ((1-(Yg**self.q))/self.q)*self.weight[indexes] - ((1-(self.k**self.q))/self.q)*self.weight[indexes]
        loss = torch.mean(loss)

        return loss

    def update_weight(self, logits, targets, indexes):
        p = F.softmax(logits, dim=1)
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        Lq = ((1-(Yg**self.q))/self.q)
        Lqk = np.repeat(((1-(self.k**self.q))/self.q), targets.size(0))
        Lqk = torch.from_numpy(Lqk).type(torch.cuda.FloatTensor)
        Lqk = torch.unsqueeze(Lqk, 1)
        

        condition = torch.gt(Lqk, Lq)
        self.weight[indexes] = condition.type(torch.cuda.FloatTensor)
