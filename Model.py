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
    ypredarr = ypred.clone().detach().cpu().numpy()
    
    ae = np.sum(np.absolute(yarr - ypredarr))
    mae = ae / yarr.flatten().shape[0]
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
        '''(N, S, E)  --> (N, 1, S, S)'''
        f = x.shape[1]
        
        I = torch.ones(f).to(self.device)
        xr = torch.einsum('bfe,h->bhfe', (x, I))   #[x, x, x, x ....]  =>  xr[:,0,:,:] == x
        xc = torch.einsum('bfe,h->bfhe', (x, I))   #[x x x x ....]     =>  xc[:,:,0,:] == x
        diff = xr - xc
        out = torch.einsum('bfge,bfge->bfg', (diff, diff))
        out = out.unsqueeze(1)
        #out = self.bn(out)
        out = F.softmax(-out/13.544, dim = -1)
        return out

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
        x = self.dropout(x)
        return x 

#----------------------------------------------------------------------------

class TransEncoder(nn.Module):
    def __init__(self, d_model, n_head, dim_ff, dropout=0.0, num_layers = 1):
        super(TransEncoder, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model, 0.1, 64)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model = d_model,
                                                    nhead = n_head,
                                                    dim_feedforward = dim_ff,
                                                    dropout = dropout,
                                                    activation = 'relu')
        encoder_norm = nn.LayerNorm(d_model)
        self.trans_encoder = nn.TransformerEncoder(encoder_layer, num_layers, encoder_norm)
                
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
                                padding = (3,1,1),
                                dilation = (3,1,1))
        self.bn1 = nn.BatchNorm3d(512)
        self.pool = nn.MaxPool3d(kernel_size = (1, 7, 7))
        self.sims = Sims()
        
        
        self.conv3x3 = nn.Conv2d(in_channels = 1,
                                 out_channels = 32,
                                 kernel_size = 3,
                                 padding = 1)
        
        self.bn2 = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout(0.25)
        self.input_projection = nn.Linear(self.num_frames * 32, 512)
        self.ln1 = nn.LayerNorm(512)
        
        self.transEncoder = TransEncoder(d_model=512, n_head=4, dropout = 0.2, dim_ff=512, num_layers = 2)
        
        #period length prediction
        self.fc1_1 = nn.Linear(512, 512)
        self.ln2 = nn.LayerNorm(512)
        self.fc1_2 = nn.Linear(512, self.num_frames//2)
        self.fc1_3 = nn.Linear(self.num_frames//2, 1)
    
    def getPeriodicity(self, periodLength):
        periodicity = torch.nn.functional.threshold(periodLength, 2, 0)
        periodicity = -torch.nn.functional.threshold(-periodicity, -1, -1)
        return periodicity
        
        
        
    def forward(self, x, ret_sims = False):
        batch_size, _, c, h, w = x.shape
        x = x.view(-1, c, h, w)
        x = self.resnetBase(x)
        x = x.view(batch_size, self.num_frames, x.shape[1],  x.shape[2],  x.shape[3])
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv3D(x)))
                        
        x = x.view(batch_size, 512, self.num_frames, 7, 7)
        x = self.pool(x).squeeze(3).squeeze(3)
        x = x.transpose(1, 2)                           #batch, num_frame, 512
        x = x.reshape(batch_size, self.num_frames, -1)

        x = F.relu(self.sims(x))
        xret = x
        
        x = F.relu(self.bn2(self.conv3x3(x)))     #batch, 32, num_frame, num_frame
        #print(x.shape)
        x = self.dropout1(x)

        x = x.permute(0, 2, 3, 1)
        x = x.reshape(batch_size, self.num_frames, -1)  #batch, num_frame, 32*num_frame
        x = self.ln1(F.relu(self.input_projection(x)))  #batch, num_frame, d_model=512
        
        x = x.transpose(0, 1)                          #num_frame, batch, d_model=512
        x = self.transEncoder(x)
        y = x.transpose(0, 1)
        
        
        y = F.relu(self.ln2(self.fc1_1(y)))
        y = F.relu(self.fc1_2(y))
        y = F.relu(self.fc1_3(y))    
        
        #y = y.transpose(1, 2)                         #Cross enropy wants (minbatch*classes*dimensions)
        if ret_sims:
            return y, xret
        return y

    
#=============== Atrous COnvs ============================================================================
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv3d(in_channels, out_channels, 3, padding=(dilation, 1, 1), dilation=(dilation, 1, 1), bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            #nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()))

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        #modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv3d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
            #print(res[-1].shape)
        #print(res[0].shape)
        res = torch.cat(res, dim=1)
        #print(res.shape)
        return self.project(res)
    
    
