# 添加NBT attention模型
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2,3'
import torch

import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker
import math

# 下面这几行是为了调用NBT att模型
from keras.models import load_model
from numpy import loadtxt, savetxt
import re

from Attention import Attention_layer


import warnings
warnings.filterwarnings("ignore")

import requests
import re
import json
from bs4 import BeautifulSoup

url = "https://pepcalc.com/ppc.php"

mydata = json.loads(
    '{"hideInputFields": "no","nTerm": "(NH2-)","cTerm": "(-COOH)","aaCode": 0,"disulphideBonds": "","sequence": ""}')

headers = {
    'Content-Type': 'application/x-www-form-urlencoded',
    'Content-Length': '<calculated when request is sent>'
}


mydict = {'A':0,'C':1,'D':2,'E':3,'F':4,'G':5,'H':6,'I':7,'K':8,'L':9,'M':10,'N':11,'P':12,'Q':13,'R':14,'S':15,'T':16,'V':17,'W':18,'Y':19}
myInvDict = dict([val, key] for key, val in mydict.items())
sigmoid = torch.sigmoid


NBTdict = {'A':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'K':9,'L':10,'M':11,'N':12,'P':13,'Q':14,'R':15,'S':16,'T':17,'V':18,'W':19,'Y':20}




MAX_MIC = math.log10(8192)
max_mic_buffer = 0.1
My_MAX_MIC = math.log10(600)


def CosineSimilarity(tensor_1, tensor_2):
    tensor_1 = tensor_1.squeeze()
    tensor_2 = tensor_2.squeeze()

    normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=-1, keepdim=True)
    normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=-1, keepdim=True)
    return (normalized_tensor_1 * normalized_tensor_2).sum()
    
def seq2num(seq):
    
    seqlist = list(seq)
    # print(seq)
    length = len(seq)
    result = re.findall(r'[BJOUXZ]',seq)
    # print(result)
    # 如果序列中有这几个氨基酸，则返回空
    if result:
        return 

    # 否则正常返回
    else:
        numlist = [NBTdict[char.upper()] for char in seqlist]
        
        zeroPad = [0 for i in range(300-length)]
        zeroPad.extend(numlist)
        zeroPad = np.array(zeroPad)
        
        return zeroPad


def dataProcessPipeline(seq):
    # 本函数先把序列转化为0-19组成的序列，然后onehot变化，再padding
    # 同时返回padding后的序列以及mask
    #print('ori seq',seq)
    testest = seq
    num_seq = [mydict[character.upper()] for character in seq]

    seq = np.array(num_seq,dtype=int)
    len = seq.shape[0]
    torch_seq = torch.tensor(seq)
    if torch.sum(torch_seq[torch_seq<0])!=0:
        print(torch_seq[torch_seq<0])
        print('wrong seq:',seq)
        print(testest)
    onehotSeq = torch.nn.functional.one_hot(torch_seq,num_classes=20)
    #onehotSeq = torch.nn.functional.one_hot(c
    pad = torch.nn.ZeroPad2d(padding=(0,0,0,100-len))
    mask = np.zeros(100,dtype = int)
    mask[len:]=1
    mask = torch.tensor(mask)

    pad_seq = pad(onehotSeq) 
    
    
    return pad_seq,mask


def num2onehot(array2d):
    result = torch.zeros_like(array2d)
    index = torch.argmax(array2d,dim = -1)
    for i in range(index.shape[0]):
        result[i,index[i]] = 1

    return result


class TrainDataset(Dataset):
    def __init__(self,data_path,transform = dataProcessPipeline):
        df = pd.read_csv(data_path,header=0)
        
        
        print(str(df.shape)+'\n')
        # df = df[df['Length']<=100]
        self.df = df
        # id = self.df['Length']<100
        # self.df = self.df[id]
        # print(self.df.shape)
        #self.df = self.df.ix[1:]
        # self.seqs = list(self.df['Sequence'])
        self.seqs = list(self.df['sequence'])

        #print(self.seqs.shape)
        self.values = self.df['value']
        # 数据集的单边阈值设置
        self.values[self.values>MAX_MIC] = MAX_MIC
        self.values = list(self.values)
        #print(self.labels.shape)
        self.transform = transform


    def __getitem__(self,idex):
        seq = self.seqs[idex]
        num_seq, mask = self.transform(seq)
        label = self.values[idex]


        return num_seq, mask, label, seq

    def __len__(self):
        return len(self.seqs)


class TestDataset(Dataset):
    def __init__(self,data_path,transform = dataProcessPipeline):
        self.df = pd.read_csv(data_path,header=0)
        self.seqs = self.df['Sequence']

        self.transform = transform


    def __getitem__(self,idex):
        seq = self.seqs[idex]
        num_seq, mask = self.transform(seq)

        return num_seq, mask, seq

    def __len__(self):
        return len(self.seqs)



class PositionalEncoding(nn.Module):
     def __init__(self, len, d_model=20, dropout=0):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(len, d_model)
        position = torch.arange(0, len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                                * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        #pe = pe.unsqueeze(0).transpose(0, 1)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

     def forward(self, x):
        x = x + self.pe
        #x = x + self.pe[:,:x.size(0), :]
        return x



pe = PositionalEncoding(len=100,d_model = 20)


class AttentionNetwork(nn.Module):
    
    def __init__(self,batch_size=128,embedding_size=20,num_tokens=100,num_classes=1,num_heads=4):
        super(AttentionNetwork,self).__init__()
        self.pe = PositionalEncoding(len=num_tokens,d_model = embedding_size)
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.num_tokens = num_tokens
        self.num_classes = num_classes
        self.num_heads = num_heads
        # self.hidden1 = 20
        self.hidden1 = 20
        self.hidden2 = 60
        self.hidden3 = 20
        self.dropout = 0.2

        # self.hidden2 = 100
        # self.hidden3 = 50
        # self.hidden4 = 20
        # self.dropout = 0.2
        self.relu = nn.ReLU()

        self.LN = nn.LayerNorm(normalized_shape = self.hidden1)
        self.fc1 = nn.Linear(self.embedding_size,self.hidden1)

        # self.qfc = nn.Linear(self.hidden1,self.hidden1)
        # self.kfc = nn.Linear(self.hidden1,self.hidden1)
        # self.vfc = nn.Linear(self.hidden1,self.hidden1)

        self.multihead_att = nn.MultiheadAttention(embed_dim=self.hidden1,num_heads = self.num_heads,batch_first=1,dropout=self.dropout)
        # 我这里先不用maxpool了
        self.flatten = nn.Flatten()
        self.fc2 = nn.Linear(self.hidden1*self.num_tokens,self.hidden2)
        self.fc3 = nn.Linear(self.hidden2,self.hidden3)
        self.new_fc4 = nn.Linear(self.hidden3,self.num_classes)
        # self.fc4 = nn.Linear(self.hidden3,self.hidden4)
        # self.fc5 = nn.Linear(self.hidden4,self.num_classes)
        self.dropout = nn.Dropout(self.dropout)
        self.softmax = nn.functional.softmax


    def initialize(self):
        #遍历每一个模块modules
        for m in self.modules():
            #判断模块是否为线性层
            if isinstance(m, nn.Linear):
                ##初始值为标准正态分布，均值为0，标准差为1，但是每层的标准差会越来越大，发生std爆炸
                # nn.init.normal_(m.weight.data) #标准正态分布，均值为0，标准差为1，标准差会发生爆炸
                
                #此时的初始化权值可以使每层的均值为0，std为1.
                # nn.init.normal_(m.weight.data, std=np.sqrt(1/self.neural_num))   

                #手动计算
                # a = np.sqrt(6 / (self.neural_num + self.neural_num))
                #计算激活函数的增益
                # tanh_gain = nn.init.calculate_gain('tanh')
                # a *= tanh_gain
                #设置均匀分布来初始化权值
                # nn.init.uniform_(m.weight.data, -a, a)

                #自动计算
                nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('tanh'))

                #手动计算
                # nn.init.normal_(m.weight.data, std=np.sqrt(2 / self.neural_num))

                #自动计算
                # nn.init.kaiming_normal_(m.weight.data)

    #这里的X当作对象 有 embedding 和 mask
    def forward(self,x,mask):
        x = self.pe(x)
        x = self.fc1(x)


        mask = mask.to(torch.bool)
        x, w1= self.multihead_att.forward(x,x,x,key_padding_mask=mask)
        # x, w1= self.multihead_att.forward(x,x,x,key_padding_mask=mask)
        # x, w1= self.multihead_att.forward(x,x,x,key_padding_mask=mask)


        # x = self.LN(x)
        # [N 100 20]

        
        
        #print(type(x),x.size)
        # x = torch.tensor(x)
        x = self.flatten(x)
        x = self.fc2(x)
        x = self.dropout(x)
        
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.new_fc4(x)
        # 输出的单边阈值设置
        # x[x>MAX_MIC] = MAX_MIC
        # x = self.LN4(x)
        # print(x.shape)
        # x = self.softmax(x)
        #print(x.shape)

        #return x, w1, w2, w3, w4
        return x

def num2seq(narr,len):
    '''
    narr:(100,26)
    '''
    
    numlist = np.argmax(narr,axis = 1)
    seq = [myInvDict[value] for value in numlist]
    seq = seq[:len]
    return seq


def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {'black': '\033[30m',  # basic colors
            'red': '\033[31m',
            'green': '\033[32m',
            'yellow': '\033[33m',
            'blue': '\033[34m',
            'magenta': '\033[35m',
            'cyan': '\033[36m',
            'white': '\033[37m',
            'bright_black': '\033[90m',  # bright colors
            'bright_red': '\033[91m',
            'bright_green': '\033[92m',
            'bright_yellow': '\033[93m',
            'bright_blue': '\033[94m',
            'bright_magenta': '\033[95m',
            'bright_cyan': '\033[96m',
            'bright_white': '\033[97m',
            'end': '\033[0m',  # misc
            'bold': '\033[1m',
            'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


def standout(seq1,seq2):

    # 双序列比对变化
    index = [1 if seq1[j]==seq2[j] else 0 for j in range(len(seq1))]
    newSeq2 = list(seq2)
    for i in range(len(seq1)):
        if index[i] == 0:
            newSeq2[i] = colorstr('blue',seq2[i])

    newSeq2 = ''.join(newSeq2)
    
    return seq1,newSeq2

def colorPrint(ls):
    newls = [ls[0]]
    for i in range(len(ls)-1):
        s1,s2 = standout(ls[i],ls[i+1])
        newls.append(s2)

    for i in newls:
        print(i)


def colorShow(ls):

    ## 累计变化
    length = len(ls[0])
    colorls = [ls[0]]
    flag = [0 for i in range(length)]  # 记录是否发生突变
    for i in range(len(ls)-1):
        index = [1 if ls[i][j]==ls[i+1][j] else 0 for j in range(length)]
        for k in range(length):
            if index[k] == 0:
                flag[k] = 1

        colorSeq = list(ls[i+1])
        for k in range(length):
            if flag[k]==1 :
                colorSeq[k] = colorstr('blue',colorSeq[k])

        colorSeq = ''.join(colorSeq)
        colorls.append(colorSeq)


    for seq in colorls:
        print(seq+',10,')
                


# model_list = {
#     # 
#     'CNN':'../../NewModel3_17_output1_Regression/CNN/mean_0_0_0/best_model0.733.pth',
#     'Transformer':'../../NewModel3_17_output1_Regression/Transformer/mean_0_0_0/0.676.pth',
#     'myAttention': '../../NewModel3_17_output1_Regression/myAttention/mean_1_0_0/0.671.pth',

#     'RCNN': '../../NewModel3_17_output1_Regression/RCNN/mean_0_0_0/0.744.pth'

# }
# # 3.21 重新训练调整后的版本
# model_list = {
#     # 
#     'CNN':'../../NewModel3_21_output1_Regression/CNN/mean_0_changeTH_0_0/_AMP0.591_total0.63.pth',
#     'Transformer':'../../NewModel3_21_output1_Regression/Transformer/mean_0_changeTH_0_0/_AMP0.783_total0.601.pth',
#     'myAttention': '../../NewModel3_21_output1_Regression/myAttention/mean_0_changeTH_0_0/_AMP0.629_total0.543.pth',
#     'RCNN': '../../NewModel3_21_output1_Regression/RCNN/mean_0_changeTH_0_0/_AMP0.689_total0.615.pth'

# }

model_list = {
    # 
    'myAttention': '../../NewModel3_21_output1_Regression/myAttention/mean_0_changeTH_0_0/_AMP0.629_total0.543.pth'
}


# model_list  = {
#     # 
#     'CNN':'../../NewModel3_21_output1_Regression/CNN/mean_0_changeTH_0_0/_AMP0.591_total0.63.pth',
#     'Transformer':'../../NewModel3_21_output1_Regression/Transformer/mean_0_changeTH_0_0/_AMP0.783_total0.601.pth',
#     'myAttention': '../../NewModel3_21_output1_Regression/myAttention/mean_0_changeTH_0_0/_AMP0.629_total0.543.pth',
#     'RCNN': '../../NewModel3_21_output1_Regression/RCNN/mean_0_changeTH_0_0/_AMP0.689_total0.615.pth'

# }



test_model_list  = {
    # 
    'CNN':'../../NewModel3_21_output1_Regression/CNN/mean_0_changeTH_0_0/_AMP0.591_total0.63.pth',
    'Transformer':'../../NewModel3_21_output1_Regression/Transformer/mean_0_changeTH_0_0/_AMP0.783_total0.601.pth',
    'myAttention': '../../NewModel3_21_output1_Regression/myAttention/mean_0_changeTH_0_0/_AMP0.629_total0.543.pth',
    'RCNN': '../../NewModel3_21_output1_Regression/RCNN/mean_0_changeTH_0_0/_AMP0.689_total0.615.pth'

}

# 应该是要在函数外就加载模型
# model_list_dict = {
#     'CNN':torch.load(model_list['CNN']),
#     'Transformer':torch.load(model_list['Transformer']),
#     'myAttention': torch.load(model_list['myAttention']),
#     'RCNN': torch.load(model_list['RCNN']),
#     'NBT':load_model('/home/user2/pj/AMP_2/src/c_AMPs-prediction/Models/att.h5', custom_objects={'Attention_layer': Attention_layer})
# }

# model_list_dict = {
#     'CNN':torch.load(model_list['CNN']),
#     'Transformer':torch.load(model_list['Transformer']),
#     'myAttention': torch.load(model_list['myAttention']),
#     'RCNN': torch.load(model_list['RCNN']),
# }


def ensamble_grad(x,mask,center_model,model_dict):
    # 给定数据向量x和对应的mask，并指定对应的核心模型，用该模型进行梯度迭代优化，如果seq存在，即x为onehot vector
    # for normal model
    preList = []
    NBTpred = -1
    for k,v in model_dict.items():
        if center_model != k and 'NBT' not in k: # 非中心模型
            attmodel = v
            attmodel.eval()
            if 'myAttention' not in k: # 不是attention模型
                out = attmodel(x)
            else: # attention模型
                out = attmodel(x,mask)
            out = torch.squeeze(out)
            out = out.cpu()
            preList.append(out.data)

    # for center_model
    attmodel = torch.load(model_list[center_model])
    if 'RCNN' not in center_model:
        attmodel.train()

    attmodel.zero_grad()
    x.retain_grad = True
    if 'myAttention' not in k: # 不是attention模型
        out = attmodel(x)
    else:
        out = attmodel(x,mask)


    out = torch.squeeze(out)
    out = out.cpu()
    ensamble_value = (preList[0]*preList[1]*preList[2]*out)**(1/4)
    ensamble_value.backward()
    grad = x.grad

    return grad, ensamble_value




def ensamble_grad2(x,mask,model_dict):
    # 给定数据向量x和对应的mask,求出所有模型导数的平均


    grads = {}
    outs = []
    for center_model,v in model_dict.items():
        grads[center_model] = []
        attmodel = v
        if 'RCNN' in center_model:
            attmodel.train()

        attmodel.zero_grad()
        x.retain_grad = True
        # x.zero_grad()
        # print(x.grad)
        if 'myAttention' not in center_model: # 不是attention模型
            out = attmodel(x)
        else:
            out = attmodel(x,mask)

        out = torch.squeeze(out)
        out = out.cpu()
        outs.append(out)
        out.backward()
        # print(x.grad)
        with torch.no_grad():
            grads[center_model] = x.grad.data.clone()

    grad_ls = list(grads.values())
    diffs = {}
    for k,v in grads.items():
        diff = [CosineSimilarity(grads[k],grad_ls[i]).data for i in range(len(grad_ls))]
        # print(k,':',diff)
    mean_grad = (grad_ls[0]+grad_ls[1]+grad_ls[2]+grad_ls[3])/4
    return mean_grad, outs


            




# opt_seqls0 = [

# 'RLRLLSTLLK',
# 'RKKVHGFRKR',
# 'FLTLLLRWFH',
# 'KLGRLAFRIPWRH',
# 'RRLRPAFKKVGVAG',
# 'CGRPHSVYRKFKL',
# 'DAHLMKKVEEQAESTKKQVIK',
# 'GGDDTLFALVDGVVRFERKGRDKK',

# 'AFLRRLGRR',
# 'FLRRLGRRA',
# 'SLAAAAAKR',
# 'LNAKVVELKEELFGLRFAA',
# 'LTVRVSGYAVNFVKLTK',
# 'IKLRSTAGTGYTYVTRKN',
# 'RVRATVNGAPKRLNVCTSCLKAGKV',
# 'LNAKVVELKEELFGLRFAAAT',

# 'FLLPGLKLL',
# 'ALRHLLVRLL',
# 'LLRLNLMRKA',
# 'LAILKKALLLLEKI',
# 'MLLLLVKFAKIHA',
# 'AAWRAYLEAKLSF',
# 'KRCAITGKGPMVGNNVSHANNKTKR',
# 'GKGPMVGNNVSHANNKTKRRF',
# 'SKRCAITGKGPMVGNNVSHANNKT',

# 'FTLLAALLRR',
# 'RKFAVLPRY',
# 'RGAGAALLR',
# 'KIIFTLLAALLRR',
# 'KLLGAFGKPTKIA',
# 'LKLFLLASLVKK',

# 'RVLAALARLR',
# 'RALVTLRRLR',
# 'RALRRVLAALARLR',
# 'LLPALAKGAHHTHT',
# 'FLANLTALRRAKTRA',

# 'RRLRLLVAPR',
# 'KGVLGLKRR',
# 'SLLAALLRI',
# 'FHRAKRLLDALLK',
# 'MHLAELANLLKRI',
# 'IVLSGFLKYAKT',

# 'FLTLLLRKH',
# 'VKFLTLLLRK',
# 'IKLLLMRKNL',
# 'LLVALGRFRVKIRF',
# 'VGHKLGEFAPTRTYKGHV',
# 'ILSLLLPLKITLI',
# 'HKLGEFAPTRTYKGHVADDRK',
# 'DMVGHKLGEFAPTRTYKGHAA',

# 'KLVLYLKKL',
# 'LKKNLYLLKH',
# 'MVGHKLGEFAPTRTYKGHAA',
# 'ASKVPAFKAGKALK',
# 'KVPAFKAGKALKDAVK',
# 'LAKGEKIQIIGFGNFEVRERAARK',
# 'LAKGEKIQIIGFGNFEVRERAARK',
# 'EIAASKVPAFKAGKALKDAVK',

# 'RQKVHGFRKR',
# 'QKVHGFRKR',
# 'RCERCGRPHSVYRKFKLCRI',
# 'KASKVPAFKAGKALKDAVK',
# 'KSLLPLLRIVQII',
# 'KIKASKVPAFKAGKALKDAVKK',
# 'GKEIKIKASKVPAFKAGKALKD',
# 'EIKIKASKVPAFKAGKALKDA',

# 'KLRLVLTLLR',
# 'LLVFLNKHRK',
# 'MLLRLLNLALPRK',
# 'TRCERCGRPHSVYRKFHLC',
# 'LLTLTLRKRLR',
# 'LAKGEKVQLIGFGNFEVRERAARK',
# 'RFQLATGQLENTARLKQVRKN'

# ]

# opt_seqls1 = [
# 'RLNLLLGRRK',
# 'LLPLLAYLRR',
# 'FIVLSGFLK',
# 'ARRLGSWLALRLS',
# 'LWMRLGAAGLKAP',
# 'LAELANLLKRITD',
# 'FLTLLLRKH',
# 'FLTLLLRKH',
# 'KLLNVVLEKK',
# 'KKLLNVVLEKKLL',
# 'YLRSYALLLLQKLR',
# 'HKLGEFAPTRTYKGHAADDR',
# 'HKLGEFAPTRTYKGHVADDRK',
# 'GHKLGEFAPTRTYKGHVADDR',
# 'AFLRRLGRRA',
# 'FQLRHLLVR',
# 'FGVRAYTRC',
# 'KLRSTAGTGYTYVTRKNR',
# 'PLIKLRSTAGTGYTYVTRK',
# 'KLGEFAPTRTFRGHVKEDK',
# 'RVRATVNGAPKRLNVCTSCLKAGKV',
# 'RVRATVNGAPKRLNVCTSCLKAGK',
# 'GDRVRVMETRPLSAQKRWRLV',
# 'KLRLVLTLLR',
# 'LAQQLLAKRL',
# 'KLRLVLTLLRLAR',
# 'KRHRQRVHGFRKRM',
# 'KNESLDDALRRFKRTVSKS',
# 'LAKGEKVQLIGFGNFEVRERAARK',
# 'DTLAKGEKVQLIGFGNFEVRE',
# 'VLAALARLR',
# 'ALHLGGLKIR',
# 'FIKVAPAVAK',
# 'AWAREIGYMFGQYKKLTGR',
# 'ALHLGGLKIRKI',
# 'EIGYMFGQYKKLTGRFDGIL',
# 'FTLLAALLRR',
# 'FTLLAALLRR',
# 'HLLANFKRVR',
# 'KIIFTLLAALLRR',
# 'GAHLLANFKRVRA',
# 'EGRAELAALAKQH',
# 'ALLLLEKLR',
# 'ALLLLEKLRL',
# 'VSLFLKKNLYLLKH',
# 'LSVLLLEKVKIQKK',
# 'SYALLLLEKLRL',
# 'LAKGEKIQIIGFGNFEVRERAARK',
# 'LAKGEKIQIIGFGNFEVRERAARK',
# 'GEKIQIIGFGNFEVRERAARK',
# 'CGRPHSVYRK',
# 'LWKTLLKDGK',
# 'RCERCGRPHSVYRKFKLCRI',
# 'GSSQIIKMKLTNVFRKL',
# 'VVRKNESLDDALRRFKRT',
# 'KIKASKVPAFKAGKALKDAVKK',
# 'IKIKASKVPAFKAGKALKDAVK',
# 'GRQKFTQADGRVDRFNKKYGL',
# 'RKKVHGFRKR',
# 'RRLRPAFKKV',
# 'LGRLAFRIPW',
# 'KLGRLAFRIPWRH',
# 'LMKKVEEQAESTKKQVIKT',
# 'ERCGRPHSVYRKFK',
# 'FCDAHLMKKVEEQAESTKKQV',
# 'GGDDTLFALVDGVVRFERKGRDKKQ',
# 'ALRHLLVRL',
# 'FLLPGLKLL',
# 'RQVHRLLRK',
# 'LAILKKALLLLEKI',
# 'KALSLPVKARKRI',
# 'KFASLALALRL',
# 'KRCAITGKGPMVGNNVSHANNKTK',
# 'TGKGPMVGNNVSHANNKTKRR',
# 'SKRCAITGKGPMVGNNVSHANNK'
# ]

# opt_seq2 = list(pd.read_csv('/home/user2/pj/AMP_2/attention_model/k_mer_selection/k_mer/5_20_species_selection_result/species_of_best20_longer14.csv',header=0)['sequence'])

# opt_seq0 = list(pd.read_csv('/home/user2/pj/AMP_2/attention_model/k_mer_selection/k_mer/5_20_species_selection_result/5_21_speciesRandom1_2.csv',header=0)['sequence'])
# opt_seq1 = list(pd.read_csv('/home/user2/pj/AMP_2/attention_model/k_mer_selection/k_mer/5_20_species_selection_result/5_21_speciesRandom_522_2.csv',header=0)['sequence'])

opt_seq = list(pd.read_csv('/home/user2/pj/AMP_2/attention_model/k_mer_selection/k_mer/5_20_species_selection_result/3_29_kmer_orfResult_ensamble_cla_reg_520_after_filter_with_species.csv',header=0)['sequence'])

opt_seq0 = [i if '.csv' not in i else ' ' for i in os.listdir('./optimizedSeq_5_2_species_random')]

opt_seq1 = [i if '.csv' not in i else ' ' for i in os.listdir('./optimizedSeq_5_22_species_random_2')]

opt_seq2 = [i if '.csv' not in i else ' ' for i in os.listdir('./optimizedSeq_5_24_species_random_larger')]

opt_seq3 = [i if '.csv' not in i else ' ' for i in os.listdir('./optimizedSeq_5_2_species_best')]

opt_seq4 = [i if '.csv' not in i else ' ' for i in os.listdir('./optimizedSeq_5_2_species_best_3_longer_att')]

opt_seq5 = [i if '.csv' not in i else ' ' for i in os.listdir('./optimizedSeq_5_2_species_best_4_longer14_att')]

opt_seq6 = [i if '.csv' not in i else ' ' for i in os.listdir('./optimizedSeq_5_2_species_best_4models_opt')]

opt_seq7 = [i if '.csv' not in i else ' ' for i in os.listdir('./optimizedSeq_5_2_species_best_20')]

opt_seq8 = [i if '.csv' not in i else ' ' for i in os.listdir('./optimizedSeq_5_24_species_random_larger')]

opt_seqls1 = list(set(opt_seq)-(set(opt_seq0)&set(opt_seq1)&set(opt_seq2)&set(opt_seq3)&set(opt_seq4)&set(opt_seq5)&set(opt_seq6)&set(opt_seq7)&set(opt_seq8)))
opt_seqls1 = list(set(opt_seqls1))
Len = [len(v) for v in opt_seqls1]
flag = [1 if v>=14 else 0 for v in Len]
print('the number of seq:',len(opt_seqls1))

opt_seqls = []
for j in opt_seqls1:
    if len(j)>=14:
        opt_seqls.append(j)

# opt_seqls = list(set(opt_seq0)-(set(opt_seqls1)&set(opt_seqls0)))
# print(opt_seq2)
# opt_seqls = list(set(opt_seq2)-set(opt_seq0)&set(opt_seq1))
print('the number of seq:',len(opt_seqls))
opt_seqls = opt_seqls[-2000:-1000]
print('the number of seq:',len(opt_seqls))
# ori_seq
# opt_seqls = [

# 'RCERCGRPHSVYRKFKLCRI',
# 'KIIFTLLAALLRR',
# 'KLRLVLTLLRLAR',
# 'RCERCGRPHSVYRKFHLCRI',
# 'RRALRRVLAALARL',
# 'ALRRVLAALARLR',
# 'RALRRVLAALARLR',
# 'KLGRLAFRIPWRH',
# 'RCERCGRPHSVYRKFKLCR',
# 'KKSLKRSLALALR',
# 'LAILKKALLLLEK',
# 'AILKKALLLLEKI',
# 'LLVALGRFRVKIR',
# 'KKLLNVVLEKKLL',
# 'MRAAFLRRLGRRA',
# 'KLRSTAGTGYTYVTRKNRR',
# 'RARRLGSWLALRL',
# 'RRLRLLVAPRRLK',
# 'RRLRLLVAPRRLR',
# 'QILRWRLTLALAR'

# ]


# opt_seqls = [
# 'RCERCGRPHSVYRKFKLCRI',
# 'RCERCGRPHSVYRKFKLCR',
# 'RCERCGRPHSVYRKFKLC',
# 'QTGKEIKIKASKVPAFKAGKA',
# 'FPKKDRVIVEGVNIVKKHQ',
# 'FPKKDRVIVEGVNIVKKH',
# 'RCERCGRPHSVYRKFHLCRI',
# 'RCERCGRPHSVYRKFHLCR',
# 'QNYTRCERCGRPHSVYRKF',
# 'RCERCGRPHSVYRKFHLC',
# 'CGRPHSVYRKFHLCRICLR',
# 'AWAREIGYMFGQYKKLTGR',
# 'MVGHKLGEFAPTRTYKGHA',
# 'MVGHKLGEFAPTRTYKGH',
# 'IEIAASKVPAFKAGKALK',
# 'RCERCGRPHSVYRKFKLCRI',
# 'RCERCGRPHSVYRKFKLCR',
# 'RCERCGRPHSVYRKFKLC',
# 'YTRCERCGRPHSVYRKFKLC',
# 'FTQADGRVDRFNKKYGFNK',
# 'YTRCERCGRPHSVYRKFK',
# 'MVGHKLGEFAPTRTYKGHA',
# 'KLRSTAGTGYTYVTRKNRR',
# 'RPLIKLRSTAGTGYTYVTRK'
# ]


iter_dict = {'CNN': 500,'Transformer':500,'myAttention':500,'RCNN':500}
lr_dict = {'CNN': 0.01,'Transformer':0.0005,'myAttention':0.005,'RCNN':0.001}
for seq in opt_seqls:
    tseq = seq

    # ModelNameList = ['CNN','Transformer','myAttention','RCNN']
    ModelNameList = model_list.keys()

    iters =200
    lambda1 = 0.5
    alpha = 0.01
    

    oriseq = tseq

    # 在当前目录生成 该序列的csv 文件
    df = pd.DataFrame(columns = ['Sequence','Length','label'])
    items = [{'Sequence':oriseq,'Length':len(oriseq)}]
    df = df.append(items,ignore_index = 1)
    df.to_csv('tempt/'+oriseq+'.csv',index = False)


    SeqPath = 'tempt/'+oriseq+'.csv'



    testData1 = TrainDataset(data_path = r'../../myRegressionData/all_balance/mean/test.csv')
    test_loader1 = DataLoader(dataset=testData1, batch_size=4,drop_last=True)


    for modelName in ModelNameList:
        alpha = lr_dict[modelName]
        iters = iter_dict[modelName]
        # modelName = 'myAttention'  # to change
        iternum = 0
        # to change
        # attmodel = torch.load('../../newModel11_3/lstm_att/0/best_model0.97.pth')     #lstm_att
        # attmodel = torch.load('../../newModel/RCNN/0/best_model0.906.pth') 
        # attmodel = torch.load('../../NewModel11_08_output1/CNN/2_0/best_model0.97_0.966.pth')     #CNN

        # attmodel = torch.load('../../NewModel11_08_output1/Transformer/2_0/0.963_0.971.pth')    # Transformer

        testData = TestDataset(data_path = SeqPath)
        test_loader = DataLoader(dataset=testData, batch_size=1)
        attmodel = torch.load(model_list[modelName])


        attmodel.cuda()
        attmodel.zero_grad()

            
        
        def score(test_loader):
            attmodel.eval()
            epi = 0.000001
            tp = 0
            tn = 0
            fp = 0
            fn = 0
            total = 0
            count = 0

            tp1 = 0
            tn1 = 0
            fp1 = 0
            fn1 = 0
            total1 = 0
            count1 = 0

            for data in test_loader:
                inputs,masks,labels, _ = data
                inputs = inputs.float()
                masks = masks.float()
                labels = labels.float()
                inputs,masks,labels = Variable(inputs),Variable(masks),Variable(labels)

                inputs = inputs.cuda()
                masks = masks.cuda()
                # out,_,_,_,_ = attmodel(inputs,masks)
                if modelName != 'myAttention' and modelName != 'Transformer2':
                    out = attmodel(inputs)
                else:
                    out = attmodel(inputs,masks)
                out = torch.squeeze(out)
                # out = sigmoid(out)
                out = out.cpu()
                # out = torch.round(out)
                out = torch.squeeze(out)
                for i,pre in enumerate(out):
                    total += 1
                    if abs(pre-labels[i])<0.5:
                        count += 1
                        if pre>labels[i]:
                            tn += 1
                        else:
                            tp += 1
                    if abs(pre-labels[i])>=0.5:
                        if pre>labels[i]:
                            fn += 1
                        else:
                            fp += 1

                    total1 += 1
                    if labels[i]<My_MAX_MIC: # AMP
                        if pre<My_MAX_MIC:
                            tp1 += 1
                            count1 += 1
                        else:
                            fn1 += 1

                    else: #nonAMP
                        if pre<My_MAX_MIC:
                            fp1 += 1 
                        else:
                            tn1 += 1
                            count1 += 1


            print("AMP回归结果:"+'\n')
            print("大致正确的范围，但是比较大胆:"+str(tp)+'\n')
            print("大致正确的范围，但是比较保守:"+str(tn)+'\n')
            print("错误的，但是比较大胆:"+str(fp)+'\n')
            print("错误的，但是比较保守"+str(fn)+'\n')
            print("准确率："+str(count/total)+'\n')

            print("AMP分类结果:"+'\n')
            print("精度:"+str(tp1/(tp1+fp1+epi))+'\n')
            print("回归率:"+str( tp1/(tp1+fn1+epi))+'\n')
            print("特异性:"+str(tn1/(tn1+fp1+epi))+'\n')
            print("F1值:"+str(2*tp1/(2*tp1+fp1+fn1+epi))+'\n')
            print("准确率："+str(count1/total1)+'\n')

            # precise = tp/(tp+fp+epi)

            # return precise


        score(test_loader1) # 对模型进行评估

        writer1 = SummaryWriter("./board/1/loss")

        
        # attmodel.train()
        print(modelName,"_V2:")
        flag = 1
        
        for data in test_loader: #序列优化 stratergy 1: 全局都用ensamble作为优化目标
            resultList = []
            # ensamble_values = []
            resultSeq = [oriseq]
            outMIC = []
            # attmodel.zero_grad()
            inputs,masks, seqs = data

            inputs = inputs.float()
            masks = masks.float()
            
            inputs = inputs.cuda()
            inputs.requires_grad = True
            masks = masks.cuda()
            print(seqs[0])

            if modelName == 'RCNN':
                attmodel.train()
            else:
                attmodel.eval()
            for iter in range(iters):
                attmodel.zero_grad()
                inputs.retain_grad = True
                
                if modelName == 'lstm_att' or modelName == 'RCNN'or modelName == 'CNN' or modelName == 'Transformer':
                    out = attmodel(inputs)
                else:
                    out = attmodel(inputs,masks)

                # out = torch.squeeze(out)
                out = out.cpu()
                conloss = out 
                conloss.backward()
                grad = inputs.grad

                colindex = masks[0]==1
                grad[0][masks[0]==1] = 0
                mylen = 100-colindex.sum()

                ori_onehot = num2onehot(inputs[0].cpu())
                result = inputs[0]- alpha*grad[0]
                result[mylen:,:] = 0
                tempt_onehot = num2onehot(result.cpu())

                if (tempt_onehot == ori_onehot).all(): # 未能引起氨基酸序列发生变化
                    flag = 0
                else: # 发生了变化，直接把新的onehot 赋值给result，作为下一步更新迭代的起点
                    # tempt_onehot[mylen:,:] = 0
                    result = tempt_onehot
                    # ensamble_values.append(ensamble_value)
                    # print('New!')
                    flag = 1

                with torch.no_grad():
                    inputs[0] = result
                # writer1.add_scalar('loss2', ensamble_value,iter)
                # writer1.add_scalars('loss',conloss,iter)
                # result = torch.softmax(result,dim = 2)

                # 下面几行只是把result进行保存，并输出对应的氨基酸序列而已
                result = result.cpu().detach().numpy()

                seq = num2seq(result,len = mylen)
                seq = ''.join(seq)
                if flag==1:
                    resultSeq.append(seq)
                # print(seq)
            # for i in resultSeq:
            #     print(i)

            writer1.close()

        print(modelName)
        colorShow(resultSeq)
        print()

        ## 保存优化的序列

        optSeqDir = './optimizedSeq_5_24_species_-2000/'+oriseq
        if not os.path.exists(optSeqDir):
            os.makedirs(optSeqDir)
        optSeqSavePath = optSeqDir+'/'+modelName+'.csv'

        result_df = pd.DataFrame(columns=['Sequence','label','Length'])
        items = []
        for seq in resultSeq:
            item = {'Sequence':seq, 'Length': len(seq)}
            items.append(item)

        result_df = result_df.append(items)
        # if result_df.shape[0]>10:
        #     result_df = result_df[:10]
        result_df.to_csv(optSeqSavePath)

        result_soli = []
        # 以上完成了序列优化的部分

        # 接下来是水溶性的预测
        # seqls = result_df['Sequence']
        # for i,seq in enumerate(seqls):
        #     mydata['sequence'] = seq
        #     res = requests.post(url, headers=headers, data=mydata)
        #     soup = BeautifulSoup(res.text, "html.parser")
        #     if soup.find(string=re.compile(r"Estimated solubility")):
        #         fresult = (soup.find(string=re.compile(r"Estimated solubility")
        #                             ).find_parent("td").find_next_sibling("td").text)+"\n"
        #         if 'Good' in fresult:
        #             result_soli.append(1)
        #         else:
        #             result_soli.append(0)

        #         if i%100==0:
        #             print(i)
        #     else:   
        #         result_soli.append(' ')
        #         print(seq)

        # result_df['water solubility'] = result_soli


        # 接下来是使用验证模型对抗菌性进行预测
        
        '''
        读什么文件
        用什么模型
        '''

        testModelNameList = test_model_list.keys()

        # testModelName = 'myAttention'
        preList = {}
        # 先试 NBT模型结果

        mylen = result_df.shape[0]
        numslist = []
        # for id in range(mylen):
        #     seq = result_df.loc[id]['Sequence']
        #     numlist = seq2num(seq)
        #     if numlist is not None:
        #         numslist.append(numlist)
        # numslist = np.array(numslist)
        # model = load_model('/home/user2/pj/AMP_2/src/c_AMPs-prediction/Models/att.h5', custom_objects={'Attention_layer': Attention_layer})
        # x = numslist

        # preds = model.predict(x)
        # result_df['NBT att'] = [round(v[0],3) for v in list(preds)]

        # ## 对上面的结果进行一下输出：
        # print('NBT att :')
        # seqs = result_df['Sequence']
        # for i in range(len(seqs)):
        #     print(seqs[i],'predict:',preds[i])

        model_out = {}
        for testModelName in testModelNameList:
            ls = []
            model_out[testModelName] = []
            testData = TestDataset(data_path = optSeqSavePath)
            test_loader = DataLoader(dataset=testData, batch_size=64)


            testData1 = TrainDataset(data_path = r'../../myRegressionData/all_balance/mean/test.csv')
            test_loader1 = DataLoader(dataset=testData1, batch_size=4,drop_last=True)


            attmodel = torch.load(test_model_list[testModelName])


            attmodel.cuda()
            attmodel.zero_grad()


            # score()

            writer1 = SummaryWriter("./board/1/loss")


            # print(optSeqSavePath)
            # print(testModelName)

            attmodel.eval()
            for data in test_loader:
                resultList = []
                attmodel.zero_grad()
                inputs,masks, seqs = data
                
                # print(inputs.is_leaf)
                # print(inputs._version)
                inputs = inputs.float()
                # print(inputs.is_leaf)
                # print(inputs._version)
                masks = masks.float()
                
                inputs = inputs.cuda()
                # inputs.requires_grad = True
                # print(inputs.is_leaf)
                # print(inputs._version)
                masks = masks.cuda()

                # for iter in range(iters):
                attmodel.zero_grad()
                # inputs.retain_grad = True
                if ('lstm_att' in testModelName )or ('RCNN' in testModelName) or ('CNN' in testModelName) or ('Transformer' in testModelName):
                    out = attmodel(inputs)
                else:
                    out = attmodel(inputs,masks)
                # out,w1 = attmodel(inputs,masks)
                # out = torch.squeeze(out)
                out = out.cpu()
                # if out.shape[0] != 1:
                if len(out.shape)>0:
                    out_ori = torch.squeeze(out)
                else:
                    out_ori = out.unsqueeze(0)
                # out_numpy = list(out.detach().numpy())
                # out_numpy = [np.round(v,3) for v in out_numpy]
                # model_out[testModelName] = list(model_out[testModelName])+out_numpy
                out_numpy = list(out_ori.detach().numpy())
                out_numpy = [round(v,3) for v in out_numpy]
                model_out[testModelName] = list(model_out[testModelName])+out_numpy
                
                # items = []
        
        for k,v in model_out.items():
            result_df[k] = v
        print(result_df['myAttention'][0])
        resultPath =  optSeqSavePath[:-4]+'_result.csv'


        ensamble_values =[(result_df['myAttention'][k]+result_df['Transformer'][k]+result_df['CNN'][k]+result_df['RCNN'][k])/4 for k in range(result_df.shape[0])]
        ensamble_values = [round(v,3) for v in ensamble_values]
        result_df['esb_reg'] = ensamble_values
        # result_df = result_df[['Sequence','esb_reg','NBT att','Length','water solubility','CNN','Transformer','myAttention','RCNN']]
        # result_df = result_df[['Sequence','esb_reg','NBT att','Length','CNN','Transformer','myAttention','RCNN']]

        result_df = result_df[['Sequence','esb_reg','Length','CNN','Transformer','myAttention','RCNN']]

        result_df.to_csv(resultPath,index=0)

        # 把所有模型的结果：绘制成图片
        import numpy as np
        import matplotlib.pyplot as plt 

        x=np.arange(result_df.shape[0])

        plt.plot(x,result_df['myAttention'],label = 'Attention')
        plt.plot(x,result_df['Transformer'],label = 'Transformer')
        plt.plot(x,result_df['CNN'],label = 'CNN')
        plt.plot(x,result_df['RCNN'],label = 'RCNN')
        plt.plot(x,result_df['esb_reg'],label = 'ensamble_values')

        plt.title(modelName+" optimization result",fontsize=15)
        plt.xlabel("X",fontsize=13)
        plt.ylabel("score",fontsize=13)
        plt.legend()
        plt.show()
        # plt.savefig(os.path.join(optSeqDir , modelName+" optimization result.png"))
        plt.cla()

