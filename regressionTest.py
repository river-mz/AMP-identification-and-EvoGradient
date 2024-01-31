#this code functions as result saver. It can save the output of model prediction scores of sequence.

#导入模块
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6,7'
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

from otherModelsRegression import CNN, RCNN, lstm_att, Transformer, Transformer2


import requests
import re
import json
from bs4 import BeautifulSoup
import pandas as pd
from keras.models import load_model
from numpy import loadtxt, savetxt
import re
from Attention import Attention_layer

# seqLs = [
# 'TLLXXXKK',
# 'XKKXTGLX',
# 'XKKXTGL',
# 'XXKKXTGL',
# 'KKXTGLX',
# 'WXAXXMY',
# 'MYXXVXLK',
# 'WWXXXKWW',
# 'XXKWWXKK',
# 'KWWXKKXX',
# 'XKKLXRKX',
# 'XXKKLXRK'
# ]

# seqLs = [
# 'LWXRKXKW',
# 'AKXMIXX',
# 'KWXGXFX',
# 'AIRXRLXX',
# 'WXRXXXTLL',
# 'KKXXWXA',
# 'MMYXXVKLK',
# 'MXXARFX',
# 'ARFXWWXX',


# ]

# seqLs = [
# 'XKWWXKKX',

# 'WWXXXKWW',

# 'KWWXKKXX',

# 'LWXRKXKW',

# 'WKKLXRKXA',

# 'WXKLXRKX',

# 'WWRXWKXLXK',

# 'WRLXXXLL'

# ]

seqLs = [
'XXKWWXKKXX',

'XWWXXXKWWX',

'XKWWXKKXXX',

'XLWXRKXKWX',

'XWKKLXRKXAX',

'XWXKLXRKXX',

'XWWRXWKXLXKX',

'XWRLXXXLLX'

]


myseq = seqLs[7]


peptideLs = [
'KWLGAFGKMRKIAIRLRLKRKKAF',
'LWWRKAKWKRKIAKRMIRVIGAAKI',
'MKKARFWWWVAWKKLLRKKA',
'MRFPWKHWWKKWKWWWKKKR',
'RKLKKLRWRAGMMYKYVKLK',
'WWRLWKTLLKAPKKLTGLRRW'
]
peptide = peptideLs[4]

alpha = 0.03

batch_size = 256
ebedding_size=20
num_tokens=100
num_classes=2
num_heads=4
lr = 0.001

MAX_MIC = math.log10(8192)
max_mic_buffer = 0.1
My_MAX_MIC = math.log10(600)

nameList = ['CNN','Transformer','myAttention','RCNN']

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



# consevative models
# model_list = {
#     # 
#     'CNN':'./NewModel3_17_output1_Regression/CNN/mean_0_0_0/best_model0.733.pth',
#     'Transformer':'./NewModel3_17_output1_Regression/Transformer/mean_0_0_0/0.676.pth',
#     'myAttention': './NewModel3_17_output1_Regression/myAttention/mean_1_0_0/0.671.pth',

#     'RCNN': './NewModel3_17_output1_Regression/RCNN/mean_0_0_0/0.744.pth'

# }
model_list = {
    # 
    'CNN':'./NewModel3_21_output1_Regression/CNN/mean_0_changeTH_0_0/_AMP0.591_total0.63.pth',
    'Transformer':'./NewModel3_21_output1_Regression/Transformer/mean_0_changeTH_0_0/_AMP0.783_total0.601.pth',
    'myAttention': './NewModel3_21_output1_Regression/myAttention/mean_0_changeTH_0_0/_AMP0.629_total0.543.pth',
    'RCNN': './NewModel3_21_output1_Regression/RCNN/mean_0_changeTH_0_0/_AMP0.689_total0.615.pth'

}



# model_list = {
#     'CNN':,
#     'Transformer':'/home/user2/pj/AMP_2/attention_model/NewModel6_1_output1_Regression_without_pretraining/Transformer/mean_0_changeTH_0_0/_AMP0.722_total0.794.pth',
#     'Attention':'/home/user2/pj/AMP_2/attention_model/NewModel6_1_output1_Regression_without_pretraining/myAttention/mean_0_changeTH_0_0/_AMP0.726_total0.788.pth',
#     'RCNN':'/home/user2/pj/AMP_2/attention_model/NewModel6_1_output1_Regression_without_pretraining/RCNN/mean_0_changeTH_0_0/_AMP0.841_total0.811.pth'



# }

testSeq = 'GQIVDNAKEKLGDVASGVAD'
# testModelNamels = ['CNN','Transformer','myAttention','RCNN']
# for testModelName in testModelNamels:
result_soli = []
# testModelName = 'CNN'
trainPath = r'./myRegressionData/all_balance/mean/train.csv'
validatePath = r'./myRegressionData/all_balance/mean/test.csv'

# testPath = '/home/user2/pj/AMP_2/attention_model/test_motif/'+myseq+'/amino_acid_sequences.csv'
# testPath = '/home/user2/pj/AMP_2/attention_model/test_motif_pipeline/'+peptide+'/'+str(alpha)+'/peptides.csv'
# testPath = '/home/user2/pj/AMP_2/attention_model/test_peptide/test_result.csv'
# testPath = '/home/user2/pj/AMP_2/attention_model/6_28_10_ori.csv'
# testPath = r'./12_07Data/test_before_knocking.csv'
testPath = trainPath
# 换一个测试数据集
# testPath = r'./k_mer_selection/k_mer/expandResultTotal.csv'
# testPath = r'./k_mer_selection/k_mer/first_ori_data(L&P)/expandResult_0_50.csv'
# testPath = r'./12_07Data/subtest.csv'
# testPath = r'2_10_testseq.csv'
# testPath = 'AM2/myAttempt/orfClaOptResultIntegation.csv'
# testPath = 'k_mer_selection/k_mer/orf_3_28_endMatch/expandResult_l50.csv'
# testPath = 'NC_vae_result.csv'
# testPath = 'k_mer_selection/k_mer/matchResult_3_29/3_29_kmer_expandResult_2.csv'
# testPath = 'NC_vae/NC_vae_result2.csv'
# testPath = './last_opt.csv'
# testPath = './myRegressionData/all_balance/mean/testAMP.csv'
# testPath = './4_8slectedPeptide/selected_seq2.csv'
# testPath = 'cpExp/HydrAMP/testOnHydrAMP.csv'


# 水溶性预测
df = pd.read_csv(testPath,header=0)
print(df.shape)

seqName = 'Sequence' if 'Sequence' in df.columns else 'sequence'

seqls = df[seqName]

mydict = {'A':0,'C':1,'D':2,'E':3,'F':4,'G':5,'H':6,'I':7,'K':8,'L':9,'M':10,'N':11,'P':12,'Q':13,'R':14,'S':15,'T':16,'V':17,'W':18,'Y':19}


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
        self.seqs = list(self.df[seqName])

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

        return num_seq, mask, label

    def __len__(self):
        return len(self.seqs)


class TestDataset(Dataset):
    def __init__(self,data_path,transform = dataProcessPipeline):
        df = pd.read_csv(data_path,header=0)


        self.df = df

        self.seqs = self.df[seqName]
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



class TrainDataset1(Dataset):
    def __init__(self,data_path,transform = dataProcessPipeline):
        self.df = pd.read_csv(data_path,header=0)
        # self.df = self.df[self.df['Length']<=100]
        # self.df = self.df.sample(n=500)
        self.seqs = self.df['Sequence']
        # self.labels = self.df['label']

        self.transform = transform


    def __getitem__(self,idex):
        seq = self.seqs[idex] 
        num_seq, mask = self.transform(seq) 
        # label = self.labels[idex] 
        # print(num_seq, mask, seq, label, ori, knock)
        return num_seq, mask, seq

    def __len__(self):
        return len(self.seqs)




trainData = TrainDataset(data_path = trainPath)
validateData = TrainDataset(data_path = validatePath)
testData = TestDataset(data_path = testPath)

train_loader = DataLoader(dataset=trainData, batch_size=batch_size, shuffle=True,num_workers=4)
test_loader = DataLoader(dataset=testData, batch_size=batch_size,shuffle=False)
validate_loader = DataLoader(dataset=validateData, batch_size=batch_size,shuffle=False)

testData1 = TestDataset(data_path = testPath)
batch_size = 1024
test_loader1 = DataLoader(dataset=testData1, batch_size=batch_size, shuffle=0,num_workers=4)


frames = []

result_df = pd.read_csv(testPath,header = 0)

model_out = {}
for modelName in nameList:
    print(modelName)
    modelPath = model_list[modelName]
    modelPreName = modelPath.split('/')[-1][:-4]
    id = modelPath.split('/')[-2]
    model_out[modelName] = []


    attmodel = torch.load(modelPath)


    # wired model
    # attmodel.load_state_dict(torch.load('./NewModel11_08_output1/myAttention/1_0/0.954_0.951.pth'),strict = 0 )
    attmodel.cuda()
    attmodel.zero_grad()
    loss_function = nn.MSELoss()

    optimizer = torch.optim.Adam(attmodel.parameters(), lr=lr,  eps=1e-08, weight_decay=0, amsgrad=False)
    softmax = nn.functional.softmax
    sigmoid = torch.sigmoid


    def test_eval(test_loader):
        attmodel.eval()
        total_loss =[]
        for i, data in enumerate(test_loader):
            inputs,masks,labels = data
            inputs = inputs.float()
            masks = masks.float()
            labels = labels.float()
            #inputs,masks,labels = Variable(inputs),Variable(masks),Variable(labels)

            inputs = inputs.cuda()
            masks = masks.cuda()
            # out,_,_,_,_ = attmodel(inputs,masks)
            if modelName != 'myAttention' and modelName!='Transformer2':
                out = attmodel(inputs)
            else:
                out = attmodel(inputs,masks)
            out = torch.squeeze(out)
            # out = sigmoid(out)
            
            out = out.cpu()
            
            # tempt_labels = torch.tensor(labels,dtype=int)
            # tempt_labels = labels.detach.numpy()
            # one_hot_labels = nn.functional.one_hot(tempt_labels,num_classes=2)

            # loss = loss_function(out.float(),tempt_labels.float()) + lambda1*torch.sum((relu(out-tempt_labels))**2)
            loss = loss_function(out,labels) 
            total_loss.append(loss.detach().numpy())

        ave = np.mean(total_loss)
        return ave

    def test_eval_distribu(test_loader):
        attmodel.eval()
        total_diff =[]
        for i, data in enumerate(test_loader):
            inputs,masks,labels = data
            inputs = inputs.float()
            masks = masks.float()
            labels = labels.float()
            #inputs,masks,labels = Variable(inputs),Variable(masks),Variable(labels)

            inputs = inputs.cuda()
            masks = masks.cuda()
            # out,_,_,_,_ = attmodel(inputs,masks)
            if modelName != 'myAttention' and modelName!='Transformer2':
                out = attmodel(inputs)
            else:
                out = attmodel(inputs,masks)
            out = torch.squeeze(out)
            # out = sigmoid(out)
            
            out = out.cpu()
            
            # tempt_labels = torch.tensor(labels,dtype=int)
            # tempt_labels = labels.detach.numpy()
            # one_hot_labels = nn.functional.one_hot(tempt_labels,num_classes=2)

            # loss = loss_function(out.float(),tempt_labels.float()) + lambda1*torch.sum((relu(out-tempt_labels))**2)
            diff = list(out.detach().numpy()-labels.detach().numpy())
            total_diff.append(diff)

        return total_diff


    def test_acc(test_loader):
        attmodel.eval()
        total_acc = []
        flags = []
        for i, data in enumerate(test_loader):
            inputs,masks,labels = data
            #labels01  =[1 for i in labels if i<30 else 0]
            #print(labels01)
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
            
            # out = np.array(torch.argmax(out,dim=1))
            # out = torch.round(out)
            #print(out.shape)
            #print(labels.shape)
            # labels = labels.int()
            #print(out.shape)
            #print(labels.shape)
            #print(out.shape)
            for j in range(out.shape[0]):
                # print(1)
                # flag = (out[j]==labels[j])
                flag = (abs(out[j]-labels[j])<0.5)
                flags.append(flag)
            
        
        acc = np.mean(flags)
        
        return acc
            
        
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
            inputs,masks,labels = data
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


    # precise = tp/(tp+fp+epi)

    # return precise

    test_batch_size = 1

    # score(validate_loader)
    loss0 = test_eval(validate_loader)
    print(modelName,':',str(loss0))

    for i, data in enumerate(test_loader1):
        inputs,masks,seqs = data
        inputs = inputs.float()
        masks = masks.float()
        #inputs = 
        # inputs,masks,labels = Variable(inputs),Variable(masks),Variable(labels)
        
        attmodel.eval()
        inputs = inputs.cuda()
        masks = masks.cuda()
        if modelName != 'myAttention' and modelName != 'Transformer2':
            out = attmodel(inputs)
        else:
            out = attmodel(inputs,masks)
        # out = softmax(out,dim=1)
        # out = torch.squeeze(out)
        # out = out.cpu()
        # predict = np.array(torch.argmax(out,dim=1))
        out = out.cpu()
        out_ori = torch.squeeze(out)
        # out_soft= softmax(out)
 
        
        # predict = torch.round(out)
        # predict = torch.squeeze(predict)

        # out_ori_numpy = list(out_ori.detach().numpy())
        # out_ori_numpy = [round(v,3) for v in out_ori_numpy]
        out_numpy = list(out_ori.detach().numpy())
        out_numpy = [round(v,3) for v in out_numpy]
        model_out[modelName] = list(model_out[modelName])+out_numpy
        # items = []

        # for j,pre in enumerate(predict):
        #     tempt = out_ori[j].detach().item()
        #     item = {'Sequence':seqs[j],modelName+' out':round(out_ori[j].item(),3),'ori':1 if oris[j].data.numpy()==1 else 0}
        #     items.append(item)
        
        # result_df = result_df.append(items,ignore_index =1)

    # frames.append(result_df)

# final_df = frames[0]
# for frame in frames[1:]:
    
#     final_df = pd.merge(final_df,frame)
#     print(final_df.shape)

for k,v in model_out.items():
    result_df[k] = v

# result_df['water solibility'] = result_soli

# 先试 NBT模型结果
NBTdict = {'A':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'K':9,'L':10,'M':11,'N':12,'P':13,'Q':14,'R':15,'S':16,'T':17,'V':18,'W':19,'Y':20}

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

mylen = result_df.shape[0]
numslist = []
for id in range(mylen):
    seq = result_df.loc[id][seqName]
    numlist = seq2num(seq)
    if numlist is not None:
        numslist.append(numlist)
numslist = np.array(numslist)
model = load_model('/home/user2/pj/AMP_2/src/c_AMPs-prediction/Models/att.h5', custom_objects={'Attention_layer': Attention_layer})
x = numslist

preds = model.predict(x)
preds = [round(v[0],3) for v in preds]
result_df['NBT att'] = preds

seqls = result_df[seqName]
# for i,seq in enumerate(seqls):
#     mydata['Sequence'] = seq
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

# result_df['water solibility'] = result_soli

result_df['Length'] = [len(v) for v in result_df[seqName]]
result_df = result_df[[seqName,'Length','CNN','Transformer','myAttention','RCNN','NBT att']]

# result_df = result_df[['Sequence','Length','CNN','Transformer','myAttention','RCNN','NBT att']]

print(result_df.shape)
# final_df = pd.concat(frames,axis=1,join = 'inner')
# saveDir = './AM/myAttempt/optimizedSeq3/'+ testSeq + '/'
# savePath =  saveDir+  testModelName +'_result.csv'
# if not os.path.exists(saveDir):
#     os.makedirs(saveDir)
# # result_df = result_df.sort_values(by = ['out_ori'],ascending=False)
# result_df.to_csv(savePath)
# print(result_df.shape)

# savePath =  './2_5_testseqResult1Softmax.csv'
# savePath =  './2_10_subtestseqSoftmax0.csv'
# savePath =  './3_25_subtestseqRegression.csv'

# result_df = result_df.sort_values(by = ['out_ori'],ascending=False)
# result_df.to_csv(savePath,index=0)
print(result_df.shape)

'''
cd /home/peijun/AMP_2/AMP2/attention_model/
conda activate AMP2
python exp12_4_V2.py 

'''

df = result_df
# x = df['CNN']*df['Transformer']*df['RCNN']*df['myAttention']
x = df['CNN']+df['Transformer']+df['RCNN']+df['myAttention']
# df['value'] = [round(math.pow(v,1/4),3) for v in x]
df['value'] = [round(v/4,3) for v in x]

df = df[[seqName,'value','CNN','Transformer','myAttention','RCNN','NBT att']]
# ori_df = pd.read_csv(validatePath)
# df['value2'] = 
# df.to_csv('first_ori_data(L&P)_ensambleResult.csv',index=0)
# df.to_csv('3_25_orfResult_ensamble2.csv',index=0)
# df.to_csv('./test14Seqs/reg_esb_am.csv',index=0)
# df.to_csv('./k_mer_selection/k_mer/matchResult_3_29/3_29_kmer_regResult_2.csv',index=0)
# df.to_csv('NC_vae/NC_vae_result2_reg.csv',index=0)

print(df)
# df.to_csv('4_7_testAMPResult.csv',index=0)
# df.to_csv('/home/user2/pj/AMP_2/attention_model/6_28_10_ori_result.csv',index=0)
# df.to_csv('/home/user2/pj/AMP_2/attention_model/test_motif/'+myseq+'/amino_acid_sequences_reg.csv',index=0)

# savePath = '/home/user2/pj/AMP_2/attention_model/test_motif_pipeline/'+peptide+'/'+str(alpha)+'/peptides_result.csv'
savePath = testPath[:-4]+'_reg_result.csv'
df.to_csv(savePath,index=0)