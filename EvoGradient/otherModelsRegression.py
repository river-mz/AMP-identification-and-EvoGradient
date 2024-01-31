# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

'''cnn'''

class CNN(nn.Module):
    def __init__(self,batch_size=128,embedding_size=20,num_tokens=100,num_filters = 100,filter_sizes = (2,3,4),num_classes=1,num_heads=4):
        super(CNN, self).__init__()
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.num_tokens = num_tokens
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters

        self.hidden1 = 20
        self.hidden2 = 60
        self.hidden3 = 20
        self.dropout = 0.3
        self.fc1 = nn.Linear(self.embedding_size,self.hidden1)
        self.relu = nn.ReLU()
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, self.num_filters, (k, self.hidden1)) for k in self.filter_sizes])
        self.dropout = nn.Dropout(self.dropout)
        self.fc2 = nn.Linear(self.num_filters, self.hidden2)
        self.new_fc3 = nn.Linear(self.hidden2, self.num_classes)
        self.softmax = nn.functional.softmax
        self.fc = nn.Linear(98*3,1)
        
        
        


    def conv_and_pool(self, x, conv):
        # x: [128 1 32 300]
        #print(x.shape) #[128, 1, 100, 20])
        x = F.relu(conv(x)).squeeze(3)  # [128 100 [31 30 29]=90 ]
        # x: [128 256 31]
        # print(x.shape)
        
        # fc = nn.Linear(x.size(2),1)
        # 利用线性FCN替代maxpooling
        #x = F.max_pool1d(x, x.size(2)).squeeze(2)
        
        # x = fc(x).squeeze(2)
        
        # [N,100]
        return x

    def forward(self, x):
        # tuple 2 [128,32]
        # [N,100,26]
        out = self.fc1(x)
        #  out: [128 32 300]
        #[N,100,20]
        out = out.unsqueeze(1)
        # out: [128 1 32 300]   32 2 3 4     31 30 29=
        # [N, 1, 100, 20]   
        # print(out.shape)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 2)
        # print(out.shape)
        out = self.fc(out).squeeze(2)

        # out: [128 768]
        # [N, 300]
        out = self.dropout(out)
        out = self.relu(out)
        # out: [128 100]
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.new_fc3(out)
        # out:[128 10]
        return out



class CNN(nn.Module):
    def __init__(self,batch_size=128,embedding_size=20,num_tokens=100,num_filters = 100,filter_sizes = (2,3,4),num_classes=1,num_heads=4):
        super(CNN, self).__init__()
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.num_tokens = num_tokens
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters

        self.hidden1 = 20
        self.hidden2 = 60
        self.hidden3 = 20
        self.dropout = 0.3
        self.fc1 = nn.Linear(self.embedding_size,self.hidden1)
        self.relu = nn.ReLU()
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, self.num_filters, (k, self.hidden1)) for k in self.filter_sizes])
        self.dropout = nn.Dropout(self.dropout)
        self.fc2 = nn.Linear(self.num_filters, self.hidden2)
        self.new_fc3 = nn.Linear(self.hidden2, self.num_classes)
        self.softmax = nn.functional.softmax
        self.fc = nn.Linear(98*3,1)
        
        
        


    def conv_and_pool(self, x, conv): # [128,1,100,20]
        # x: [128 1 32 300]
        #print(x.shape) #[128, 1, 100, 20])
        x = F.relu(conv(x)).squeeze(3)  # [128 100 [31 30 29]=90 ]
        # x: [128 256 31]
        # print(x.shape)
        
        # fc = nn.Linear(x.size(2),1)
        # 利用线性FCN替代maxpooling
        #x = F.max_pool1d(x, x.size(2)).squeeze(2)
        
        # x = fc(x).squeeze(2)
        
        # [N,100]
        return x

    def forward(self, x):
        # tuple 2 [128,32]
        # [N,100,26]
        out = self.fc1(x) # in[128,100,20]  out[128,100,20] 
        #  out: [128 32 300]
        #[N,100,20]
        out = out.unsqueeze(1) #in[128,100,20] out[128,1,100,20] 
        # out: [128 1 32 300]   32 2 3 4     31 30 29=
        # [N, 1, 100, 20]   
        # print(out.shape)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 2)
        # print(out.shape)
        out = self.fc(out).squeeze(2)

        # out: [128 768]
        # [N, 300]
        out = self.dropout(out)
        out = self.relu(out)
        # out: [128 100]
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.new_fc3(out)
        # out:[128 10]
        return out






'''Recurrent Convolutional Neural Networks for Text Classification'''

''' lstm '''


class RCNN(nn.Module):
    def __init__(self, batch_size=128,embedding_size=20,num_tokens=100,num_filters = 100,filter_sizes = (2,3,4),num_classes=1,num_heads=4):
        super(RCNN, self).__init__()

        # if config.embedding_pretrained is not None:
        #     self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        # else:
        #     self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        

        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.num_tokens = num_tokens
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters

        self.hidden1 = 20
        self.hidden2 = 60
        self.hidden3 = 20
        self.dropout_rate = 0.3
        self.num_layers = 1
        self.pad_size = num_tokens

        self.fc1 = nn.Linear(self.embedding_size,self.hidden1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout_rate)
        # self.fc2 = nn.Linear(self.num_filters * len(self.filter_sizes), self.hidden2)
        # self.fc3 = nn.Linear(self.hidden2, self.num_classes)
        

        self.fc1 = nn.Linear(self.embedding_size,self.hidden1)
        self.lstm = nn.LSTM(self.hidden1, self.hidden2, self.num_layers,
                            bidirectional=True, batch_first=True, dropout=self.dropout_rate)
        self.maxpool = nn.MaxPool1d(self.pad_size)
        self.fc4 = nn.Linear(self.pad_size,1)
        self.new_fc = nn.Linear(self.hidden2 * 2 + self.hidden1, self.num_classes)
        self.softmax = torch.softmax

    def forward(self, x):
        # x, _ = x # [128,32]   [N,100,26]  
        embed = self.fc1(x) # [N, 100, 20]
        # embed = self.embedding(x)  # [batch_size, seq_len, embeding]=[64, 32, 64]  # [128 32 300]
        out, _ = self.lstm(embed) # [128 32 300]    [N 100 40]
        out = torch.cat((embed, out), 2)  #[128 32 812]   [N 100 60]
        out = F.relu(out) # [128 32 812]  [N 100 60]
        out = out.permute(0, 2, 1) #[128 812 32]   [N 60 100]
        # out = self.maxpool(out).squeeze() # [128 812]  [N 60]
        out = self.fc4(out).squeeze() # [128 812]  [N 60]
        out = self.dropout(out)
        out = self.new_fc(out) # [128 10]  [N 2]
        return out






'''lstm_attention'''

class lstm_att(nn.Module):
    def __init__(self, batch_size=128,embedding_size=20,num_tokens=100,num_filters = 100,filter_sizes = (2,3,4),num_classes=1,num_heads=4):
        super(lstm_att, self).__init__()

        # if config.embedding_pretrained is not None:
        #     self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        # else:
        #     self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        
        
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.num_tokens = num_tokens
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters

        self.hidden1 = 20
        self.hidden2 = 60
        self.hidden3 = 50
        self.dropout_rate = 0.3
        self.num_layers = 1
        self.pad_size = num_tokens
        
        self.fc1 = nn.Linear(self.embedding_size,self.hidden1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout_rate)
        self.fc2 = nn.Linear(self.hidden2*2, self.hidden2)
        self.new_fc3 = nn.Linear(self.hidden2, self.num_classes)
        

 
        self.lstm = nn.LSTM(self.hidden1, self.hidden2, self.num_layers,
                            bidirectional=True, batch_first=True, dropout=self.dropout_rate)


        self.softmax = torch.softmax
        
   
        self.tanh1 = nn.Tanh()
        # self.u = nn.Parameter(torch.Tensor(config.hidden_size * 2, config.hidden_size * 2))
        self.w = nn.Parameter(torch.zeros(self.hidden2 * 2))
        self.tanh2 = nn.Tanh()


    def forward(self, x): #  2 [128 32]  [128]
        # [N 100 26]
        embed = self.fc1(x)  # [batch_size, seq_len, embeding]=[128, 32, 300]  [N 100 20]
        H, _ = self.lstm(embed)  # [batch_size, seq_len, hidden_size * num_direction]=[128, 32, 256]    [N 100 120]

        M = self.tanh1(H)  # [128, 32, 256]     [N 100 120] [120]
        # M = torch.tanh(torch.matmul(H, self.u))
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)  # [128, 32, 1]   [N 100 1]
        out = H * alpha  # [128, 32, 256]     [N 100 120] [N 100 1]->[N 100 120]
        out = torch.sum(out, 1)  # [128, 256]    [N 120]
        out = self.dropout(out)
        out = F.relu(out)   
        out = self.fc2(out)   #[128 64]   [N 60]
        out = self.dropout(out)
        out = F.relu(out) 
        out = self.new_fc3(out)  # [128, 10]     [N 2]
        return out




'''transformer'''

# the following code is transformer structure:
class Transformer(nn.Module):
    def __init__(self, batch_size=128,embedding_size=20,num_tokens=100,num_filters = 100,filter_sizes = (2,3,4),num_classes=1,num_heads=4):
        super(Transformer, self).__init__()
        # if config.embedding_pretrained is not None:
        #     self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        # else:
        #     self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)

        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.num_tokens = num_tokens
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.pad_size = num_tokens
        self.num_encoder = 2


        self.hidden1 = 20
        self.hidden2 = 128
        self.hidden3 = 20
        self.dropout_rate = 0.3
        self.num_layers = 1
        self.pad_size = num_tokens
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.postion_embedding = Positional_Encoding(self.hidden1, self.pad_size, self.dropout_rate, self.device)
        self.encoder = Encoder(self.hidden1, self.num_heads, self.hidden2, self.dropout_rate)
        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            # Encoder(config.dim_model, config.num_head, config.hidden, config.dropout)
            for _ in range(self.num_encoder)])

        self.new_fc1 = nn.Linear(self.pad_size * self.hidden1, self.num_classes)
        # self.fc2 = nn.Linear(config.last_hidden, config.num_classes)
        self.fc = nn.Linear(self.embedding_size, self.hidden1)

    def forward(self, x): # [128 32] [128]   [N 100 26]
        out = self.fc(x) #[128 32 300]    [N 100 20]
        out = self.postion_embedding(out) #[128 32 300] [N 100 20]
        for encoder in self.encoders: #[128 32 300] [128]
            out = encoder(out)
        out = out.view(out.size(0), -1) #[128 9600] [N 2000]
        # out = torch.mean(out, 1)
        out = self.new_fc1(out) #[128 10]  [N 2]
        return out


class Encoder(nn.Module):
    def __init__(self, dim_model, num_head, hidden, dropout):
        super(Encoder, self).__init__()
        self.attention = Multi_Head_Attention(dim_model, num_head, dropout)
        self.feed_forward = Position_wise_Feed_Forward(dim_model, hidden, dropout)

    def forward(self, x): #[128 32 300]
        out = self.attention(x) #[128 32 300]
        out = self.feed_forward(out) #[128 32 300]
        return out


class Positional_Encoding(nn.Module):
    def __init__(self, embed, pad_size, dropout, device):
        super(Positional_Encoding, self).__init__()
        self.device = device
        self.pe = torch.tensor([[pos / (10000.0 ** (i // 2 * 2.0 / embed)) for i in range(embed)] for pos in range(pad_size)])
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = x + nn.Parameter(self.pe, requires_grad=False).to(self.device)
        out = self.dropout(out)
        return out


class Scaled_Dot_Product_Attention(nn.Module):
    '''Scaled Dot-Product Attention '''
    def __init__(self):
        super(Scaled_Dot_Product_Attention, self).__init__()

    def forward(self, Q, K, V, scale=None):
        '''
        Args:
            Q: [batch_size, len_Q, dim_Q]
            K: [batch_size, len_K, dim_K]
            V: [batch_size, len_V, dim_V]
            scale: 缩放因子 论文为根号dim_K
        Return:
            self-attention后的张量，以及attention张量
        '''
        attention = torch.matmul(Q, K.permute(0, 2, 1))
        if scale:
            attention = attention * scale
        # if mask:  # TODO change this
        #     attention = attention.masked_fill_(mask == 0, -1e9)
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)
        return context


class Multi_Head_Attention(nn.Module):
    def __init__(self, dim_model, num_head, dropout=0.3):
        super(Multi_Head_Attention, self).__init__()
        self.num_head = num_head
        assert dim_model % num_head == 0
        self.dim_head = dim_model // self.num_head
        self.fc_Q = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_K = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_V = nn.Linear(dim_model, num_head * self.dim_head)
        self.attention = Scaled_Dot_Product_Attention()
        self.fc = nn.Linear(num_head * self.dim_head, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x): #[128 32 300]
        batch_size = x.size(0) 
        Q = self.fc_Q(x) #[128 32 300]
        K = self.fc_K(x) #[128 32 300]
        V = self.fc_V(x) #[128 32 300]
        Q = Q.view(batch_size * self.num_head, -1, self.dim_head) # [640 32 60]
        K = K.view(batch_size * self.num_head, -1, self.dim_head)
        V = V.view(batch_size * self.num_head, -1, self.dim_head)
        # if mask:  # TODO
        #     mask = mask.repeat(self.num_head, 1, 1)  # TODO change this
        scale = K.size(-1) ** -0.5  # 缩放因子
        context = self.attention(Q, K, V, scale)

        context = context.view(batch_size, -1, self.dim_head * self.num_head)
        out = self.fc(context)
        out = self.dropout(out)
        out = out + x  # 残差连接
        # out = self.layer_norm(out)
        return out


class Position_wise_Feed_Forward(nn.Module):
    def __init__(self, dim_model, hidden, dropout=0.0):
        super(Position_wise_Feed_Forward, self).__init__()
        self.fc1 = nn.Linear(dim_model, hidden)
        self.fc2 = nn.Linear(hidden, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x): #[128 32 300]
        out = self.fc1(x) #[128 32 1024]
        out = F.relu(out) 
        out = self.fc2(out) #[128 32 300]
        out = self.dropout(out)
        out = out + x  # 残差连接  [128 32 300]
        # out = self.layer_norm(out) #[128 32 300]
        return out






## transformer2
class Transformer2(nn.Module):
    def __init__(self, batch_size=128,embedding_size=20,num_tokens=100,num_filters = 100,filter_sizes = (2,3,4),num_classes=1,num_heads=4):
        super(Transformer2, self).__init__()

        # if config.embedding_pretrained is not None:
        #     self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        # else:
        #     self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        

        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.num_tokens = num_tokens
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters

        self.hidden1 = 20
        self.hidden2 = 60
        self.hidden3 = 20
        self.dropout_rate = 0.3
        self.num_layers = 1
        self.pad_size = num_tokens

        self.fc1 = nn.Linear(self.embedding_size,self.hidden1)
        self.relu = nn.ReLU()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=20, nhead=4,dim_feedforward=128,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)

        self.dropout = nn.Dropout(self.dropout_rate)
        self.fc2 = nn.Linear(self.num_filters * len(self.filter_sizes), self.hidden2)
        self.fc3 = nn.Linear(self.hidden2, self.num_classes)
        

        self.fc1 = nn.Linear(self.embedding_size,self.hidden1)
        self.lstm = nn.LSTM(self.hidden1, self.hidden2, self.num_layers,
                            bidirectional=True, batch_first=True, dropout=self.dropout_rate)
        self.maxpool = nn.MaxPool1d(self.pad_size)
        self.fc4 = nn.Linear(self.pad_size,1)
        self.fc = nn.Linear(self.hidden2 * 2 + self.hidden1, self.num_classes)
        self.softmax = torch.softmax


        self.fc2 = nn.Linear(self.hidden1,1)
        self.fc3 = nn.Linear(self.num_tokens,self.num_classes)

    def forward(self, x, mask):

        #[N,100,26]
        out = self.fc1(x) #[N 100 20]
        out = self.transformer_encoder(out,src_key_padding_mask = mask) #[N 100 20]
        out = self.fc2(out).squeeze(2)
        out = self.relu(out)
        out = self.fc3(out)
        # print(out.shape)cd 




        # # x, _ = x # [128,32]   [N,100,26]  
        # embed = self.fc1(x) # [N, 100, 20]
        # # embed = self.embedding(x)  # [batch_size, seq_len, embeding]=[64, 32, 64]  # [128 32 300]
        # out, _ = self.lstm(embed) # [128 32 300]    [N 100 40]
        # out = torch.cat((embed, out), 2)  #[128 32 812]   [N 100 60]
        # out = F.relu(out) # [128 32 812]  [N 100 60]
        # out = out.permute(0, 2, 1) #[128 812 32]   [N 60 100]
        # # out = self.maxpool(out).squeeze() # [128 812]  [N 60]
        # out = self.fc4(out).squeeze() # [128 812]  [N 60]
        # out = self.fc(out) # [128 10]  [N 2]
        # out = self.softmax(out,dim = -1)


        return out
