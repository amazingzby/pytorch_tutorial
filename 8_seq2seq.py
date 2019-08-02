import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import os
import unicodedata
import numpy as np

device = torch.device("cpu")
MAX_LENGTH = 10 #Maximum sentence length

PAD_token = 0
SOS_token = 1
EOS_token = 2

class Voc:
    def __init__(self,name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token:"PAD",SOS_token:"SOS",EOS_token:"EOS"}
        self.num_words = 3
    def addSentence(self,sentence):
        for word in sentence.split(' '):
            self.addWord(word)
    def addWord(self,word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def trim(self,min_count):
        if self.trimmed:
            return
        self.trimmed = True
        keep_words = []
        for k,v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(len(keep_words),
                len(self.word2index),len(keep_words)/len(self.word2index)))

        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3
        for word in keep_words:
            self.addWord(word)

def normalizeString(s):
    s = s.lower()
    #re 正则表达式 sub 用于替换 \1是匹配第一个分组匹配到的内容，也就是所谓的\1引用了第一个()匹配到的内容
    s = re.sub(r"([.!?])",r" \1",s)
    s = re.sub(r"[^a-zA-Z.!?]+"," ",s)
    return s

def indexsFromSentence(voc,sentence):
    return [voc.word2index[word] for word in sentence.split(' ')]+[EOS_token]

class EncoderRNN(nn.Module):
    def __init__(self,hidden_size,embedding,n_layers=1,dropout = 0):
        super(EncoderRNN,self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding
        #torch.nn.GRU(*args, **kwargs)
        #input_size 输入x的features数量
        #hidden_size 隐层 features数量
        #num_layers rnn层数
        #bias 是否使用bias
        #batch_first 是否batch为第一个维度
        #dropout 是否使用dropout
        self.gru = nn.GRU(hidden_size,hidden_size,n_layers,dropout=(0 if n_layers==1 else dropout),bidirectional=True)

    def forward(self,input_seq,input_lengths,hidden=None):
        embedded = self.embedding(input_seq)
        #pack_padded_sequence(input, lengths, batch_first=False, enforce_sorted=True):
        #输入为T*B*X(T为最长序列，B为batchsize，X为维度，如果batch_first为True，为B*T*X)
        #作用为将input_seq pad
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded,input_lengths)
        outputs,hidden = self.gru(packed,hidden)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs[:,:,self.hidden_size] + outputs[:,:,self.hidden_size]
        return outputs,hidden

class Attn(nn.Module):
    def __init__(self,method,hidden_size):
        super(Attn,self).__init__()
        self.method = method
        if self.method not in ['dot','general','concat']:
            raise ValueError(self.method,"is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = torch.nn.Linear(self.hidden_size,hidden_size)
        elif self.method == 'concat':
            self.attn = torch.nn.Linear(self.hidden_size*2,hidden_size)
            self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self,hidden,encoder_output):
        return torch.sum(hidden*encoder_output,dim=2)
    def general_score(self,hidden,encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden*energy,dim=2)

    def concat_score(self,hidden,encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0),-1,-1),encoder_output),2)).tanh()
        return torch.sum(self.v * energy,dim=2)

    def forward(self, hidden,encoder_outputs):
        if self.method == 'general':
            attn_energies = self.general_score(hidden,encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden,encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden,encoder_outputs)
        attn_energies = attn_energies.t()
        return F.softmax(attn_energies,dim=1).unsqueeze(1)



































