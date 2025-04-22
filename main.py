import numpy as np
import random
import os
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW

max_length = 100
n_epochs = 200
embedding_dim = 75
windows_size = [2, 4, 3]
feature_size = 50
n_class = 4
learn_rate = 3e-4
dropout = 0.1

def process_data(train,test):
    train_data=[]
    test_data=[]
    train_label=[]
    test_label=[]
    for i in range(len(train)):
        lis=[]
        li= []
        for j in range(len(train[i][0])):
            li.append(char2id[train[i][0][j]])
        for k in range(max_length-len(train[i][0])):
            li.append(char2id['<pad>'])
        lis.append(li)
        data= torch.LongTensor(lis)
        train_data.append(data)
        label_li=[]
        label_li.append(labeldict[train[i][1]])
        label_data= torch.Tensor(label_li)
        train_label.append(label_data)
    for i in range(len(test)):
        lis=[]
        li= []
        for j in range(len(test[i][0])):
            li.append(char2id[test[i][0][j]])
        for k in range(max_length-len(test[i][0])):
            li.append(char2id['<pad>'])
        lis.append(li)
        data= torch.LongTensor(lis)
        test_data.append(data)
        label_li=[]
        label_li.append(labeldict[test[i][1]])
        label_data= torch.Tensor(label_li)
        test_label.append(label_data)
    return train_data,test_data,train_label,test_label

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, windows_size, max_len, feature_size, n_class, dropout=0.2):
        super(TextCNN, self).__init__()
        # embedding层
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        # 卷积层特征提取
        self.conv1 = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=embedding_dim, out_channels=feature_size, kernel_size=h),
                          nn.LeakyReLU(),
                          nn.MaxPool1d(kernel_size=max_len-h+1),
                          )
            for h in windows_size]
        )
        # 全连接层
        self.fc = nn.Linear(feature_size*len(windows_size), n_class)
        # dropout防止过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embed(x) # [batch, seq_len, embed_dim]
        x = x.permute(0, 2, 1) # [batch, embed_dim, seq_len]
        x = [conv(x) for conv in self.conv1]
        x = torch.cat(x, 1)
        x = x.view(-1, x.size(1)) # [batch, feature_size*len(windows_size)]
        x = self.dropout(x)
        x = self.fc(x)# [batch, n_class]
        return x






if __name__=='__main__':
    with open("./train/All_train",'r') as f:
        train_data = f.readlines()
    with open("./test/All_test",'r') as f:
        test_data = f.readlines()
    print(len(train_data))
    print(len(test_data))
    train=[]
    test=[]
    for i in range(len(train_data)):
        lis=[]
        lis.append(train_data[i].split('\t\t')[0])
        lis.append(train_data[i].split('\t\t')[1].strip())
        train.append(lis)
    for i in range(len(test_data)):
        lis=[]
        lis.append(test_data[i].split('\t\t')[0])
        lis.append(test_data[i].split('\t\t')[1].strip())
        test.append(lis)
    print(len(train))
    print(len(test))
    char_set=set()
    label_set=set()
    max_length=0
    for i in range(len(train)):
        if max_length<len(train[i][0]):
            max_length = len(train[i][0])
        for j in range(len(train[i][0])):
            char_set.add(train[i][0][j])
        label_set.add(train[i][1])
    for i in range(len(test)):
        if max_length<len(test[i][0]):
            max_length = len(test[i][0])
        for j in range(len(test[i][0])):
            char_set.add(test[i][0][j])
        label_set.add(test[i][1])
    print(char_set)
    print(label_set)
    print(max_length)
    char2id={}
    id2char={}
    labeldict={'ADVISE':[1,0,0,0],'INT':[0,1,0,0],'EFFECT':[0,0,1,0],'MECHANISM':[0,0,0,1]}
    i=0
    for item in char_set:
        id2char[i]=item
        char2id[item]=i
        i+=1
    char2id['<pad>']=len(char2id)
    id2char[74]='<pad>'
    print(char2id)
    print(id2char)
    char_embed=np.zeros((len(char2id),len(char2id)))
    for item in char2id:
        char_embed[char2id[item]][char2id[item]]=1
    print(char_embed)
    print(char_embed.shape)
    device = "cpu"
    model = TextCNN(75, embedding_dim, windows_size, max_length, feature_size, n_class, dropout)
    print(model.embed.weight.data)
    for i in range(len(char_embed)):
        for j in range(len(char_embed[0])):
            model.embed.weight.data[i][j]=char_embed[i][j]
    print(model.embed.weight.data)
    optimizer = Adam(model.parameters(), lr=learn_rate)
    criterion = nn.CrossEntropyLoss()
    # print(train[0])
    train_data,test_data,train_label,test_label = process_data(train,test)
    # for epoch in range(n_epochs):
    #     model.train()
    #     for i in range(len(train_data)):
    #         output = model(train_data[i])
    #         real_label = train_label[i]
    #         loss = criterion(output,real_label) 
    #         # print(output)
    #         # print(loss)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         print('epoch: {}, i: {}, loss: {}'.format(epoch, i + 1, loss))
    # model.save()
    # torch.save(model, 'textcnn.pth')
    model = torch.load('textcnn.pth')
    model.eval()
    accurate=0
    with torch.no_grad():
        for i in range(len(test_data)):
            res = model(test_data[i])
            _, index = torch.max(res, 1)
            # print(index)
            res_one_hot = torch.zeros(1,4)
            res_one_hot[0][index]=1
            # print(res_one_hot[0])
            # print(test_label[i][0])
            compare_tensor=torch.eq(res_one_hot[0], test_label[i][0])
            # print(compare_tensor)
            if compare_tensor.all():
                accurate+=1
    #     # print(res)
    #     # break
    print(accurate)
            # break
    # print(len(train_data))
    # print(len(train_label))
    # print(len(test_data))
    print(len(test_label))
    accurate = accurate/len(test_label)
    print("accurate:",accurate)
    # print(train_data[0])
    # print(train_label[0])
        
    
        
    
    
    
