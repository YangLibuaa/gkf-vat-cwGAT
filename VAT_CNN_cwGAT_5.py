# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 18:40:54 2020

@author: Administrator
"""
import os
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
import random
import quanfea_vat as vat
from sklearn.metrics import confusion_matrix
from sklearn import metrics


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(2)



##定义模型结构

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features  
        self.out_features = out_features
        self.dropout = dropout    # dropout参数
        self.alpha = alpha     # leakyrelu激活的参数
        self.concat = concat
        
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))  
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  # 初始化
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)   # 初始化
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
    
    def forward(self, inp, adj):
        N = inp.size()[1]    
        B = inp.size()[0]         
        h = torch.matmul(inp, self.W)  
        a_input = torch.cat([h.repeat(1, 1, N).view(B, N * N, self.out_features), h.repeat(1, N, 1)], dim=2).view(B, N, N, 2 * self.out_features)  
        
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))
        e = torch.mul(e,adj)    
        zero_vec = -1e12 * torch.ones_like(e)    
        attention = torch.where(adj>0, e, zero_vec) 
        attention = F.softmax(attention, dim=2)   
        attention = F.dropout(attention, self.dropout, training=self.training) 
        h_prime = torch.bmm(attention, h) 
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime 
    
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class Net(nn.Module):
    def __init__(self, dfcnum, out_features, dropout, alpha):
        super(Net, self).__init__() #调用Net父类的init方法
        self.dfcnum = dfcnum
        self.out_features = out_features
        self.dropout = dropout    # dropout参数
        self.alpha = alpha     # leakyrelu激活的参数        
        self.conv1 = nn.Conv2d(1, 1, (5,1),stride=(2, 1))
        self.pool = nn.MaxPool2d((3, 1), stride=(2, 1))
        self.oneGAT = GraphAttentionLayer(33, self.out_features, self.dropout, self.alpha) #定义一层CWGAT 33为输入全局和局部特征的总个数，该参数根据实际特征数决定
        self.fc1 = nn.Linear(self.out_features * self.dfcnum, 32) #全连接层
        self.fc2 = nn.Linear(32, 3)#全连接层

    def forward(self, x, adj, tfeas):
        x = self.pool(F.relu(self.conv1(x)))
        x = torch.squeeze(x,1) 
        x = x.permute(0,2,1) 
        
        xquan = torch.cat((tfeas,x), 2)
        
        """
        非全特征的话，就把xquan换成x
        """
        gat_out = self.oneGAT(xquan, adj)
        x1 = gat_out.view(gat_out.size(0), -1)
        x1 = F.relu(self.fc1(x1))
        x1 = self.fc2(x1)
        return x1, gat_out


if __name__ == '__main__':    
    
    #读取数据
    lamdas=['0.16']
    R1=str(0.6)
    highorder_lamda = 0.22
    for lamda in lamdas:
        #载入网络
        netfile = '高阶网络结果/Net_order0_lamda_'+str(highorder_lamda)+'.mat'
        print(os.path.join(netfile))
        net_datas = scipy.io.loadmat(netfile)
        netdata = net_datas['high_net']
        netdata = netdata.swapaxes(2,0)#将原来第0、2维交换
        netdata = netdata.swapaxes(1,2)#将第1、2维交换
        netdata = netdata + np.repeat(np.eye(netdata.shape[1])[np.newaxis,...], netdata.shape[0], axis=0) #添加自连接
        netdata = abs(netdata) #取绝对值

        #载入mean1
        Featurefile = 'mean1结果/' + 'mean1lamda_' + lamda  + '_' + str(R1) + '.mat'       
        print(os.path.join(Featurefile))
        datas = scipy.io.loadmat(Featurefile)
        mean1 = datas['mean1']
        X = mean1.swapaxes(2,0)#将原来第0、2维交换
        X = X.swapaxes(1,2) 
        X = np.expand_dims(X,axis=1)#增加一个维度
        _, _, n_step, featureNum = X.shape
        print(X.shape)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                for k in range(X.shape[2]):
                    X[i][j][k] = preprocessing.scale(X[i][j][k])
        labels = scipy.io.loadmat('label.mat')
        labels = labels['label']
        labels = np.array(labels, dtype=np.int32)  # 三分类，0 1 2
        
        #加载全局特征
        quanfile = 'samples_TFeature_all/TFea_lamda_' + lamda  + '_' + str(R1) + '.mat'
        print(os.path.join(quanfile))
        quan_feas = scipy.io.loadmat(quanfile)
        Tfeas = quan_feas['Tfeas']
        kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=2)
    
        total_samples = 0
        total_correct = 0
        total_cfm = np.zeros((3,3))
        total_auc = 0
        kflodCount = 0
        for train_idx, test_idx in kf.split(X, labels):
            kflodCount = kflodCount + 1
            X_train = X[train_idx]
            X_test = X[test_idx]
            Y_train = labels[train_idx]
            Y_test = labels[test_idx]
            train_net = netdata[train_idx]     #训练集网络结构
            test_net = netdata[test_idx]       #测试集网络结构
            train_Tfeas = Tfeas[train_idx]     #训练集全特征
            test_Tfeas = Tfeas[test_idx]       #测试集全特征
            
            X_train = torch.Tensor(X_train)
            X_test = torch.Tensor(X_test)
            Y_train = torch.tensor(Y_train)
            Y_test = torch.tensor(Y_test)
            train_net = torch.Tensor(train_net)
            test_net = torch.Tensor(test_net)
            train_Tfeas = torch.Tensor(train_Tfeas)
            test_Tfeas = torch.Tensor(test_Tfeas)
        
        
            Y_train = torch.squeeze(Y_train)#压缩维度
            Y_test = torch.squeeze(Y_test)#压缩维度
            Y_train = Y_train.long()
            Y_test = Y_test.long()
            
            
            model = Net(dfcnum = X_train.size(3), out_features = 5, dropout = 0.6, alpha = 0.2)#确认有多少个dfc时间序列
            print(model)
            ##针对不平衡样本，对数目更少的类别给出更大的误差惩罚权重（按样本数量反比设置）
            weight_CE=torch.FloatTensor([1.2656,1,1.8455])
            if(torch.cuda.is_available()):
                weight_CE=weight_CE.cuda()
            criterion = nn.CrossEntropyLoss(weight=weight_CE)
            optimizer = optim.Adam(model.parameters(), lr=0.01,weight_decay=0.01)# weight_decay=0.1
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.99)#每2步lr变为0.99倍

            if(torch.cuda.is_available()): #判断是否可用GPU
                model = model.cuda()
                X_train, Y_train, X_test, Y_test = X_train.cuda(), Y_train.cuda(), X_test.cuda(), Y_test.cuda()
                train_net, test_net = train_net.cuda(), test_net.cuda()
                train_Tfeas, test_Tfeas = train_Tfeas.cuda(), test_Tfeas.cuda()
            
            ##训练集训练
            for epoch in range(200):
                
                model.train()#有Batch Normalization或DropOut时 要加model.train()和model.eval()
                
                # zero the parameter gradients
                optimizer.zero_grad()
                
                # forward + backward + optimize
                outputs, train_fea = model(X_train, train_net,train_Tfeas)
                loss = criterion(outputs, Y_train)             
                
                #在对抗训练中，标签也用于产生对抗性扰动。产生扰动使得分类器的预测标签y'变得与实际标签y不同。
                #在虚拟对抗训练中，不使用标签信息，仅使用模型输出生成扰动。产生扰动使得扰动输入的输出不同于原始输入的模型输出（与地面实况标签相反）。
                vat_loss = vat.VATLoss(xi=10.0, eps=1.0, ip=1)
                lds = vat_loss(model, X_train, train_net, train_Tfeas)
                loss = loss + 0.6 * lds
                
                if(torch.cuda.is_available()): #判断是否可用GPU
                    loss = loss.cuda()
                ##
                loss.backward()
                optimizer.step()
                
                running_loss = loss.item()
                _, train_predicted = torch.max(outputs, 1)
                train_correct = (train_predicted == Y_train).sum().item()#计算对了几个
                
                lr_scheduler.step()
                
                if epoch % 50 == 49:
                    ##
                    with torch.no_grad():
                        model.eval()
                        test_outputs, _ = model(X_test, test_net, test_Tfeas)
                        test_loss = criterion(test_outputs, Y_test)
                        running_test_loss=test_loss.item()
                     
                        train_samples = Y_train.size(0)
                     
                        test_samples = Y_test.size(0)
                        _, test_predicted = torch.max(test_outputs, 1)
                        test_correct = (test_predicted == Y_test).sum().item() #计算对了几个
                    print('[%d] trainloss: %.5f testloss: %.5f train_accuracy: %.5f %% test_accuracy: %.5f %%'%
                          (epoch + 1, running_loss, running_test_loss, (100 * train_correct / train_samples), (100 * test_correct / test_samples)))                                  
            print('Finished Training')
            
            ##测试集测试 
            with torch.no_grad():
                model.eval()
                test_outputs, test_fea = model(X_test, test_net, test_Tfeas)
                _, predicted = torch.max(test_outputs, 1) #找出第一维的最大值，predicted里面保存的是下标
                
                test_samples = Y_test.size(0)
                total_samples = total_samples + test_samples
                correct = (predicted == Y_test).sum().item() #计算对了几个
                total_correct = total_correct + correct
                print('Accuracy: %.5f %%' % (100 * correct / test_samples))
                
                if(torch.cuda.is_available()): #判断是否可用GPU
                    Y_test, test_outputs, train_fea, test_fea  = Y_test.cpu(), test_outputs.cpu(), train_fea.cpu(), test_fea.cpu()
                cfm = confusion_matrix(Y_test, test_outputs.argmax(axis=1)) #混淆矩阵
                print(cfm)
                total_cfm = total_cfm + cfm
                
                auc = metrics.roc_auc_score(torch.nn.functional.one_hot(Y_test, 3), test_outputs, average="weighted")
                total_auc = total_auc + auc
                print('AUC: %.5f' % (auc))
                train_fea1 = train_fea
                train_fea1 = train_fea1.detach().numpy()
        print('Total Accuracy: %.5f %%' % (100 * total_correct / total_samples))
        print('Total AUC: %.5f' % (total_auc / 10))
        print(total_cfm)
        scipy.io.savemat('threeclass/lamda'+lamda+ '_' + str(R1) + '_' + str(highorder_lamda) +'acc.mat', mdict={'Accuracy': 100*total_correct/total_samples,'AUC': total_auc/10,'cfm': total_cfm})
