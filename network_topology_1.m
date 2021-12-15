load('fmri.mat') %fmri是三维矩阵 具体如下：
[Z,M,N]=size(fmri); %Z为时间序列长度 M为脑区个数 N为样本数
mkdir('网络结构结果');
for lamda=0.16 %该参数可以调节[0.01:0.01:0.3]
    [low_net] = net_built_grouplasso(fmri,lamda);
    save(['网络结构结果\low_net_lamda_',num2str(lamda),'.mat'],'low_net');
end