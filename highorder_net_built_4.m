function [msg]=highorder_net_built_4()
mkdir('�߽�������');
load('mean1���/mean1lamda_0.16_0.6.mat');
for lamda=0.22
    [high_net] = lasso(mean1,lamda);
    save(['�߽�������\Net_order0_lamda_',num2str(lamda),'.mat'],'high_net');
end
msg = 'done';