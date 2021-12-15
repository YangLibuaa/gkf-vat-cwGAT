function [msg]=highorder_net_built_4()
mkdir('高阶网络结果');
load('mean1结果/mean1lamda_0.16_0.6.mat');
for lamda=0.22
    [high_net] = lasso(mean1,lamda);
    save(['高阶网络结果\Net_order0_lamda_',num2str(lamda),'.mat'],'high_net');
end
msg = 'done';