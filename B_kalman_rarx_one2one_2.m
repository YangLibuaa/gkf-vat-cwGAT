clc;
clear;
warning('off');
load('fmri.mat') %fmri是三维矩阵 具体如下：
[Z,M,N]=size(fmri); %Z为时间序列长度 M为脑区个数 N为样本数
fmri=permute(fmri,[3,1,2]);%交换维度
mkdir('kal网络结果');
mkdir('cts结果');
mkdir('cts1结果');
for lamda = 0.16 %该参数可调节[0.01:0.01:0.3]
    load(['网络结构结果\low_net_lamda_',num2str(lamda),'.mat'],'low_net');
    for kf = 0.6 %该参数可调节[0.1:0.1:0.9]
        kalnet=zeros(M,M,Z-1,N);
        for k=1:N
            relation = zeros(M,M,Z-1);
            temp =  squeeze(fmri(k,:,1:M));
            for i=1:M
                for j=1:M
                    if abs(low_net(i,j,k))<0.001
%                         disp('为零脑区，脑区间无联系');
                    else
                        % 输入 第一个脑区Z个时间点
                        input = temp(:,j);
                        % 输出 另一个脑区Z个时间点
                        output = temp(:,i);
                        z = [output input];
                        nn = [0 1 1];%该参数可调节
                        [thm,yhat,P,phi] = rarx(z,nn,'kf',kf);
                        relation(i,j,:)=thm(2:Z);
                    end
                end
            end
            kalnet(:,:,:,k)=relation;
        end
        disp(kf);
        save(['kal网络结果\kalnet_lamda_',num2str(lamda),'_',num2str(kf),'.mat'],'kalnet');
		%为动态时间序列特征提取与高阶网络构建做准备
        cts=zeros(Z-1,M*M-M,N);
        k=0;
        for i=1:M
            for j=setdiff(1:M,i)
                k=k+1;
                for z=1:N
                    cts(:,k,z)=squeeze(kalnet(i,j,:,z));
                end
            end
        end
        save(['cts结果\cts_lamda_',num2str(lamda),'_',num2str(kf),'.mat'],'cts');
        [xx,y,z]=size(cts);
        for j=1:z
            cts1((xx*j-xx+1):(xx*j),:)=cts(:,:,j);
        end
        save(['cts1结果\cts1_lamda_',num2str(lamda),'_',num2str(kf),'.mat'],'cts1'); 
    end
    disp(num2str(lamda));
end


