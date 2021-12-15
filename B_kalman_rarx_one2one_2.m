clc;
clear;
warning('off');
load('fmri.mat') %fmri����ά���� �������£�
[Z,M,N]=size(fmri); %ZΪʱ�����г��� MΪ�������� NΪ������
fmri=permute(fmri,[3,1,2]);%����ά��
mkdir('kal������');
mkdir('cts���');
mkdir('cts1���');
for lamda = 0.16 %�ò����ɵ���[0.01:0.01:0.3]
    load(['����ṹ���\low_net_lamda_',num2str(lamda),'.mat'],'low_net');
    for kf = 0.6 %�ò����ɵ���[0.1:0.1:0.9]
        kalnet=zeros(M,M,Z-1,N);
        for k=1:N
            relation = zeros(M,M,Z-1);
            temp =  squeeze(fmri(k,:,1:M));
            for i=1:M
                for j=1:M
                    if abs(low_net(i,j,k))<0.001
%                         disp('Ϊ������������������ϵ');
                    else
                        % ���� ��һ������Z��ʱ���
                        input = temp(:,j);
                        % ��� ��һ������Z��ʱ���
                        output = temp(:,i);
                        z = [output input];
                        nn = [0 1 1];%�ò����ɵ���
                        [thm,yhat,P,phi] = rarx(z,nn,'kf',kf);
                        relation(i,j,:)=thm(2:Z);
                    end
                end
            end
            kalnet(:,:,:,k)=relation;
        end
        disp(kf);
        save(['kal������\kalnet_lamda_',num2str(lamda),'_',num2str(kf),'.mat'],'kalnet');
		%Ϊ��̬ʱ������������ȡ��߽����繹����׼��
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
        save(['cts���\cts_lamda_',num2str(lamda),'_',num2str(kf),'.mat'],'cts');
        [xx,y,z]=size(cts);
        for j=1:z
            cts1((xx*j-xx+1):(xx*j),:)=cts(:,:,j);
        end
        save(['cts1���\cts1_lamda_',num2str(lamda),'_',num2str(kf),'.mat'],'cts1'); 
    end
    disp(num2str(lamda));
end


