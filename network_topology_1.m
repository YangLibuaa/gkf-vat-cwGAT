load('fmri.mat') %fmri����ά���� �������£�
[Z,M,N]=size(fmri); %ZΪʱ�����г��� MΪ�������� NΪ������
mkdir('����ṹ���');
for lamda=0.16 %�ò������Ե���[0.01:0.01:0.3]
    [low_net] = net_built_grouplasso(fmri,lamda);
    save(['����ṹ���\low_net_lamda_',num2str(lamda),'.mat'],'low_net');
end