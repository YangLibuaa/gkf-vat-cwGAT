function  [net] = net_built_grouplasso(P,lamda)
[m,n,t]=size(P);
tmp=P(1:m-1,:,:);
P = cat(2,P(2:m,:,:),tmp);
net=zeros(n,2*n,t);
Ps = [];
for i = 1:t
    Ps = cat(1,Ps,P(:,:,i));
end
for j=1:n
    opts = [];
    opts.q = 2;
    opts.init = 2;
    opts.tFlag = 5;
    opts.maxIter = 1000;
    opts.nFlag = 0;
    opts.rFlag = 1;
    opts.ind = [];
    [MM, ~] = size(Ps);
    Ys = Ps(:,j);
    PP=Ps;
    PP(:,j) = zeros(MM,1);  
    opts.ind =0:MM/t:MM;
    [x1, ~, ~] = mtLeastR(PP,Ys,lamda,opts);
    net(j,:,:)=x1;
    disp(['建网络已完成',num2str(j/n*100),'%']);
end
net1 = net;
net =net1(:,1:n,:)+net1(:,n+1:2*n,:);
end