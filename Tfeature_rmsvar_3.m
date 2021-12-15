mkdir('samples_TFeature_all');
mkdir('mean1结果');
pd = pwd;
cd(pd)
for lamda = [0.16]%[0.01:0.01:0.3]
    for kf = (0.6)%[0.1:0.1:0.9]
        load([pd,'\cts1结果\cts1_lamda_',num2str(lamda),'_',num2str(kf),'.mat']);
        load([pd,'\cts结果\cts_lamda_',num2str(lamda),'_',num2str(kf),'.mat']);
		[x,y,z]=size(cts);
        k=0;
        for i=1:y
            if length(find(cts1(:,i)==0))<(x*z/2) %(find(cts1(:,i)<0.01))<(x*z/3)
                k=k+1;
            end
        end
        mean1=zeros(x,k,z);
        k=0;
        for i=1:y
            if length(find(cts1(:,i)==0))<(x*z/2) %(find(cts1(:,i)<0.01))<(x*z/3)
                k=k+1;
                mean1(:,k,:)=cts(:,i,:);
            end
        end
        save([pd,'\mean1结果\mean1lamda_',num2str(lamda),'_',num2str(kf),'.mat'],'mean1');
        Tfeas=zeros(z,k,2);
        for i=1:z
            for j=1:k
                Tfeas(i,j,1)=std(mean1(:,j,i),1);
                Tfeas(i,j,2) = rms(mean1(:,j,i));
            end
        end
        save([pd,'\samples_TFeature_all\TFea_lamda_',num2str(lamda),'_',num2str(kf),'.mat'],'Tfeas');
    end
end
