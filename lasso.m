function   [net] = lasso(mean1,lambda)
ROIP=size(mean1,2);
Net_fmri = zeros(ROIP,ROIP,size(mean1,3));
opts = [];
opts.init = 1;
opts.tFlag = 5;
opts.maxIter = 1000;
opts.nFlag = 0;
opts.rFlag = 1;

for i = 1:size(mean1,3)  % �� i ������
    disp(['  ��',num2str(i),'������']);
    fmri = mean1(:,:,i);
    [Msize, Nsize] = size(fmri);
    for j = 1:Nsize  % �� j �� ROI   
        A = fmri;
        %A = MatrixScale(fmri);
        A(:,j) = zeros(Msize,1);
        if sum(abs(fmri(:,j)))>0
            [Beta, ~, ~] = LeastR(A,fmri(:,j),lambda,opts);
            %[Beta, ~, ~] = LeastR(A,MatrixScale(fmri(:,j)),lambda,opts);
            index = find(abs(Beta)>1e-6);
            if any(index==j)
                msgbox('���ִ���ѡ���������ӣ���������','error','error');
                msg = 'Error happens !!';
                return;
            end
            Net_fmri(:,j,i) = Beta;
        end
    end
end
net=Net_fmri;

function Mscaled = MatrixScale(MOrignal)
N = size(MOrignal,2);
Mscaled = zeros(size(MOrignal));
for i = 1:N
    temp = norm(MOrignal(:,i),2);
    if temp<eps
        continue;
    end
    Mscaled(:,i) = MOrignal(:,i)./temp;
end


