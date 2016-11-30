function [Trained_outputs,Test_outputs]=Learn_by_TimeSeries_Dists(TrainData,TestData,Params)

% Params.TimeSeries.Type='LCSS' ; % DTW  or  LCSS
% Params.LCSS.Epsilon=0.7; Params.LCSS.Sigma=8;
% 
% Params.TimeSeries.DescriptionType='TorsoDists'; % TorsoDists or PairDist  
% Params.TimeSeries.Version='Standard';  %  Standard  or FirstDereviate
% Params.TimeSeries.dereviateStep=3; % > should be more than 1
[TrainDescriptors,TestDescriptors]=PrepareDescriptors(TrainData,TestData,Params);

Params.TimeSereies.kCntNeibor=1;

switch Params.DTW.Featureweighting
    case 0
        DTW_Weights=ones(size(TrainDescriptors{1,1},1),1);
    case 2
        DTW_Weights=Calc_DTWweights_byRyese(TrainDescriptors_Torso,TrainTargets,Params);
end

Trained_outputs=TrainDTW(TrainDescriptors,TrainData.Targets,Params,DTW_Weights);

% Epsilons=[0.7];
% Sigmas=[ 8];
% for i=1:length(Epsilons)
%   for j=1:length(Sigmas)
%     Params.LCSS.Epsilon=Epsilons(i);
%     Params.LCSS.Sigma=Sigmas(j);
%     Test_outputs=Apply_DTW(TrainDescriptors,TrainData.Targets,TestDescriptors,TestData.Targets,Params,DTW_Weights);
%     Acc(i,j)=Test_outputs.Accuracy
%   end
% end
% [BestAcc,ind]=max(Acc(:));
% [m,n]=ind2sub(size(Acc),ind);
% Params.LCSS.Epsilon=Epsilons(m);
% Params.LCSS.Sigma=Sigmas(n);

Test_outputs=Apply_DTW(TrainDescriptors,TrainData.Targets,TestDescriptors,TestData.Targets,Params,DTW_Weights);
end

function Trained_DTW=TrainDTW(TrainDescriptors,TrainTargets,Params,Weights)

N_Train = length(TrainDescriptors) ;

%%% reorder samples
r=1:length(TrainDescriptors) ; %r=randperm(length(TrainDescriptors));
TrainDescriptors=TrainDescriptors(r);
TrainTargets=TrainTargets(r);

Trained_DTW.Abstract_level_output=[];
Trained_DTW.Rank_level_output=[];
Trained_DTW.Measurment_level_output=[];
Trained_DTW.Accuracy=[];
Trained_DTW.ConfusionMatrix=[];

for c=1:Params.N_validation_folds   
   if c==1,
      N_Train_fold=round(N_Train/2);
      N_Valid_fold=N_Train-N_Train_fold;
      selTrain      = find(mod(0:N_Train_fold-1, N_Train_fold+N_Valid_fold) < N_Train_fold) ;
      selValidation = setdiff(1:N_Train, selTrain) ;
   else
      a=selTrain;
      selTrain=selValidation;
      selValidation=a;
   end
   
   TrainDescrs_fold =       TrainDescriptors(selTrain);
   TrainTars =              TrainTargets(selTrain);
   ValidSamples_Descrs =    TrainDescriptors(selValidation);
   ValidationTars=          TrainTargets(selValidation);
   
   Trained_DTW_fold=Apply_DTW(TrainDescrs_fold, TrainTars,ValidSamples_Descrs,ValidationTars,Params,Weights) ;
   
   Trained_DTW.Abstract_level_output=     [Trained_DTW.Abstract_level_output;Trained_DTW_fold.Abstract_level_output];
   Trained_DTW.Rank_level_output=         [Trained_DTW.Rank_level_output;Trained_DTW_fold.Rank_level_output];
   Trained_DTW.Measurment_level_output=   [Trained_DTW.Measurment_level_output;Trained_DTW_fold.Measurment_level_output];
   Trained_DTW.Accuracy=                  mean([Trained_DTW.Accuracy;Trained_DTW_fold.Accuracy]);
   ConfusionMatrix(:,:,c)=                Trained_DTW_fold.ConfusionMatrix;
   CM_ind{c}= Trained_DTW_fold.CM_ind;
end
a=[selTrain,selValidation];
[~,b]=sort(r);
order=a(b);  

Trained_DTW.Abstract_level_output=Trained_DTW.Abstract_level_output(order);
Trained_DTW.Measurment_level_output=Trained_DTW.Measurment_level_output(order,:);
Trained_DTW.Rank_level_output= Trained_DTW.Rank_level_output(order,:);
Trained_DTW.ConfusionMatrix=sum(ConfusionMatrix,3);
Trained_DTW.CM_ind=cellfun(@(x,y)[x,y],CM_ind{1},CM_ind{2}, 'UniformOutput', false);
end

function DTW_outputs=Apply_DTW(TrainDescriptors,TrainTargets,TestDescriptors,TestTargets,Params,W) 

%%% reduce the dimension of descriptors by choosing every n frames (n = Params.NBNN.framestep)
TrainDescriptors=cellfun(@(x) x(:,1:Params.DTW.framestep:end),TrainDescriptors,'UniformOutput',false);
TestDescriptors=cellfun(@(x) x(:,1:Params.DTW.framestep:end),TestDescriptors,'UniformOutput',false);

prediction=zeros(length(TestDescriptors),1);
DP=zeros(length(TestDescriptors),Params.N_sel_classes);
sorted_classes=DP;

if size(W,2)==1, W=repmat(W,1,Params.N_sel_classes); end;

for t=1:length(TestTargets)
%   if TestTargets(t)==3
    TestSample=TestDescriptors{t};
%     figure,plot(TestSample','DisplayName','TestPatterns','YDataSource','TestPatterns');figure(gcf)
    [prediction(t),sklScoreList,sampleDP]=CalcSklScoreList_v3(TestSample, TrainDescriptors',TrainTargets,Params,W);
    %sample_DP=sample_DP./sum(sample_DP);
    DP(t,:)=sampleDP; %sample_DP;
    [~,sorted_classes(t,:)]=sort(DP(t,:),'descend');
%   end
end

DP=mapminmax(DP,0,1);

DTW_outputs.Abstract_level_output=prediction;
DTW_outputs.Rank_level_output=sorted_classes;
DTW_outputs.Measurment_level_output=DP;
DTW_outputs.Accuracy=100*mean(prediction==TestTargets);
DTW_outputs.ConfusionMatrix=confusionmat(TestTargets,prediction);

N_class=size(DP,2);
[~,~,CM_ind,~]=confusion(Vectorize_targets(TestTargets,N_class),DP');
DTW_outputs.CM_ind=CM_ind;
end

function [TrainDescriptors,TestDescriptors]=PrepareDescriptors(TrainData,TestData,Params)

switch Params.TimeSeries.DescriptionType
  case 'TorsoDists'
    TrainDescriptors =   TrainData.ExtrFeat_TorsoDist;
    TestDescriptors  =   TestData.ExtrFeat_TorsoDist;
  case 'PairDists'
    TrainDescriptors =   TrainData.ExtrFeat_PairDist;
    TestDescriptors  =   TestData.ExtrFeat_PairDist;
end

switch Params.TimeSeries.Version
  case 'Standard'
    % Do Nothing
  case 'FirstDereviate'
    c=Params.TimeSeries.dereviateStep;
    TrainDescriptors =   cellfun(@(x) x(:,c:end)-x(:,1:end-c+1),TrainDescriptors,'UniformOutput',false);
    TestDescriptors =    cellfun(@(x) x(:,c:end)-x(:,1:end-c+1),TestDescriptors,'UniformOutput',false);
end

Params.DTW.CutBegEndframes=4;
% TrainDescriptors=cellfun(@(x) x(:,4:end-4),TrainDescriptors,'UniformOutput',false);
% TestDescriptors=cellfun(@(x) x(:,4:end-4),TestDescriptors,'UniformOutput',false);

end
