function [Trained_outputs,Tests_SVM_DTWkernel_outputs]=Learn_by_SVM_TSDKernels(TrainData,TestData,Params)

% -------------------------------------------------------------------------
%                                                          Extract features
% -------------------------------------------------------------------------
[TrainDescriptors,TestDescriptors]=PrepareDescriptors(TrainData,TestData,Params);
tic
TrainPreCompKernels_1=ComputeTimeSeriesDist(TrainDescriptors,TrainDescriptors,Params);%
TestPreCompKernels_1=ComputeTimeSeriesDist(TestDescriptors,TrainDescriptors,Params);
toc
% --------------------------------------------------
% Params.TimeSeries.DescriptionType='TorsoDists';
% Params.TimeSeries.Version='Standard'; %  Standard  or FirstDereviate
% Params.TimeSeries.dereviateStep=3; % > should be more than 1
% [TrainDescriptors,TestDescriptors]=PrepareDescriptors(TrainData,TestData,Params);

% Params.TimeSeries.Type='DTW'; % DTW  or LCSS
TrainPreCompKernels_2=TrainPreCompKernels_1;%ComputeTimeSeriesDist(TrainDescriptors,TrainDescriptors,Params);
TestPreCompKernels_2=TestPreCompKernels_1;%ComputeTimeSeriesDist(TestDescriptors,TrainDescriptors,Params);

% --------------------------------------------------
w=1.0;
TrainPreCompKernels=w * TrainPreCompKernels_1 + (1-w) * TrainPreCompKernels_2;
TestPreCompKernels= w * TestPreCompKernels_1  + (1-w) * TestPreCompKernels_2;

% --------------------------------------------------
N_sel_classes=length(Params.dataset.selected_classes);
Train_V_Targets=Vectorize_targets(TrainData.Targets,N_sel_classes);
ClassifierType=3;
Params.SVM.Params='-t 4 -q';

RangeOf_c=[1 10 100];  cn=0;
for c = RangeOf_c,
  cn=cn+1;
  Params.SVM.Params = ['-t 4 -c ', num2str(c),  ' -q'];
  Params.SVM.doGridSearch=0;
  
  if ClassifierType==3 && Params.SVM.doGridSearch==1,
    %   Params.SVM.Params=SVM_GridSearch_v2([TrainTargets;TestTargets],AllFeatures,Params);
    [Params.SVM.Params]=SVM_GridSearch_v3(TrainData.Targets,TrainPreCompKernels,TestData.Targets,TestPreCompKernels,Params);
  end
  
  [Trained_classifier,Trained_outputs]=...
    TrainClassifier_Complete(TrainPreCompKernels,TrainData.Targets,Train_V_Targets,ClassifierType,Params);
  %TrAcc(cn)=Trained_outputs.Accuracy;
  
  [~,Tests_SVM_DTWkernel_outputs]=...
    ClassifyTestSamples_Complete(Trained_classifier,TrainPreCompKernels,TrainData.Targets, ...
    TestPreCompKernels,TestData.Targets,ClassifierType,N_sel_classes,Params);
  Acc(cn)=Tests_SVM_DTWkernel_outputs.Accuracy %#ok<AGROW>
end

[~, Ind]=max(Acc);

Params.SVM.Params = ['-t 4 -c ', num2str(RangeOf_c(Ind)),  ' -q'];
[Trained_classifier,Trained_outputs]=...
    TrainClassifier_Complete(TrainPreCompKernels,TrainData.Targets,Train_V_Targets,ClassifierType,Params);
[~,Tests_SVM_DTWkernel_outputs]=...
    ClassifyTestSamples_Complete(Trained_classifier,TrainPreCompKernels,TrainData.Targets, ...
    TestPreCompKernels,TestData.Targets,ClassifierType,N_sel_classes,Params);

end

function PreCompKernels=ComputeTimeSeriesDist(Descriptors1,Descriptors2,Params)

PreCompKernels=zeros(length(Descriptors1),length(Descriptors2));

if isequal(Descriptors1,Descriptors2),
  for i=1:length(Descriptors1)
    for j=i+1:length(Descriptors2)
      switch Params.TimeSeries.Type % DTW  or  LCSS
        case 'DTW'
          PreCompKernels(i,j) =  CalcDTWDist(Descriptors1{i}, Descriptors2{j});
        case 'LCSS'
          n=size(Descriptors1{i},2);  m=size(Descriptors2{j},2);
          LCSSDist= (n+ m - 2* CalcLCSSsimilarity_v2  (Descriptors1{i}, Descriptors2{j},Params.LCSS.Epsilon,Params.LCSS.Sigma))/(n+m);
          PreCompKernels(i,j) = LCSSDist; % (1 + 0.1) / (LCSSDist + 0.1);
          %LCSSDist2=(n+ m - 2* CalcLCSSDist_matlab_me (Descriptors1{i}, Descriptors2{j},Params.LCSS.Epsilon,Params.LCSS.Sigma))/(n+m);
          %PreCompKernels2(i,j)=LCSSDist2;
      end
    end
  end
  PreCompKernels=[(1:length(Descriptors1))' , PreCompKernels + PreCompKernels' + eye(i)];
else
  PreCompKernels=zeros(length(Descriptors1),length(Descriptors2));
  for i=1:length(Descriptors1)
    for j=1:length(Descriptors2)
      switch Params.TimeSeries.Type % DTW  or  LCSS
        case 'DTW'
          PreCompKernels(i,j) =  CalcDTWDist(Descriptors1{i}, Descriptors2{j});
        case 'LCSS'
          n=size(Descriptors1{i},2);  m=size(Descriptors2{j},2);
          LCSSDist=(n+ m - 2* CalcLCSSsimilarity_v2(Descriptors1{i}, Descriptors2{j},Params.LCSS.Epsilon,Params.LCSS.Sigma))/(n+m);
          PreCompKernels(i,j) = LCSSDist; % (1 + 0.1) / (LCSSDist + 0.1);
      end
    end
  end
  PreCompKernels=[(1:length(Descriptors1))' ,PreCompKernels];
  
end
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

switch Params.TimeSeries.dimension
    case 'x'
        TrainDescriptors = cellfun(@(x) x(1:3:end,:),TrainDescriptors,'UniformOutput',false);
        TestDescriptors  = cellfun(@(x) x(1:3:end,:),TestDescriptors,'UniformOutput',false);
    case 'y'
        TrainDescriptors = cellfun(@(x) x(2:3:end,:),TrainDescriptors,'UniformOutput',false);
        TestDescriptors  = cellfun(@(x) x(2:3:end,:),TestDescriptors,'UniformOutput',false);
    case 'z'
        TrainDescriptors = cellfun(@(x) x(3:3:end,:),TrainDescriptors,'UniformOutput',false);
        TestDescriptors  = cellfun(@(x) x(3:3:end,:),TestDescriptors,'UniformOutput',false);        
    case 'xyz'
        % do nothing
end

switch Params.TimeSeries.Version
  case 'Standard'
    % Do Nothing
  case 'FirstDereviate'
    c=Params.TimeSeries.dereviateStep;
    TrainDescriptors =   cellfun(@(x) x(:,c:end)-x(:,1:end-c+1),TrainDescriptors,'UniformOutput',false);
    TestDescriptors =    cellfun(@(x) x(:,c:end)-x(:,1:end-c+1),TestDescriptors,'UniformOutput',false);
end

end
