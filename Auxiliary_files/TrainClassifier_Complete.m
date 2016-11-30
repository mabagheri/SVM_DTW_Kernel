function [TrainedCLF,CLF_Train_output]=...
   TrainClassifier_Complete(TrainPatterns,TrainTargets,Train_V_Targets,ClassifierType,pars)

switch (ClassifierType)
   
   case {1} % Multi layer perceptron  (MLP)
      CLF = newff(TrainPatterns',Train_V_Targets,pars.MLP.N_nodes,{pars.MLP.Fnc},'trainlm');
      CLF.trainParam.showWindow = false;
      CLF.trainParam.showCommandLine = false;
      TrainedCLF=train(CLF,TrainPatterns',Train_V_Targets);
      net_TrainOut=sim(TrainedCLF,TrainPatterns');
      
      % Produce 3 types of label output
      DP=mapminmax(net_TrainOut',0,1); % measurment level output (Decision profile)
      [temp,Predicted_class]=max(net_TrainOut); %  the abstract level output
      Predicted_class=Predicted_class';
      [temp,Ranked_class]=sort(net_TrainOut,'descend'); % the rank level output
      
      [TrainError,ConfusionMatrix,CM_ind,CM_per]=confusion(Train_V_Targets,net_TrainOut);
      accuracy=1-TrainError;
      
   case {2}  %KNN
      % Do Nothing!
      TrainedCLF=[];
      Predicted_class=[];
      Ranked_class=[];
      DP=[];
      ConfusionMatrix=[];
      CM_per=[];
      accuracy=[];
      
   case {3}  %LIB SVM
      SVM_Params=[pars.SVM.Params ' -b 1'];
      TrainedCLF = svmtrain2(TrainTargets, TrainPatterns,SVM_Params);
      [Predicted_class,accuracy,prob_estimates]=svmpredict2(TrainTargets,TrainPatterns,TrainedCLF,'-b 1');
      accuracy=accuracy(1);
      
      DP(:,TrainedCLF.Label)=prob_estimates;
      N_class=size(Train_V_Targets,1); N_Train=length(TrainTargets);
      if max(TrainTargets)~=N_class
         DP=[DP zeros(N_Train,N_class-max(TrainTargets))];
      end
      DP=mapminmax(DP,0,1);
      [~,Ranked_class]=sort(DP,2,'descend'); % the rank level output
      ConfusionMatrix=confusionmat(TrainTargets,Predicted_class);
      [~,~,CM_ind,~]=confusion(Train_V_Targets,DP');
      
   case {4} %LibLinear SVM
      LLSVM_Params=pars.LLSVM.Params;
      TrainedCLF = train(TrainTargets, sparse(TrainPatterns),LLSVM_Params);
      [Predicted_class, accuracy, prob_estimates]=llpredict(TrainTargets,sparse(TrainPatterns),TrainedCLF);
      accuracy=accuracy(1);
      
      DP(:,TrainedCLF.Label)=prob_estimates;
      N_class=size(Train_V_Targets,1); N_Train=length(TrainTargets);
      if max(TrainTargets)~=N_class
         DP=[DP zeros(N_Train,N_class-max(TrainTargets))];
      end
      DP=mapminmax(DP,0,1);
      [~,Ranked_class]=sort(DP,2,'descend'); % the rank level output
      ConfusionMatrix=confusionmat(TrainTargets,Predicted_class);
      [~,~,CM_ind,~]=confusion(Train_V_Targets,DP');
      
   case {5}  %LIBSVM smaji
      SVM_Params=[pars.SVM.Params ' -b 1'];  % t=5: Intersection kernel; 6:chi-squared; 7 -- Jenson-Shannon
      TrainedCLF = svmtrain_smaji(TrainTargets, TrainPatterns,SVM_Params);
      [Predicted_class,accuracy,prob_estimates]=svmpredict_smaji(TrainTargets,TrainPatterns,TrainedCLF,' -b 1');
      accuracy=accuracy(1);
      
      DP(:,TrainedCLF.Label)=prob_estimates;
      N_class=size(Train_V_Targets,1); N_Train=length(TrainTargets);
      if max(TrainTargets)~=N_class
         DP=[DP zeros(N_Train,N_class-max(TrainTargets))];
      end
      DP=mapminmax(DP,0,1);
      [~,Ranked_class]=sort(DP,2,'descend'); % the rank level output
      ConfusionMatrix=confusionmat(TrainTargets,Predicted_class);
      [~,~,CM_ind,~]=confusion(Train_V_Targets,DP');
      
   case 6 % Linear SVM _vlfeat
      opts.kernel='linear';
      opts.C=10;
      TrainedCLF= svmtrain_vlfeat(TrainTargets', TrainPatterns',opts);
      [Predicted_class,accuracy,prob_estimates]=svmpredict_vlfeat(TrainTargets',TrainPatterns',TrainedCLF);
      
      DP=mapminmax(prob_estimates,0,1);
      [~,Ranked_class]=sort(DP,2,'descend'); % the rank level output
      ConfusionMatrix=confusionmat(TrainTargets,Predicted_class);
      [~,~,CM_ind,~]=confusion(Train_V_Targets,DP');
      
      
   case{7} %decision tree
      N_class=size(Train_V_Targets,1);
      %     for ii=1:N_class
      %       Nc(ii)=sum(TrainTargets==ii);
      %     end
      %     SplitMin=max((min(10, min(Nc))),2);%A number n such that impure nodes must have n or more observations to be split (default 10).
      %     try
      %    SplitMin=10; %10;
      TrainedCLF.Tree = treefit(TrainPatterns,TrainTargets,'method','classification');%,'splitmin',SplitMin);
      TrainedCLF.ClassIndex=unique(TrainTargets);
      %     catch e
      %       TrainedCLF=[]; CLF_Train_output=[];
      %       return
      %     end
      Predicted_class=treeval(TrainedCLF.Tree,TrainPatterns);
      accuracy=mean(Predicted_class==TrainTargets);
      
      Ranked_class=zeros(size(TrainPatterns,1),N_class);
      DP=transpose(Vectorize_targets(Predicted_class,N_class));
      %[Train_error,ConfusionMatrix,CM_ind,CM_per]=confusion(Train_V_Targets,DP');
end

CLF_Train_output.Abstract_level_output=Predicted_class;
CLF_Train_output.Rank_level_output=Ranked_class;
CLF_Train_output.Measurment_level_output=DP;
CLF_Train_output.Accuracy=accuracy;
CLF_Train_output.ConfusionMatrix=ConfusionMatrix;
CLF_Train_output.CM_ind=CM_ind;


% CLF_Train_output={Predicted_class Ranked_class DP accuracy ConfusionMatrix CM_ind};
%%%% Train Outputs
% CLF_Train_output = struct('Abstract_level_output'     , Predicted_class,...
%   'Rank_level_output'         , Ranked_class, ...
%   'Measurment_level_output'   , DP, ...
%   'Accuracy'    , accuracy ,...
%   'ConfusionMatrix'           , ConfusionMatrix,...
%   'CM_ind',                   CM_ind);
% %   'ConfusionMatri_Percentage' , CM_per);
end