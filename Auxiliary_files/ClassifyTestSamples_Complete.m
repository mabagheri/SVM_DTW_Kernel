function [Predicted_class,CLF_Test_output,CM_ind]= ... %[Predicted_class,Ranked_class,DP]=...
   ClassifyTestSamples_Complete(TrainedCLF,TrainIn, TrainTargets, ...
   TestPatterns,TestTargets,ClassifierType,N_class,Params)

switch (ClassifierType)
   case {1} %MLP
      net_Testout=sim(TrainedCLF,TestPatterns');
      [~,Predicted_class]=max(net_Testout);
      Predicted_class=Predicted_class';
      
      % Produce 3 types of label output
      DP=mapminmax(net_Testout',0,1); % measurment level output (Decision profile)
      [~,Ranked_class]=sort(DP,2,'descend'); % the rank level output
      %[Train_error,ConfusionMatrix,CM_ind,CM_per]=confusion(Train_V_Targets,net_TrainOut);
      accuracy=sum(TestTargets==Predicted_class)/length(TestTargets);
      
      ConfusionMatrix=confusionmat(TestTargets,Predicted_class);
      %   [~,~,CM_ind,~]=confusion(Vectorize_targets(TestTargets,N_class),DP');
      
      
   case {2} %KNN (1NN)
      [Predicted_class,DP] = myknnclassify(TestPatterns, TrainIn, TrainTargets,pars.KNN.K,pars.KNN.distance,'nearest',Params);
      [~,Ranked_class]=sort(DP,2,'descend'); % the rank level output
      accuracy=mean(Predicted_class==TestTargets);
      
   case {3}  % LIB SVM
      [Predicted_class,accuracy,prob_est]=svmpredict2(TestTargets,TestPatterns,TrainedCLF,' -b 1');
      accuracy=accuracy(1);
      DP(:,TrainedCLF.Label)=prob_est;
      [~,Ranked_class]=sort(DP,2,'descend'); % the rank level output
      ConfusionMatrix=confusionmat(TestTargets,Predicted_class);
      [~,~,CM_ind,~]=confusion(Vectorize_targets(TestTargets,N_class),DP');
      
   case 4 %liblinear
      [Predicted_class,accuracy,prob_est]=predict(TestTargets,sparse(TestPatterns),TrainedCLF);
      accuracy=accuracy(1);
      DP(:,TrainedCLF.Label)=prob_est;
      [~,Ranked_class]=sort(DP,2,'descend'); % the rank level output
      ConfusionMatrix=confusionmat(TestTargets,Predicted_class);
      [~,~,CM_ind,~]=confusion(Vectorize_targets(TestTargets,N_class),DP');
      
   case 5  % LIBSVM smaji
      [Predicted_class,accuracy,prob_est]=svmpredict_smaji(TestTargets,TestPatterns,TrainedCLF,'-b 1');
      accuracy=accuracy(1);
      DP(:,TrainedCLF.Label)=prob_est;
      [~,Ranked_class]=sort(DP,2,'descend'); % the rank level output
      ConfusionMatrix=confusionmat(TestTargets,Predicted_class);
      [~,~,CM_ind,~]=confusion(Vectorize_targets(TestTargets,N_class),DP');
      
   case 6
      [Predicted_class,accuracy,prob_estimates]=svmpredict_vlfeat(TestTargets',TestPatterns',TrainedCLF);
      
      DP=mapminmax(prob_estimates,0,1);
      [~,Ranked_class]=sort(DP,2,'descend'); % the rank level output
      ConfusionMatrix=confusionmat(TestTargets,Predicted_class);
      [~,~,CM_ind,~]=confusion(Vectorize_targets(TestTargets,N_class),DP');

      
   case 7
      Predicted_class=treeval(TrainedCLF.Tree,TestPatterns);
      Predicted_class=TrainedCLF.ClassIndex(Predicted_class);
      accuracy=mean(Predicted_class==TestTargets);
      Ranked_class=zeros(size(TestPatterns,1),N_class);
      DP=transpose(Vectorize_targets(Predicted_class,N_class));
      %[Train_error,ConfusionMatrix,CM_ind,CM_per]=confusion(Train_V_Targets,DP);
end

CLF_Test_output.Abstract_level_output=Predicted_class;
CLF_Test_output.Rank_level_output=Ranked_class;
CLF_Test_output.Measurment_level_output=DP;
CLF_Test_output.Accuracy=accuracy;
CLF_Test_output.ConfusionMatrix=ConfusionMatrix;
CLF_Test_output.CM_ind=CM_ind;

% CLF_Test_output={class Ranked_class DP accuracy ConfusionMatrix CM_ind};
% CLF_Test_output = struct('Abstract_level_output'     , Predicted_class,...
%                          'Rank_level_output'         , Ranked_class, ...
%                          'Measurment_level_output'   , DP, ...
%                          'Accuracy'                  , accuracy, ...
%                          'ConfusionMatrix'           , ConfusionMatrix);%,...
% %                          'CM_ind',                   CM_ind);
% %                          'ConfusionMatri_Percentage' , CM_per);

end