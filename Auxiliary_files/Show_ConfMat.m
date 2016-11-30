function CM=Show_ConfMat(Targets,Abstract_level_output,ClassNames)
if nargin < 3
  ClassNames=[];
end
   CM_EXP = confusionmat(Targets,Abstract_level_output);
   CM=100*bsxfun(@times,CM_EXP ,1 ./ sum(CM_EXP,2));
   Title=sprintf('Average classification accuracy=%2.2f',100*sum(diag(CM_EXP))/sum(CM_EXP(:)));
   h = confmatrix(CM, ClassNames,Title);
end