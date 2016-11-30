function Data=pre_process(Data,pars)

if pars.pre_process.do_cropping
  SF=pars.pre_process.start_end_cut(1);
  EF=pars.pre_process.start_end_cut(2);
  step=pars.pre_process.step;
  Data.RawSkeleton=cellfun(@(x) x(:,SF:step:end-EF,:),Data.RawSkeleton,'UniformOutput',false);
end


if pars.pre_process.remove_bad_samples
  GoodSamples=1:length(Data.Targets);
  Currupted_Samples=pars.pre_process.bad_samples;

  switch pars.dataset_family          
      case {'MSRDailyActivity'}
          Durations=cell2mat(cellfun(@(x) size(x,2), Data.RawSkeleton,'UniformOutput',false));
          ind=find(Durations > (mean(Durations) + 3*std(Durations)));
          Currupted_Samples=[Currupted_Samples , ind];
          
      case {'MSRAction3D'}
          % % MyBadSamples=sort([67,68,69,541,542,543,76,77,78,134,135],'descend');
          % MyBadSamples=sort([541,542,543,560,561,567],'descend');
          % GoodSamples=1:567;
          % % BadSamples=[35,88, 169,362,363,364,374,556,558,567];
  end
  
  GoodSamples(Currupted_Samples)=[];
  Data=PickData(Data,GoodSamples);
end

% reorder_classed=0; % just a crazy thing to check some thing ! no need at all...
% if reorder_classed
% Data2.ClassNames=[];
% Data2.filenames=[];
% Data2.RawSkeleton=[];
% Data2.Subjects=[];
% Data2.Targets=[];
%   k=unique(Data.Targets);
%   k=randperm(max(k));
%
%   for c=1:length(k)
%     ind=find(Data.Targets==k(c));
%     Data2.ClassNames=[Data2.ClassNames; Data.ClassNames(ind)];
%     Data2.filenames=[Data2.filenames; Data.filenames(ind)];
%     Data2.RawSkeleton=[Data2.RawSkeleton; Data.RawSkeleton(ind)];
%     Data2.Subjects=[Data2.Subjects; Data.Subjects(ind)];
%     Data2.Targets=[Data2.Targets; c*ones(length(ind),1)];
%   end
%   Data=Data2;
% end


end
