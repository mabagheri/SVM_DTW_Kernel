function Data=DescribeSkeletonData_v3(Data,pars) % Extract features

RawSkeleton                 = Data.RawSkeleton;
AllDescriptors_TorsoDist    = cell(size(RawSkeleton));
AllDescriptors_PairDist     = cell(size(RawSkeleton));

%% Decribe using position of interesting skeleton joint points relative to Torso
Interesting_points= pars.FE.TorsoDist_suitable_points;
N_joint_points=length(Interesting_points);
for i=1:size(RawSkeleton,1)
   sample_pattern=RawSkeleton{i};
   if strcmp(pars.dataset_family,'CAD'),
       centerPntPosi = sample_pattern(3,:,:);
   else
       centerPntPosi = mean([sample_pattern(1,:,:); sample_pattern(2,:,:); sample_pattern(3,:,:); ], 1);
   end
   sample_descriptor=sample_pattern(Interesting_points,:,:)-repmat(centerPntPosi ,N_joint_points,1);
   sample_descriptor=shiftdim(sample_descriptor,2);
   sample_descriptor=reshape(sample_descriptor,N_joint_points*3,[]); % the first three rows are for the first interesting points and so on  
   AllDescriptors_TorsoDist{i,1}=sample_descriptor;
end
Data.ExtrFeat_TorsoDist=AllDescriptors_TorsoDist;


%% Decribe using position of interesting skeleton joint points relative to each other (pairwise distances)
N_joint_points=length(pars.FE.PairDist_suitable_points);
for i=1:size(RawSkeleton,1)
  sample_pattern=RawSkeleton{i};
  sample_pattern=sample_pattern(pars.FE.PairDist_suitable_points,:,:);
  All_Frames_PairDist=[];
  
  switch pars.Description.PairWiseDist_version 
    case 1
      ind=1;
      for j1=1:N_joint_points
        for j2=j1+1:N_joint_points
          All_Frames_PairDist(ind,:)=(sample_pattern(j1,:,1)-sample_pattern(j2,:,1)).^2 + ...
            (sample_pattern(j1,:,2)-sample_pattern(j2,:,2)).^2 + ...
            (sample_pattern(j1,:,3)-sample_pattern(j2,:,3)).^2;
          ind=ind+1;
        end
        %plot(All_Frames_PairDist');figure(gcf); legend(Appendix(1:2:end))
      end       
    case 2
      ind=1;
      for j1=1:N_joint_points
        for j2=j1+1:N_joint_points
          All_Frames_PairDist(ind:ind+2,:)=[sample_pattern(j1,:,1)-sample_pattern(j2,:,1); ...
            sample_pattern(j1,:,2)-sample_pattern(j2,:,2); ...
            sample_pattern(j1,:,3)-sample_pattern(j2,:,3)];
          ind=ind+3;
        end
      end
      
  end 
  AllDescriptors_PairDist{i,1}=All_Frames_PairDist;
end

Data.ExtrFeat_PairDist=AllDescriptors_PairDist;

%% Decribe using movement of Torso-based distance
% dt=pars.description.tempral_diff;
% amplifier_dt=pars.description.tempral_diff_amplifier;
% Data.ExtrFeat_TorsoMovement=cellfun(@(x) amplifier_dt*diff(x,dt,2), AllDescriptors_TorsoDist,'UniformOutput',false);

end