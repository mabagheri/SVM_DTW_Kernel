function pars=Set_Parameters_v3(pars)

% switch Stage

% =======================  Initialization  =======================
% case 'Initialization'
if strfind(pars.dataset_name,'Action3D'),         pars.dataset_family='MSRAction3D';        end;
if strfind(pars.dataset_name,'Daily'),            pars.dataset_family='MSRDailyActivity';   end;
if strfind(pars.dataset_name,'CAD'),              pars.dataset_family='CAD';                end;
if strfind(pars.dataset_name,'Caltech'),          pars.dataset_family='Caltech';            end;

switch pars.dataset_family
   case {'Caltech'}
      pars.original_data_path             = '..\..\Datasets\';
      pars.dataset.selected_classes       = 1:102;   
      pars.dataset.n_Train                = 30;  % number of samples to choose for training
      pars.dataset.n_Test                 = 15;  % number of samples to choose for testing
      pars.dataset.DataPartitioning_type  ='Fixed_Num_of_Samples_per_Class';
            
   case {'MSRAction3D'}
      pars.dataset.selected_classes=1:20;
      pars.dataset.DataPartitioning_type='fixed_cross_subject';
      pars.dataset.data_partition_TrSubjects = [1 3 5 7 9];
      pars.dataset.data_partition_TsSubjects = [2 4 6 8 10];
      pars.dataset.ClassNames= {'high arm wave', 'horizontal arm wave', 'hammer','hand catch', 'forward punch',...
         'high throw', 'draw x', 'draw tick', 'draw circle', 'hand clap', ...
         'two hand wave', 'sideboxing', 'bend', 'forward kick', 'side kick',...
         'jogging', 'tennis swing', 'tennis serve', 'golf swing', 'pick up&throw'};
      
   case {'MSRDailyActivity'}
      pars.dataset.selected_classes=1:16;
      pars.dataset.DataPartitioning_type='fixed_cross_subject';
      pars.dataset.data_partition_TrSubjects=[1 3 5 7 9];
      pars.dataset.data_partition_TsSubjects=[2 4 6 8 10];
      pars.dataset.ClassNames= {'drink','eat','read book','call cellphone','write on paper', 'use laptop','use vacuum cleaner',...
         'cheer up', 'sit still', 'toss paper', 'play game', 'lie down on sofa', 'walk', 'play guitar','stand up','sit down'};
      
   case 'MSRDailyActivity_PreComputedDCSF_f1'
      pars.dataset.selected_classes=1:12;
      pars.dataset.DataPartitioning_type='Predefined_MSRDailyActivity_Hugo';
      
   case 'Chalearn'
      pars.dataset.ClassNames={'vattene' ;  'vieniqui' ;  'perfetto' ;  'furbo' ;  'cheduepalle' ; ...
         'chevuoi' ;  'daccordo' ;  'seipazzo' ;  'combinato' ;  'freganiente' ; ...
         'ok' ;  'cosatifarei' ;  'basta' ;  'prendere' ;  'noncenepiu' ; ...
         'fame' ;  'tantotempo' ;  'buonissimo' ;  'messidaccordo' ;  'sonostufo' };
      
   case 'CAD'
      pars.dataset.selected_classes=1:12;
      pars.dataset.DataPartitioning_type='fixed_cross_subject';
      pars.dataset.data_partition_TrSubjects = {[2 3 4],[1 3 4],[1 2 4],[1 2 3]};
      pars.dataset.data_partition_TsSubjects = {1 , 2, 3, 4};
      pars.dataset.ClassNames={'talking on the phone';'writing on whiteboard';'drinking water';'rinsing mouth with water';'brushing teeth';...
         'wearing contact lenses';'talking on couch';'relaxing on couch';'cooking (chopping)';'cooking (stirring)';
         'opening pill container';'working on computer'};
end
pars.dataset.reordr_samples=1;

% =======================  Preprocessing  =======================
% case 'Preprocess'
pars.pre_process.do_cropping=0;             %only for action datasets
pars.pre_process.start_end_cut=[0 0];     %only for action datasets
pars.pre_process.remove_bad_samples=0;
pars.pre_process.bad_samples=[];
pars.pre_process.step=1;


% ======================= description / Feature Extraction =======================

pars.Description.PairWiseDist_version=2;
pars.FE.TorsoDist_suitable_points=[4,6,7,8,10,11,12,14,15,18,19];
pars.FE.PairDist_suitable_points= [4,6,7,8,10,11,12,14,15,18,19];      % MSRAction3D


% pars.FE.type='PairDist'; % For Action: TorsoDist PairDist TorsoDist&Motion DCSF DMM ---- For images: SIFT
% 
% switch pars.FE.type
%    case {'TorsoDist', 'PairDist', 'TorsoDist&Motion'}
%       pars.FE.motion_tempral_diff=2;
%       pars.FE.motion_tempral_diff_amplifier=10;
%       pars.FE.TorsoDist_suitable_points=[6,7,8,10,11,12,14,15,18,19];  % MSRAction3D
%       pars.FE.PairDist_suitable_points= [4,6,7,8,10,11,12,14,15,18,19];% MSRAction3D
%       % pars.FE.TorsoDist_suitable_points=[4 6 8  10 12 ]; % Chalearn
%       % pars.FE.PairDist_suitable_points=[4 6 8  10 12 ];               % Chalearn
%       % pars.FE.TorsoDist_suitable_points=[4,6,8,10,12,14,16,18,20];         % MSRDailyActivity
%       % pars.FE.PairDist_suitable_points =[1,4,6,7,8,10,11,12,14,16,18,20];  % MSRDailyActivity
%       % pars.FE.TorsoDist_suitable_points=1:15;  % CAD60
%       % pars.FE.PairDist_suitable_points= 1:15;  % CAD60
%       
%    case 'DCSF'
%       pars.FE.DCSF_TAO=8;  %for MSRDailyActivity=10;
%       pars.FE.DCSF_nPoints=160;  %for MSRDailyActivity=350;
%       pars.FE.DCSF_sigma=5;
%       
%    case 'DMM'
%       pars.FE.DMM_frame_remove=0.1; %percent of frames from beggining and end that we are going to remove
%       pars.FE.DMM_step=7;
%       pars.FE.DMM_crop_rows=11:230; %for CAD60
%       pars.FE.DMM_crop_cols=61:260; %for CAD60 
%       
%    case 'SIFT'
%       pars.FE.SIFT_gridSpacing   = 6;
%       pars.FE.SIFT_patchSize     = 16;
%       pars.FE.SIFT_maxImSize     = 300;
%       pars.FE.SIFT_nrml_threshold = 1; %low contrast region normalization threshold (descriptor length)
%       
%    case 'SIFT_vl'
%       pars.FE.SIFT_vl_step=4;
%       pars.FE.SIFT_vl_scales=2.^(0:-.5:-3);
% end
% 

end