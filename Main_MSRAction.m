clc; clear; close all
rng(1);
addpath('Auxiliary_files')

% ---------------------------------------------------------
% Please kindly cite one of the following papers:
%	[1]	M .A. Bagheri, Q. Gao, and S. Escalera, â€œSupport Vector Machines with Time Series Distance Kernels for Action Classificationâ€?, 
%	in Proc. IEEE Winter Conference on Applications of Computer Vision, New York, 2016.

%	[2]	M.A. Bagheri, Q. Gao, and S. Escalera, "Action Recognition by Pairwise Proximity Function Support Vector Machines with Dynamic Time Warping Kernels", 
%	in Proc. 29th Canadian conf. on Artificial Intelligence, BC, Canada, 2016.

% ---------------------------------------------------------

pars.dataset_name='MSRAction3D';    % MSRAction3D, MSRDailyActivity, Chalearn, UCF101
pars=Set_Parameters_v3(pars);

% -------------------------------------------------------------------------
%                              Load data,Partition data into train and test
% -------------------------------------------------------------------------
if ~exist('Data','var'), Data=importdata(fullfile('', [pars.dataset_name '.mat'])); end;

Data=pre_process(Data,pars);

% -------------------------------------------------------------------------
%                                    Describe Samples   =  Ectract Features
% -------------------------------------------------------------------------   
Data=DescribeSkeletonData_v3(Data,pars);
% visualize_Skeleton_data(Data);


[TrainData,TestData]=...  %Partion data into train and test
    DataPartitioning_v2(Data,pars);

%% Learning
disp('Learning by SVM with D-DTW Kernel')

% settings of this learning
pars.TimeSeries.Type                    = 'DTW';            % DTW  or  LCSS
pars.TimeSeries.DescriptionType         = 'TorsoDists';     % TorsoDists or PairDists
pars.TimeSeries.Version                 = 'FirstDereviate'; % Standard  or FirstDereviate
pars.TimeSeries.dereviateStep           = 2;                % Required if TimeSeries.Version='FirstDereviate' % should be more than 1
pars.TimeSeries.dimension               = 'xyz';            % dimension to use

[CLF_Train_output,CLF_Test_output]=Learn_by_SVM_TSDKernels(TrainData,TestData,pars);

Show_ConfMat(TestData.Targets,CLF_Test_output.Abstract_level_output,pars.dataset.ClassNames);
