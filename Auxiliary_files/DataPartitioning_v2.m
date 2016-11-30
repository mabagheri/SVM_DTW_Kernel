function [TrainData,TestData]=DataPartitioning_v2(Data,pars)

tr_idx = [];
ts_idx = [];

switch pars.dataset.DataPartitioning_type
  
  case{'Fixed_Num_of_Samples_per_Class'} % e.x. for Caltech101
    
    for cc=1:numel(pars.dataset.selected_classes)
      class=pars.dataset.selected_classes(cc);
      idx_label = find(Data.Targets==class);    
      
      tr_idx = [tr_idx, vl_colsubset(idx_label', pars.dataset.n_Train,'beginning')];
      ts_idx = [ts_idx,  vl_colsubset(setdiff(idx_label', tr_idx'), pars.dataset.n_Test,'beginning')];

      % num=length(idx_label);
      % idx_rand = randperm(num);
      % tr_idx = [tr_idx; idx_label(idx_rand(1:pars.dataset.n_Train))];
      % ts_idx = [ts_idx; idx_label(idx_rand(pars.dataset.n_Train+1:pars.dataset.n_Train+pars.dataset.n_Test))];
    end
    tr_idx=tr_idx'; ts_idx=ts_idx';
    
  case {'fixed_cross_subject'} % for MSR-Action3D dataset (subject 1,3,5,7,9 for train and the other subjects for test)
    train_actors=pars.dataset.data_partition_TrSubjects;
    test_actors =pars.dataset.data_partition_TsSubjects;
    % train_actors = [1 3 5 7 9]; test_actors = [2 4 6 8 10];
    % train_actors = 1:5; test_actors = 6: 10;
        
    for cc=1:numel(pars.dataset.selected_classes) %N_sel_classes
      class=pars.dataset.selected_classes(cc);
      class_selected_samples = find(Data.Targets==class);
      for i=train_actors
        Selected_samples_train_i=(Data.Subjects(class_selected_samples)==i);
        tr_idx = [tr_idx;class_selected_samples(Selected_samples_train_i)];
      end
      for i=test_actors
        Selected_samples_test_i=(Data.Subjects(class_selected_samples)==i);
        ts_idx = [ts_idx;class_selected_samples(Selected_samples_test_i)];
      end
    end   
    
  case 'Predefined_MSRDailyActivity_Hugo'
    tr_idx=1:Data.TrainTest_indicies(1);
    ts_idx=Data.TrainTest_indicies(1)+1:Data.TrainTest_indicies(1)+Data.TrainTest_indicies(2);
    tr_idx=tr_idx'; ts_idx=ts_idx';
    
end

if pars.dataset.reordr_samples
    tr_idx=tr_idx(randperm(length(tr_idx)));
    ts_idx=ts_idx(randperm(length(ts_idx)));
end

Selected_samples=[tr_idx;ts_idx];
Selected_data=PickData(Data,Selected_samples);
[~,~,Selected_data.Targets]=unique(Selected_data.Targets);

TrainData=PickData(Selected_data,1:length(tr_idx));
TestData=PickData(Selected_data,length(tr_idx)+1:length(Selected_samples));

TrainData.Indices=tr_idx;
TestData.Indices=ts_idx;

end