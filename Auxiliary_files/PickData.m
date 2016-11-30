function ChosenSamples=PickData(Data,TrainIndices)
  Items=fieldnames(Data);
  N_items=numel(Items);
  
  for i=1:N_items
    ChosenSamples.(Items{i})=Data.(Items{i})(TrainIndices,:);  
  end
%   ChosenSamples=Data;
end
