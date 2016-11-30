function V_Targets=Vectorize_targets(Targets,N_class)
N_Samples=length(Targets);
V_Targets=zeros(N_class,N_Samples);
temp=0:N_class:(N_Samples-1)*N_class;
V_Targets(Targets'+temp)=1; 
end
