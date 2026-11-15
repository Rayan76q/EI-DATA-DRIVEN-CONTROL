root = fileparts(mfilename('fullpath'));
project_root = fileparts(fileparts(fileparts(root)));
addpath(fullfile(project_root,'src','drone','closed_loop'),fullfile(project_root,'data','data_generator'));
set(groot,'defaultFigureVisible','off');
Simulink.fileGenControl('set','CacheFolder',fullfile(root,'simulink_cache'), ...
    'CodeGenFolder',fullfile(root,'simulink_codegen'),'createDir',true);
EINT_RL_eval(fullfile(root,'z_dqn'),0,fullfile(root,'plant_z_dqn'));
EINT_RL_eval(fullfile(root,'z_sac'),0,fullfile(root,'plant_z_sac'));
generate_training_data('z',fullfile(root,'generated_data'),3,.3,0);
generate_training_data('z_psi',fullfile(root,'generated_data'),3,.3,0);
load_system(fullfile(project_root,'src','drone','plant','EINT_system_simulationPID.slx'));
close_system('EINT_system_simulationPID',0);
fid = fopen(fullfile(root,'remaining_matlab_checks_passed.txt'),'w');
fprintf(fid,'Additional policies, both data generators, and imported PID model loaded successfully.\n');fclose(fid);
