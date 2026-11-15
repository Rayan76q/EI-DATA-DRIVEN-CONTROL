root = fileparts(mfilename('fullpath'));
project_root = fileparts(fileparts(fileparts(root)));
addpath(fullfile(project_root,'data','data_generator'));
Simulink.fileGenControl('set','CacheFolder',fullfile(root,'simulink_cache'), ...
    'CodeGenFolder',fullfile(root,'simulink_codegen'),'createDir',true);
generate_training_data('z',fullfile(root,'generated_data'),3,.3,0);
generate_training_data('z_psi',fullfile(root,'generated_data'),3,.3,0);
load_system(fullfile(project_root,'src','drone','plant','EINT_system_simulationPID.slx'));
close_system('EINT_system_simulationPID',0);
fid = fopen(fullfile(root,'data_generation_checks_passed.txt'),'w');
fprintf(fid,'Both generators and the PID model imported from main passed.\n');fclose(fid);
