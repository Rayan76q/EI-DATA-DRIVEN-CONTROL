root = fileparts(mfilename('fullpath'));
project_root = fileparts(fileparts(fileparts(root)));
addpath(fullfile(project_root,'data','data_generator'));
connect_training_inputs(fullfile(project_root,'src','drone','plant','EINT_system_simulation.slx'));
connect_training_inputs(fullfile(project_root,'matlab_files','EINT_system_simulation.slx'));
Simulink.fileGenControl('set','CacheFolder',fullfile(root,'simulink_cache'), ...
    'CodeGenFolder',fullfile(root,'simulink_codegen'),'createDir',true);
generate_training_data('z',fullfile(root,'generated_data'),3,.3,0);
generate_training_data('z_psi',fullfile(root,'generated_data'),3,.3,0);
