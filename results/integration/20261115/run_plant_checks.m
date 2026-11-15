root = fileparts(mfilename('fullpath'));
project_root = fileparts(fileparts(fileparts(root)));
addpath(fullfile(project_root,'src','drone','closed_loop'));
set(groot,'defaultFigureVisible','off');
Simulink.fileGenControl('set','CacheFolder',fullfile(root,'simulink_cache'), ...
    'CodeGenFolder',fullfile(root,'simulink_codegen'),'createDir',true);
validate_pipeline_exports({fullfile(root,'z_ppo'),fullfile(root,'z_sac'), ...
    fullfile(root,'z_dqn'),fullfile(root,'z_psi_ppo')},fullfile(root,'matlab_policy_validation.json'));
EINT_RL_eval(fullfile(root,'z_ppo'),0,fullfile(root,'plant_z_ppo'));
EINT_RL_eval2(fullfile(root,'z_psi_ppo'),0,fullfile(root,'plant_z_psi_ppo'));
