root = fileparts(mfilename('fullpath'));
project_root = fileparts(fileparts(fileparts(root)));
addpath(fullfile(project_root,'src','drone','closed_loop'));
Simulink.fileGenControl('set','CacheFolder',fullfile(root,'simulink_cache'), ...
    'CodeGenFolder',fullfile(root,'simulink_codegen'),'createDir',true);
for name = {'20261105_202000_ztol02','20261105_193731_full'}
    EINT_RL_eval(fullfile(root,'sac_transfer',name{1}),0, ...
        fullfile(root,'sac_transfer',name{1},'plant'),false);
end
