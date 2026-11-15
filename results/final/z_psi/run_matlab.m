% Validate inference, then run the policy on the Simulink plant.
addpath('/home/rayan768/CS/drone/EI-DATA-DRIVEN-CONTROL/src/drone/closed_loop');
report = struct('status', 'failed');
try
    report.policy = check_policy_export('/home/rayan768/CS/drone/EI-DATA-DRIVEN-CONTROL/results/final/z_psi');
    if isfile(fullfile('/home/rayan768/CS/drone/EI-DATA-DRIVEN-CONTROL/results/final/z_psi', 'rnn.mat'))
        report.rnn = check_rnn_export('/home/rayan768/CS/drone/EI-DATA-DRIVEN-CONTROL/results/final/z_psi');
    end
    if exist('sim', 'file') > 0 && exist('load_system', 'file') > 0
        Simulink.fileGenControl('set', 'CacheFolder', '/home/rayan768/CS/drone/EI-DATA-DRIVEN-CONTROL/results/final/z_psi/simulink_cache', ...
            'CodeGenFolder', '/home/rayan768/CS/drone/EI-DATA-DRIVEN-CONTROL/results/final/z_psi/simulink_codegen', 'createDir', true);
        EINT_RL_eval('/home/rayan768/CS/drone/EI-DATA-DRIVEN-CONTROL/results/final/z_psi', 0, '/home/rayan768/CS/drone/EI-DATA-DRIVEN-CONTROL/results/final/z_psi/plant');
        report.status = 'completed';
    else
        report.status = 'simulink_unavailable';
        report.reason = 'MATLAB actor checks passed; install Simulink to run the plant.';
    end
catch exception
    report.reason = exception.message;
    report.identifier = exception.identifier;
    disp(getReport(exception, 'extended'));
end
fid = fopen('/home/rayan768/CS/drone/EI-DATA-DRIVEN-CONTROL/results/final/z_psi/matlab_status.json', 'w');
fprintf(fid, '%s\n', jsonencode(report));
fclose(fid);
if strcmp(report.status, 'failed')
    error('drone:ValidationFailed', '%s', report.reason);
end
