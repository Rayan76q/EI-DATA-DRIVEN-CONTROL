function report = validate_pipeline_exports(export_dirs, report_path)
%VALIDATE_PIPELINE_EXPORTS Check exported actors and RNNs without Simulink.
checks = cell(size(export_dirs));
for k = 1:numel(export_dirs)
    check = check_policy_export(export_dirs{k});
    check.export_dir = export_dirs{k};
    check.rnn = check_rnn_export(export_dirs{k});
    checks{k} = check;
end
report = struct('matlab_version', version, 'policy_checks', {checks}, ...
    'simulink_installed', exist('sim', 'file') > 0 && exist('load_system', 'file') > 0, ...
    'simulink_licensed', logical(license('test', 'Simulink')));
if nargin > 1
    file = fopen(report_path, 'w');
    assert(file >= 0, 'Cannot write validation report.');
    cleanup = onCleanup(@() fclose(file));
    fprintf(file, '%s\n', jsonencode(report));
end
end
