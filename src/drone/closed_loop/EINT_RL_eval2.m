function out = EINT_RL_eval2(export_dir, noise_gain, results_dir)
%EINT_RL_EVAL2 Run an altitude/yaw export using the common plant evaluator.
if nargin < 1
    error('Pass the Python export directory.');
end
if nargin < 2
    noise_gain = 0;
end
if nargin < 3
    results_dir = [];
end
P = load(fullfile(export_dir, 'policy.mat'));
assert(size(P.W1, 2) == 6 && numel(P.action_low) == 2, ...
    'Export an altitude/yaw policy with six observations and two actions.');
out = EINT_RL_eval(export_dir, noise_gain, results_dir);
end
