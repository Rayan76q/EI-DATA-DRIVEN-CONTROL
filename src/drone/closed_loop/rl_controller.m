function [u, z_ref, obs] = rl_controller(t, z, vz, az_prev, u_prev) %#ok<INUSD>
%RL_CONTROLLER Build the three observations expected by the altitude policy.
% The unused previous action remains part of the Simulink block interface.
persistent P loaded_policy
export_dir = getenv('DRONE_POLICY_EXPORT_DIR');
assert(~isempty(export_dir), 'Run EINT_RL_eval with a Python export directory.');
policy_file = fullfile(export_dir, 'policy.mat');
if isempty(P) || ~strcmp(loaded_policy, policy_file)
    P = load(policy_file);
    loaded_policy = policy_file;
end
z_ref = drone_policy_reference(P, t);
obs = double([z - z_ref; vz; az_prev]);
action = drone_policy_action(P, obs);
assert(numel(action) == 1, 'Altitude controller requires one physical action.');
u = action(1);
end
