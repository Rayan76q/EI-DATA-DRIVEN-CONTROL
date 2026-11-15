function [u, z_ref, obs] = rl_controller(t, z, vz, az_prev, u_prev) %#ok<INUSD>
%RL_CONTROLLER Altitude policy adapter called by EINT_RL_closed_loop.slx.
persistent P loaded_policy
export_dir = getenv('DRONE_POLICY_EXPORT_DIR');
if isempty(export_dir)
    export_dir = fullfile(fileparts(mfilename('fullpath')), 'legacy_exports');
end
policy_file = fullfile(export_dir, 'policy.mat');
if isempty(P) || ~strcmp(loaded_policy, policy_file)
    P = load(policy_file);
    loaded_policy = policy_file;
end
if isfield(P, 'mission_t')
    z_ref = drone_policy_reference(P, t);
else
    % Compatibility with the original altitude-only export.
    if t < 2, z_ref = 0; elseif t < 6, z_ref = 6; else, z_ref = -0.3; end
end
obs = double([z - z_ref; vz; az_prev]);
action = drone_policy_action(P, obs);
assert(numel(action) == 1, 'Altitude controller requires one physical action.');
u = action(1);
end
