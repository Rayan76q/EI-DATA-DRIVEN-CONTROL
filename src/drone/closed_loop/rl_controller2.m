function [u1, u2, u3, u4, z_ref, obs] = rl_controller2(t, z, vz, az_prev, ...
    phi, theta, psi, wx, wy, wz, wdotx_prev, wdoty_prev, wdotz_prev, ...
    u1_prev, u2_prev, u3_prev, u4_prev)
%RL_CONTROLLER2 Evaluate [thrust, yaw] and mix them into four motor commands.
% The Simulink block passes the full state; this policy uses six observations.
persistent P loaded_policy
export_dir = getenv('DRONE_POLICY_EXPORT_DIR');
assert(~isempty(export_dir), 'Run EINT_RL_eval with a Python export directory.');
policy_file = fullfile(export_dir, 'policy.mat');
if isempty(P) || ~strcmp(loaded_policy, policy_file)
    P = load(policy_file);
    loaded_policy = policy_file;
end
[z_ref, psi_ref] = drone_policy_reference(P, t);
assert(size(P.W1, 2) == 6, 'Altitude/yaw controller requires six observations.');

% Keep yaw unwrapped so that a full 2*pi turn differs from zero rotation.
obs = double([z-z_ref; vz; az_prev; psi-psi_ref; wz; wdotz_prev]);
action = drone_policy_action(P, obs);
assert(numel(action) == 2, 'Altitude/yaw controller requires [thrust, yaw].');
motors = double(P.motor_mixer) * action;
assert(numel(motors) == 4, 'Controller must return four motor commands.');
u1 = motors(1);
u2 = motors(2);
u3 = motors(3);
u4 = motors(4);
end
