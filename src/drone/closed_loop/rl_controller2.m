function [u1, u2, u3, u4, z_ref, obs] = rl_controller2(t, z, vz, az_prev, ...
    phi, theta, psi, wx, wy, wz, wdotx_prev, wdoty_prev, wdotz_prev, ...
    u1_prev, u2_prev, u3_prev, u4_prev)
%RL_CONTROLLER2 Adapt the existing four-motor diagram to the altitude/yaw policy.
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
[z_ref, psi_ref] = drone_policy_reference(P, t);
if size(P.W1, 2) == 6
    % Unwrapped yaw: 0 and a full 2*pi turn are distinct, as in DroneEnvRotation.
    obs = double([z-z_ref; vz; az_prev; psi-psi_ref; wz; wdotz_prev]);
    action = drone_policy_action(P, obs);
    assert(numel(action) == 2, 'Altitude/yaw controller requires [thrust,yaw].');
    motors = double(P.motor_mixer) * action;
elseif size(P.W1, 2) == 16
    % Existing four-motor policies retain their original observation contract.
    obs = double([z-z_ref; vz; az_prev; phi; theta; psi-psi_ref; wx; wy; wz; ...
                  wdotx_prev; wdoty_prev; wdotz_prev; u1_prev; u2_prev; u3_prev; u4_prev]);
    motors = drone_policy_action(P, obs);
else
    error('Unsupported altitude/yaw observation dimension: %d.', size(P.W1, 2));
end
assert(numel(motors) == 4, 'Controller must return four motor commands.');
u1 = motors(1); u2 = motors(2); u3 = motors(3); u4 = motors(4);
end
