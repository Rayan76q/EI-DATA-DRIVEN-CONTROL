function [u, z_ref, obs] = rl_controller(t, z, vz, az_prev, u_prev)
%RL_CONTROLLER  The trained policy, called at each step by EINT_RL_closed_loop.slx.
%   mission() and observation() match the Python environment. policy.mat is written by
%   export_for_matlab (export_policy.py).
persistent P loaded_policy
export_dir = getenv('DRONE_POLICY_EXPORT_DIR');
if isempty(export_dir)
    export_dir = fullfile(fileparts(mfilename('fullpath')), 'legacy_exports');
end
policy_file = fullfile(export_dir, 'policy.mat');
if isempty(P) || ~strcmp(loaded_policy, policy_file)
    P = load(policy_file);
    P.acts = strsplit(P.activations, ',');
    loaded_policy = policy_file;
end
z_ref = mission(t, P);
m = struct('t', t, 'z', z, 'vz', vz, 'az_prev', az_prev, 'z_ref', z_ref, 'u_prev', u_prev);
obs = double(observation(m));
x = obs(:);
for i = 1:double(P.n_layers)
    x = P.(sprintf('W%d', i)) * x + P.(sprintf('b%d', i));
    switch P.acts{i}
        case 'relu', x = max(x, 0);
        case 'tanh', x = tanh(x);
    end
end
if isfield(P, 'pre_output_low')
    x = min(max(x, P.pre_output_low), P.pre_output_high);           % SAC gSDE mean clip
end
lo = P.action_low(:);
hi = P.action_high(:);
switch P.output
    case 'discrete'                                               % DQN
        [~, action_index] = max(x);
        x = P.action_values(action_index);
    case 'tanh_scale', x = lo + 0.5 * (tanh(x) + 1) .* (hi - lo);   % SAC
    case 'scale',      x = lo + 0.5 * (x + 1) .* (hi - lo);         % TD3 (tanh in the network)
    otherwise,         x = min(max(x, lo), hi);                     % PPO
end
u = x(1);
end

function obs = observation(m)
% Same three inputs as DroneEnv._obs(): altitude error, velocity, acceleration.
obs = [m.z - m.z_ref; m.vz; m.az_prev];
end

function z_ref = mission(t, P)
% Use the selected Python mission when its samples accompany the policy.
if isfield(P, 'mission_t') && isfield(P, 'mission_z_ref')
    z_ref = interp1(P.mission_t(:), P.mission_z_ref(:), t, 'previous', 'extrap');
    return
end
% Default Python mission for policies exported without reference samples.
if t < 2.0
    z_ref = 0.0;
elseif t < 6.0
    z_ref = 6.0;
else
    z_ref = -0.3;
end
end
