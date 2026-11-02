function [u, z_ref, obs] = rl_controller(t, z, vz, az_prev, u_prev)
%RL_CONTROLLER  The trained policy, called at each step by EINT_RL_closed_loop.slx.
%   Write mission() and observation() at the end of this file. policy.mat is written by
%   export_for_matlab (export_policy.py).
persistent P
if isempty(P)
    P = load(fullfile(fileparts(mfilename('fullpath')), 'policy.mat'));
    P.acts = strsplit(P.activations, ',');
end
z_ref = mission(t);
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
lo = P.action_low(:);
hi = P.action_high(:);
switch P.output
    case 'tanh_scale', x = lo + 0.5 * (tanh(x) + 1) .* (hi - lo);   % SAC
    case 'scale',      x = lo + 0.5 * (x + 1) .* (hi - lo);         % TD3 (tanh in the network)
    otherwise,         x = min(max(x, lo), hi);                     % PPO
end
u = x(1);
end

function obs = observation(m)
% The observation: the same computation as in your Python environment, in the same order.
%   m: struct of scalars t, z, vz, az_prev, z_ref, u_prev (the measurements of the drone).
%   obs: vector, e.g. obs = [m.z; m.vz];
% A memory (an integral of the error, past commands...) goes in persistent variables: they are
% reset by "clear rl_controller", at the start of each flight.
error('TODO: write observation() at the end of rl_controller.m (the same as in track_b.ipynb)');
end

function z_ref = mission(t)
% The reference altitude at time t [s], the same as in your Python environment: the steps of
% Figure 1 (0 m on the platform, 0.5 m, -0.25 m on the ground), within the 10 s of the flight.
error('TODO: write mission() at the end of rl_controller.m (Figure 1 of the subject)');
end
