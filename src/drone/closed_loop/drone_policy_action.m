function u = drone_policy_action(P, observation)
%DRONE_POLICY_ACTION Reproduce the exported actor in physical command units.
% Observation scaling precedes the network. Action scaling follows clipping.
x = double(observation(:)) ./ P.observation_scale(:);
assert(numel(x) == size(P.W1, 2), 'Policy observation dimension mismatch.');
activations = strsplit(strtrim(P.activations), ',');
for k = 1:double(P.n_layers)
    x = P.(sprintf('W%d', k)) * x + P.(sprintf('b%d', k));
    switch strtrim(activations{k})
        case 'relu'
            x = max(x, 0);
        case 'tanh'
            x = tanh(x);
        case 'none'
        otherwise
            error('Unsupported policy activation: %s', activations{k});
    end
end
% SAC with gSDE clips its mean before applying the final tanh.
if isfield(P, 'pre_output_low')
    x = min(max(x, P.pre_output_low), P.pre_output_high);
end
lo = P.action_low(:);
hi = P.action_high(:);
switch strtrim(P.output)
    case 'discrete'
        [~, index] = max(x);
        x = P.action_values(index);
    case 'tanh_scale'
        x = lo + 0.5 * (tanh(x) + 1) .* (hi - lo);
    case 'clip'
        x = min(max(x, lo), hi);
    otherwise
        error('Unsupported policy output: %s', P.output);
end
% DQN's action grid already uses physical units; continuous actors need scaling.
if isfield(P, 'action_scale')
    x = P.action_offset(:) + P.action_scale(:) .* x;
end
u = x;
end
