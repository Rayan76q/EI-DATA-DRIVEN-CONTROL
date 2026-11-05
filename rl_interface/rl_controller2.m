function [u1, u2, u3, u4, z_ref, obs] = rl_controller2(t, z, vz, az_prev, ...
    phi, theta, psi, wx, wy, wz, ...
    wdotx_prev, wdoty_prev, wdotz_prev, ...
    u1_prev, u2_prev, u3_prev, u4_prev)
%RL_CONTROLLER  Contrôleur RL multi-rotors (4 sorties, 17 mesures en entrée)
%   Appelé à chaque pas Ts = 0.05s par le bloc 'policy' de Simulink.

persistent P
if isempty(P)
    P = load(fullfile(fileparts(mfilename('fullpath')), 'policy.mat'));
    P.acts = strsplit(P.activations, ',');
end

% 1. Calcul de la consigne de mission
[z_ref,psi_ref] = mission(t);

% 2. Regroupement des mesures
m = struct('t', t, 'z', z, 'vz', vz, 'az_prev', az_prev, ...
    'phi', phi, 'theta', theta, 'psi', psi, ...
    'wx', wx, 'wy', wy, 'wz', wz, ...
    'wdotx_prev', wdotx_prev, 'wdoty_prev', wdoty_prev, 'wdotz_prev', wdotz_prev, ...
    'u1_prev', u1_prev, 'u2_prev', u2_prev, 'u3_prev', u3_prev, 'u4_prev', u4_prev, ...
    'z_ref', z_ref, 'psi_ref', psi_ref);

% 3. Construction du vecteur d'observation pour le réseau
obs = double(observation(m));
x = obs(:);

% 4. Inférence du réseau de neurones (Forward Pass)
for i = 1:double(P.n_layers)
    x = P.(sprintf('W%d', i)) * x + P.(sprintf('b%d', i));
    switch P.acts{i}
        case 'relu', x = max(x, 0);
        case 'tanh', x = tanh(x);
    end
end

% 5. Saturation / Bornage des 4 commandes
lo = P.action_low(:);
hi = P.action_high(:);
switch P.output
    case 'tanh_scale', x = lo + 0.5 * (tanh(x) + 1) .* (hi - lo);   % SAC
    case 'scale',      x = lo + 0.5 * (x + 1) .* (hi - lo);         % TD3
    otherwise,         x = min(max(x, lo), hi);                     % PPO
end

% 6. Séparation des 4 commandes moteurs
u1 = x(1);
u2 = x(2);
u3 = x(3);
u4 = x(4);
end

% =========================================================================
% Fonctions locales
% =========================================================================
function obs = observation(m)
% Erreur d'angle psi normalisée
err_psi = m.psi - m.psi_ref;

obs = [m.z - m.z_ref; ...
    m.vz; ...
    m.az_prev; ...
    m.phi; ...              % roullis (doit rester 0)
    m.theta; ...            % tangage (doit rester 0)
    err_psi; ...            % erreur de cap 360°
    m.wx; m.wy; m.wz; ...
    m.wdotx_prev; m.wdoty_prev; m.wdotz_prev; ...
    m.u1_prev; m.u2_prev; m.u3_prev; m.u4_prev];
end
function [z_ref, psi_ref] = mission(t)
% Mission 12s : Décollage -> Montée 1m -> Tour 360° -> Atterrissage
if t < 2.0
    z_ref = 0.0;
    psi_ref = 0.0;
elseif t < 4.0
    z_ref = 1.0;
    psi_ref = 0.0;
elseif t < 8.0
    z_ref = 1.0;
    % Rotation progressive de 0 à 2*pi radians (360°)
    psi_ref = 2.0 * pi * (t - 4.0) / (8.0 - 4.0);
else
    z_ref = -0.3;          % Atterrissage
    psi_ref = 2.0 * pi;    % Reste à 360°
end
end