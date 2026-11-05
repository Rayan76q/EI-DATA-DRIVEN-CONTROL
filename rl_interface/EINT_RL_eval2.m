%% Track B: Evaluation sans contrôle x, y (Altitude + Attitude 360°)
clear; close all; clc;
here = fileparts(mfilename('fullpath'));
cd(here);
addpath(here, fullfile(here, '..', 'Matlab files'));

% Paramètres
random_init;                   % graines de bruit
Ts = 0.05;
T_end = 12;                    % 12 secondes pour la mission
g_noise = 0.1;                 % bruit

clear rl_controller2
% 1. Simulation du nouveau schéma
out = sim('EINT_RL_closed_loop_2');
L = out.rl_log;

% 2. Extraction exacte des 18 signaux de votre Mux
t        = L(:, 1);
z        = L(:, 2);
vz       = L(:, 3);
az       = L(:, 4);
phi      = L(:, 5);
theta    = L(:, 6);
psi      = L(:, 7);
wx       = L(:, 8);
wdotx    = L(:, 9);
wy       = L(:, 10);
wz       = L(:, 11);
wdoty    = L(:, 12);
wdotz    = L(:, 13);
u        = L(:, 14:17);        % Les 4 moteurs [u1, u2, u3, u4]
z_ref    = L(:, 18);

% 3. Tracés d'évaluation
figure('Name', 'Test Altitude + 360°');

subplot(3, 1, 1);
plot(t, z_ref, 'k--', t, z, 'b', 'LineWidth', 1.2);
ylabel('z [m]'); legend('z_{ref}', 'z'); grid on;
title(sprintf('Suivi altitude (RMSE: %.2f m)', sqrt(mean((z_ref - z).^2))));

subplot(3, 1, 2);
plot(t, rad2deg(psi), 'r', 'LineWidth', 1.2);
ylabel('\psi [deg]'); title('Rotation en lacet (360°)'); grid on;

subplot(3, 1, 3);
plot(t, u);
ylabel('u_1..u_4'); xlabel('Temps [s]'); grid on;
legend('u_1', 'u_2', 'u_3', 'u_4');