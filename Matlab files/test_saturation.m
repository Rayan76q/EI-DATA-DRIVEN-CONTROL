% Script de test des Saturations 



clc; clear; close all;

%% 1. Initialisation
disp('Initialisation des variables du workspace...');
random_init; 

if ~exist('Ts', 'var')
    Ts = 0.05; % Période d'échantillonnage par défaut[cite: 2]
end

T_sim = 25; % Une simulation courte de 25 secondes suffit
t = (0:Ts:T_sim)';

%% 2. Création d'un signal de test en escalier (Staircase)
% Ce signal va forcer le système à passer par tous les extrêmes
u_start = 0; 
u_end = -5;    

u_input = u_start + (u_end - u_start) * (t / T_sim);

% Création de l'objet timeseries attendu par le bloc "From Workspace"
sim_input = timeseries(u_input, t);
assignin('base', 'sim_input', sim_input);

%% 3. Lancement rapide de la simulation
disp('Lancement de la simulation (Mode Accelerator)...');
% Le mode accelerator permet de gagner beaucoup de temps de calcul
out = sim('EINT_system_simulation', 'StopTime', num2str(T_sim), ...
    'FixedStep', num2str(Ts), 'SimulationMode', 'accelerator');
disp('Simulation terminée.');

%% 4. Extraction robuste des données
% Récupération de la matrice de sortie peu importe la configuration du bloc
if isa(out.simout, 'timeseries')
    all_outputs = out.simout.Data;
elseif isstruct(out.simout) && isfield(out.simout, 'signals')
    all_outputs = out.simout.signals.values;
else
    all_outputs = out.simout; 
end

% Indices de l'axe Z (à vérifier selon votre modèle)
idx_vz = 6;  
idx_az = 9;

vz = all_outputs(1:length(t), idx_vz);
az = all_outputs(1:length(t), idx_az);

%% 5. Détection des saturations sur a_z
az_max = max(az);
az_min = min(az);
tol = 1e-2; % Tolérance de détection

sat_haute = abs(az - az_max) < tol;
sat_basse = abs(az - az_min) < tol;

%% 6. Affichage graphique
figure('Name', 'Analyse des Saturations du Modèle', 'Position', [100, 100, 900, 700]);

% Entrée u
subplot(3, 1, 1);
stairs(t, u_input, 'k', 'LineWidth', 1.5);
title('Signal d''entrée en escalier (Commande u)', 'FontWeight', 'bold');
ylabel('Amplitude'); grid on;

% Vitesse vz
subplot(3, 1, 2);
plot(t, vz, 'g', 'LineWidth', 1.5);
title('Vitesse Verticale (v_z)', 'FontWeight', 'bold');
ylabel('m/s'); grid on;

% Accélération az avec marqueurs de saturation
subplot(3, 1, 3);
hold on;
plot(t, az, 'b', 'LineWidth', 1.5);

% Affichage des points rouges là où ça sature
plot(t(sat_haute), az(sat_haute), 'r.', 'MarkerSize', 12);
plot(t(sat_basse), az(sat_basse), 'r.', 'MarkerSize', 12);
hold off;

title('Accélération (a_z) - Points de saturation en ROUGE', 'FontWeight', 'bold', 'Color', 'r');
xlabel('Temps (s)'); ylabel('m/s^2'); grid on;
legend('a_z simulé', 'Saturation', 'Location', 'best');

% Ajout des lignes limites
yline(az_max, 'r--', sprintf('Max = %.2f', az_max));
yline(az_min, 'r--', sprintf('Min = %.2f', az_min));