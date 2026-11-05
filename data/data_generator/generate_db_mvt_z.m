% module de génération d'une base de données pour apprendre un mvt selon
% l'axe z uniquement

clc; clear; close all;
generator_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(fileparts(generator_dir));
plant_dir = fullfile(project_root, 'src', 'drone', 'plant');
generated_dir = fullfile(project_root, 'data', 'generated_data');
addpath(plant_dir);
if ~isfolder(generated_dir); mkdir(generated_dir); end;
random_init;
load_system(fullfile(plant_dir, 'EINT_system_simulation.slx')); 

%% Paramètres de génération
N_series = 20;               % Nombre de séries (épisodes) que tu veux générer
T_sim = 50;                  % Durée totale de chaque simulation (secondes)
if ~exist('Ts', 'var')
    Ts = 0.05;               % Période d'échantillonnage par défaut
end
t = (0:Ts:T_sim)';           % Vecteur temps

% Fichiers de sortie uniques
fid_in = fopen(fullfile(generated_dir, 'rnn_input_z_trap.csv'), 'w');
fid_out = fopen(fullfile(generated_dir, 'rnn_output_z_trap.csv'), 'w');

disp(['Génération de ', num2str(N_series), ' séries de simulation...']);

for s = 1:N_series
    fprintf('-> Simulation %d/%d en cours...\n', s, N_series);

    % Génération du signal d'entrée (Échelons aléatoires)
    u_base = 0; 
    u_amp = 3;  
    min_step_duration = 0; 
    max_step_duration = 2; 

    u_input = zeros(length(t), 1);
    current_time = 0;
    while current_time < T_sim
        step_duration = min_step_duration + (max_step_duration - min_step_duration) * rand();
        step_value = u_base + (rand() * 2 - 1) * u_amp;
        idx_start = max(1, round(current_time / Ts) + 1);
        idx_end = min(length(t), round((current_time + step_duration) / Ts) + 1);
        u_input(idx_start:idx_end) = step_value;
        current_time = current_time + step_duration;
    end

    % Envoi vers Simulink
    sim_input = timeseries(u_input, t);
    assignin('base', 'sim_input', sim_input);

    % Exécution de la simulation
    out = sim('EINT_system_simulation', 'StopTime', num2str(T_sim), ...
        'FixedStep', num2str(Ts), 'SimulationMode', 'accelerator');

    % Extraction des données
    if isa(out.simout, 'timeseries')
        all_outputs = out.simout.Data;
    elseif isstruct(out.simout) && isfield(out.simout, 'signals')
        all_outputs = out.simout.signals.values;
    elseif isa(out.simout, 'Simulink.SimulationData.Dataset')
        all_outputs = out.simout{1}.Values.Data;
    else
        all_outputs = out.simout; 
    end

    idx_z  = 3;  % Position Z
    idx_vz = 6;  % Vitesse Z
    idx_az = 9;  % Accélération Z
    z_data = all_outputs(:, [idx_z, idx_vz, idx_az]);

    N_min = min(length(u_input), size(z_data, 1));
    in_final = u_input(1:N_min)';
    out_final = z_data(1:N_min, :)'; % Format 3 x N_min

    % Écriture dans les fichiers CSV (1 ligne = 1 série)
    % --- Écriture de l'entrée ---
    fprintf(fid_in, '%f,', in_final);
    fprintf(fid_in, '\n'); % Retour à la ligne pour la série suivante

    % --- Écriture de la sortie ---
    out_flat = reshape(out_final, 1, []); 
    fprintf(fid_out, '%f,', out_flat);
    fprintf(fid_out, '\n'); % Retour à la ligne pour la série suivante
end

% Fermeture des fichiers
fclose(fid_in);
fclose(fid_out);

