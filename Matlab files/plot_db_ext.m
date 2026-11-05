% Script de plot des premières séries générées d'une db

clc; clear; close all;

file_in  = 'db_regime_z_input.csv';
file_out = 'db_regime_z_output.csv';

% --- PARAMÈTRES D'AFFICHAGE ---
N_series_a_afficher = 5;   % Nombre de séries (épisodes) à superposer
Ts = 0.05;                 % Période d'échantillonnage
fenetres_separees   = false; % 'false' = 1 grand dashboard (22 sous-graphiques)
                             % 'true'  = 22 fenêtres pop-up indépendantes

%% 1. Lecture ciblée des N premières lignes
disp('Lecture des séries...');
fid_in  = fopen(file_in, 'r');
fid_out = fopen(file_out, 'r');

if fid_in == -1 || fid_out == -1
    error('Impossible d''ouvrir les fichiers CSV. Vérifiez leur nom et présence.');
end

inputs_cell  = cell(N_series_a_afficher, 1);
outputs_cell = cell(N_series_a_afficher, 1);
series_lues  = 0;

for s = 1:N_series_a_afficher
    line_in  = fgetl(fid_in);
    line_out = fgetl(fid_out);
    
    if ~ischar(line_in) || ~ischar(line_out)
        break; % Fin de fichier atteinte
    end
    
    % Conversion textuelle -> vecteurs numériques
    val_in  = str2double(strsplit(line_in, ','));
    val_out = str2double(strsplit(line_out, ','));
    
    val_in(isnan(val_in))   = [];
    val_out(isnan(val_out)) = [];
    
    % Dégroupage des signaux :
    % Entrées : 4 canaux entrelacés
    N_steps_in = floor(length(val_in) / 4);
    inputs_cell{s} = reshape(val_in(1:N_steps_in*4), 4, N_steps_in)'; % [N_steps x 4]
    
    % Sorties : 18 canaux entrelacés
    N_steps_out = floor(length(val_out) / 18);
    outputs_cell{s} = reshape(val_out(1:N_steps_out*18), 18, N_steps_out)'; % [N_steps x 18]
    
    series_lues = series_lues + 1;
end

fclose(fid_in);
fclose(fid_out);

fprintf('%d séries chargées avec succès.\n', series_lues);

%% 2. Noms et étiquettes des 22 signaux
labels = cell(22, 1);
for i = 1:4
    labels{i} = sprintf('Entrée u_%d', i);
end
for j = 1:18
    switch j
        case 3,  spec = ' (Z - Position)';
        case 6,  spec = ' (Vz - Vitesse)';
        case 9,  spec = ' (Az - Accélération)';
        otherwise, spec = '';
    end
    labels{4 + j} = sprintf('Sortie y_{%d}%s', j, spec);
end

%% 3. Tracé des 22 signaux
couleurs = lines(series_lues);

if ~fenetres_separees
    % Dashboard unique organisé en grille 6x4 (24 emplacements pour 22 signaux)
    figure('Name', 'Dashboard 22 Signaux (4 In / 18 Out)', 'Position', [30, 30, 1600, 950]);
end

for k = 1:22
    if fenetres_separees
        figure('Name', labels{k}, 'Position', [100 + mod(k,5)*40, 100 + mod(k,4)*40, 600, 350]);
    else
        subplot(6, 4, k);
    end
    
    hold on;
    for s = 1:series_lues
        if k <= 4
            signal = inputs_cell{s}(:, k);
        else
            signal = outputs_cell{s}(:, k - 4);
        end
        t = (0:length(signal)-1) * Ts;
        
        if k <= 4
            stairs(t, signal, 'Color', couleurs(s, :), 'LineWidth', 1.0, ...
                   'DisplayName', sprintf('Série %d', s));
        else
            plot(t, signal, 'Color', couleurs(s, :), 'LineWidth', 1.0, ...
                 'DisplayName', sprintf('Série %d', s));
        end
    end
    hold off;
    
    title(labels{k}, 'FontSize', 9, 'FontWeight', 'bold');
    grid on;
    if k >= 19
        xlabel('Temps (s)', 'FontSize', 8);
    end
end

if ~fenetres_separees
    sgtitle(sprintf('Visualisation des %d premières séries (4 Entrées en escalier, 18 Sorties simulées)', series_lues), ...
            'FontSize', 13, 'FontWeight', 'bold');
end