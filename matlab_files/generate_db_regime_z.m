
clc; clear; close all;
random_init; 

% Initialisation des flux aléatoires
s_rand = RandStream('mt19937ar', 'Seed', 'shuffle');
RandStream.setGlobalStream(s_rand);
model_name = 'EINT_system_simulation';

% Nombre d'épisodes à ajouter à ta base existante
N_series = 1250;               
T_segment = 8.0;         
Ts = 0.05;               
t = (0:Ts:T_segment)';
N_steps = length(t);

% --- Paramètres Communs ---
Hover_voltage = 0.5;
U_MAX_PHYS = 3.0; % Limite physique des moteurs dans Simulink

% Nouveaux fichiers pour stocker uniquement cet ajout
file_in  = 'db_regime_z_input.csv';
file_out = 'db_regime_z_output.csv';

% Initialisation : on efface les fichiers s'ils existent déjà
fid = fopen(file_in, 'w');  fclose(fid);
fid = fopen(file_out, 'w'); fclose(fid);

% --- Matrice de mixage ---
M = [1,  1,  1,  1; 
     1, -1,  1, -1; 
     1, -1, -1,  1; 
     1,  1, -1, -1];
     
disp('Lancement Génération DB (Régime 5 : Montées fortes et Descentes actives)...');

for s = 1:N_series
    if mod(s, 10) == 0; fprintf('Génération Ajout Série %4d/%4d...\n', s, N_series); end
    
    % --- Génération exclusive du Régime 5 ---
    % Temps de palier plus longs (1.0s à 3.0s) pour laisser le temps d'accélérer
    h_steps = round((1.0 + rand()*2.0) / Ts);
    
    % THRUST ACTIF : 
    % Probabilité 50/50. 
    % Montée : jusqu'à +2.5V (ascension violente)
    % Descente : jusqu'à -2.5V (propulsion vers le bas)
    prob_montee = 0.50; 
    U_thrust = generate_aprbs_biased(N_steps, h_steps, -2.5, Hover_voltage, 2.5, prob_montee); 
    
    % Verrouillé à plat pour isoler la dynamique verticale absolue
    U_roll   = zeros(N_steps, 1); 
    U_pitch  = zeros(N_steps, 1); 
    
    % Lacet isolé pour apprendre la rotation Z
    U_yaw    = generate_aprbs(N_steps, h_steps, -0.2, 0.2); 
    
    % Mixage des commandes vers les 4 moteurs
    U_virt = [U_thrust, U_roll, U_pitch, U_yaw];
    u_raw = (M * U_virt')';
    
    % Sécurité anti-saturation absolue
    u_raw = max(-U_MAX_PHYS, min(U_MAX_PHYS, u_raw));
    
    % Envoi direct des créneaux
    assignin('base', 'sim_u1', timeseries(u_raw(:, 1), t));
    assignin('base', 'sim_u2', timeseries(u_raw(:, 2), t));
    assignin('base', 'sim_u3', timeseries(u_raw(:, 3), t));
    assignin('base', 'sim_u4', timeseries(u_raw(:, 4), t));
    
    % --- Exécution Simulink ---
    out = sim(model_name, 'StopTime', num2str(T_segment), 'FixedStep', num2str(Ts));
    
    if isa(out.simout, 'timeseries'); all_outputs = out.simout.Data;
    elseif isstruct(out.simout); all_outputs = out.simout.signals.values;
    elseif isa(out.simout, 'Simulink.SimulationData.Dataset'); all_outputs = out.simout{1}.Values.Data;
    else; all_outputs = out.simout; end
    
    N_min = min(N_steps, size(all_outputs, 1));
    
    % --- Écriture CSV (Données brutes) ---
    fid_in = fopen(file_in, 'a'); 
    fprintf(fid_in, '%f,', reshape(u_raw(1:N_min,:)', 1, [])); 
    fprintf(fid_in, '\n'); fclose(fid_in);
    
    fid_out = fopen(file_out, 'a'); 
    fprintf(fid_out, '%f,', reshape(all_outputs(1:N_min,1:18)', 1, [])); 
    fprintf(fid_out, '\n'); fclose(fid_out);
end
disp('Génération du Régime 5 terminée avec succès !');

% ========================================================================
% FONCTIONS DE GENERATION DE SIGNAUX
% ========================================================================

function sig = generate_aprbs(N, hold_steps, u_min, u_max)
    sig = zeros(N, 1);
    current_val = u_min + (u_max - u_min) * rand(); 
    for k = 1:N
        if mod(k-1, hold_steps) == 0
            current_val = u_min + (u_max - u_min) * rand();
        end
        sig(k) = current_val;
    end
end

% APRBS Biaisé avec gestion de la tension négative (Reverse Thrust)
function sig = generate_aprbs_biased(N, hold_steps, u_min, u_hover, u_max, prob_up)
    sig = zeros(N, 1);
    
    if rand() < prob_up
        current_val = u_hover + (u_max - u_hover) * rand(); 
    else
        current_val = u_min + (u_hover - u_min) * rand();   
    end
    
    for k = 1:N
        if mod(k-1, hold_steps) == 0
            if rand() < prob_up
                current_val = u_hover + (u_max - u_hover) * rand();
            else
                current_val = u_min + (u_hover - u_min) * rand();
            end
        end
        sig(k) = current_val;
    end
end