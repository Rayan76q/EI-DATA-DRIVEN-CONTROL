% module de génération d'une base de données pour apprendre tout type de
% mvt. Utilisation de CL de
% tensions pour mimer des mouvements réels de roll pitch yaw thrust.

clc; clear; close all;
generator_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(fileparts(generator_dir));
plant_dir = fullfile(project_root, 'src', 'drone', 'plant');
generated_dir = fullfile(project_root, 'data', 'generated_data');
addpath(plant_dir);
if ~isfolder(generated_dir); mkdir(generated_dir); end;
random_init; 

% Initialisation des flux aléatoires
s_rand = RandStream('mt19937ar', 'Seed', 'shuffle');
RandStream.setGlobalStream(s_rand);
model_name = 'EINT_system_simulation';
load_system(fullfile(plant_dir, [model_name '.slx']));
N_series = 50;               
T_segment = 8.0;         
Ts = 0.05;               
t = (0:Ts:T_segment)';
N_steps = length(t);

% Paramètres Communs 
Hover_voltage = 0.5;
U_MAX_PHYS = 3.0;
file_in  = fullfile(generated_dir, 'db_5regimes_input.csv');
file_out = fullfile(generated_dir, 'db_5regimes_output.csv');
fid = fopen(file_in, 'w');  fclose(fid);
fid = fopen(file_out, 'w'); fclose(fid);

% Matrice de mixage 
M = [1,  1,  1,  1; 
     1, -1,  1, -1; 
     1, -1, -1,  1; 
     1,  1, -1, -1];

for s = 1:N_series
    if mod(s, 10) == 0; fprintf('Génération Série %4d/%4d...\n', s, N_series); end
    
 
    regime = mod(s-1, 5) + 1;
    
    U_thrust = zeros(N_steps, 1);
    U_roll   = zeros(N_steps, 1);
    U_pitch  = zeros(N_steps, 1);
    U_yaw    = zeros(N_steps, 1);
    
    switch regime
        case 1 % Régime 1 : Micro-corrections (Palier: 0.5s à 1.5s)
            h_steps = round((0.5 + rand()*2.5) / Ts);
            U_thrust = generate_aprbs(N_steps, h_steps, 0.45, 0.55);
            U_roll   = generate_aprbs(N_steps, h_steps, -0.05, 0.05);
            U_pitch  = generate_aprbs(N_steps, h_steps, -0.05, 0.05);
            U_yaw    = generate_aprbs(N_steps, h_steps, -0.05, 0.05);
            
        case 2 % Régime 2 : Agilité Classique (Mouvements amples mais gérables)
            U_thrust = Hover_voltage + generate_doublet(N_steps, Ts, 0, 0.1);
            U_roll   = generate_doublet(N_steps, Ts, 0, 0.1);
            U_pitch  = generate_doublet(N_steps, Ts, 0, 0.1);
            U_yaw    = generate_doublet(N_steps, Ts, 0, 0.1);
            
        case 3 % Régime 3 : Chaos (Asynchrone)
            h_t = round((0.5 + rand()*1.5) / Ts);
            h_r = round((0.5 + rand()*1.5) / Ts);
            h_p = round((0.5 + rand()*1.5) / Ts);
            h_y = round((0.5 + rand()*1.5) / Ts);
            
            U_thrust = generate_aprbs(N_steps, h_t, 0.35, 0.65);
            U_roll   = generate_aprbs(N_steps, h_r, -0.05, 0.05);
            U_pitch  = generate_aprbs(N_steps, h_p, -0.05, 0.05);
            U_yaw    = generate_aprbs(N_steps, h_y, -0.05, 0.05);
            
        case 4 % Régime 4 : Impulsions ultra-courtes, amplitudes de 0.1 à 3.0
            % Pour le Thrust, on limite le max à U_MAX_PHYS - Hover_voltage pour ne pas dépasser 3 au total
            U_thrust = Hover_voltage + generate_bang_bang(N_steps, Ts, 0.1, U_MAX_PHYS - Hover_voltage);
            U_roll   = generate_bang_bang(N_steps, Ts, 0.1, U_MAX_PHYS);
            U_pitch  = generate_bang_bang(N_steps, Ts, 0.1, U_MAX_PHYS);
            U_yaw    = generate_bang_bang(N_steps, Ts, 0.1, U_MAX_PHYS);
            
        case 5 % Régime 5 : Ascension pure et Lacet isolé (Découplage pour apprendre à monter)
            h_steps = round((0.5 + rand()*2.0) / Ts);
            U_thrust = generate_aprbs(N_steps, h_steps, 0.2, 1.5); % Plage large pour forcer la montée et la chute
            U_roll   = zeros(N_steps, 1); % Verrouillé à plat pour ne pas dériver
            U_pitch  = zeros(N_steps, 1); % Verrouillé à plat pour ne pas dériver
            U_yaw    = generate_aprbs(N_steps, h_steps, -0.2, 0.2); % Lacet isolé pour apprendre la rotation Z
    end
    
    % Mixage des commandes vers les 4 moteurs
    U_virt = [U_thrust, U_roll, U_pitch, U_yaw];
    u_raw = (M * U_virt')';
    
    % Sécurité anti-saturation absolue
    u_raw = max(-3, min(3, u_raw));
    
    % Envoi direct des créneaux (Simulink se charge du filtrage interne)
    assignin('base', 'sim_u1', timeseries(u_raw(:, 1), t));
    assignin('base', 'sim_u2', timeseries(u_raw(:, 2), t));
    assignin('base', 'sim_u3', timeseries(u_raw(:, 3), t));
    assignin('base', 'sim_u4', timeseries(u_raw(:, 4), t));
    
    % Exécution Simulink 
    out = sim(model_name, 'StopTime', num2str(T_segment), 'FixedStep', num2str(Ts));
    
    if isa(out.simout, 'timeseries'); all_outputs = out.simout.Data;
    elseif isstruct(out.simout); all_outputs = out.simout.signals.values;
    elseif isa(out.simout, 'Simulink.SimulationData.Dataset'); all_outputs = out.simout{1}.Values.Data;
    else; all_outputs = out.simout; end
    
    N_min = min(N_steps, size(all_outputs, 1));
    
    % Écriture CSV (Données brutes, non filtrées) 
    fid_in = fopen(file_in, 'a'); 
    fprintf(fid_in, '%f,', reshape(u_raw(1:N_min,:)', 1, [])); 
    fprintf(fid_in, '\n'); fclose(fid_in);
    
    fid_out = fopen(file_out, 'a'); 
    fprintf(fid_out, '%f,', reshape(all_outputs(1:N_min,1:18)', 1, [])); 
    fprintf(fid_out, '\n'); fclose(fid_out);
end
disp('Génération DB 5 Régimes terminée avec succès !');


% Fonction 1 : APRBS Classique
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

% Fonction 2 : Doublet Classique (Temps d'impulsion : 0.5s à 1.0s)
function sig = generate_doublet(N, Ts, amp_min, amp_max)
    sig = zeros(N, 1);
    idx = 1;
    while idx <= N
        pause_steps = round((0.5 + rand()*1.0) / Ts);
        idx = idx + pause_steps;
        if idx > N; break; end
        
        impulse_steps = round((0.5 + rand()*0.5) / Ts);
        amp = amp_min + rand() * (amp_max - amp_min);
        
        sign_dir = sign(rand() - 0.5);
        if sign_dir == 0; sign_dir = 1; end
        
        end_idx = min(idx + impulse_steps - 1, N);
        sig(idx:end_idx) = sign_dir * amp;
        idx = end_idx + 1;
        
        if idx > N; break; end
        
        end_idx2 = min(idx + impulse_steps - 1, N);
        sig(idx:end_idx2) = -sign_dir * amp;
        idx = end_idx2 + 1;
    end
end

% Fonction 3 : Micro-Doublet Extrême "Bang-Bang" (Temps d'impulsion : 0.05s à 0.15s)
function sig = generate_bang_bang(N, Ts, amp_min, amp_max)
    sig = zeros(N, 1);
    idx = 1;
    while idx <= N
        % Temps de pause plus long pour absorber le choc (1.0s à 2.0s)
        pause_steps = round((1.0 + rand()*1.0) / Ts);
        idx = idx + pause_steps;
        if idx > N; break; end
        
        % Impulsion ultra-courte (0.05s à 0.15s max)
        impulse_steps = round((0.05 + rand()*0.10) / Ts); 
        
        % Tirage aléatoire de l'amplitude extrême
        amp = amp_min + rand() * (amp_max - amp_min); 
        
        sign_dir = sign(rand() - 0.5);
        if sign_dir == 0; sign_dir = 1; end
        
        end_idx = min(idx + impulse_steps - 1, N);
        sig(idx:end_idx) = sign_dir * amp;
        idx = end_idx + 1;
        
        if idx > N; break; end
        
        end_idx2 = min(idx + impulse_steps - 1, N);
        sig(idx:end_idx2) = -sign_dir * amp;
        idx = end_idx2 + 1;
    end
end