% module de génération d'une base de données pour apprendre tout type de
% mvt. Module abandonné car génère des sorties trop aléatoires avec aucune 
% situation stable.




clc; clear; close all;
generator_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(fileparts(generator_dir));
plant_dir = fullfile(project_root, 'src', 'drone', 'plant');
generated_dir = fullfile(project_root, 'data', 'generated_data');
addpath(plant_dir);
if ~isfolder(generated_dir); mkdir(generated_dir); end;
random_init; 

model_name = 'EINT_system_simulation';
load_system(fullfile(plant_dir, [model_name '.slx']));
N_series = 4000;               
T_segment = 8.0;         
Ts = 0.05;               
t = (0:Ts:T_segment)';
N_steps = length(t);

% Paramètres des 4 quarts
U_MIN = -3;              % Tension minimum autorisée (Quarts 1, 2, 3)
U_MAX = 3;               % Tension maximum autorisée (Quarts 1, 2, 3)
u0_min = -2.95;          % Plage basse pour le u0 commun (Quart 4)
u0_max = 2.95;           % Plage haute pour le u0 commun (Quart 4)
amp_small = 0.05;        % Micro-perturbations autour de u0 (Quart 4)
hold_steps = round(0.5 / Ts); 

file_in  = fullfile(generated_dir, 'db2_4quarts_input.csv');
file_out = fullfile(generated_dir, 'db2_4quarts_output.csv');

fid = fopen(file_in, 'w');  fclose(fid);
fid = fopen(file_out, 'w'); fclose(fid);


for s = 1:N_series
    if mod(s, 10) == 0
        fprintf('Génération DB 4 Quarts - Série %4d/%4d...\n', s, N_series);
    end
    
    % Détermination du quart actuel (1, 2, 3 ou 4)
    quart = mod(s-1, 4) + 1;
    u_matrix = zeros(N_steps, 4);
    
    switch quart
        case 1 % 4 entrées égales (APRBS absolu)
            sig = generate_aprbs_abs(N_steps, hold_steps, U_MIN, U_MAX);
            u_matrix = [sig, sig, sig, sig];
            
        case 2 % Égales 2 à 2 (2 APRBS absolus indépendants)
            sig1 = generate_aprbs_abs(N_steps, hold_steps, U_MIN, U_MAX);
            sig2 = generate_aprbs_abs(N_steps, hold_steps, U_MIN, U_MAX);
            idx = randperm(4); 
            u_matrix(:, idx(1)) = sig1;
            u_matrix(:, idx(2)) = sig1;
            u_matrix(:, idx(3)) = sig2;
            u_matrix(:, idx(4)) = sig2;
            
        case 3 % 3 identiques, 1 solo (2 APRBS absolus indépendants)
            sig_maj = generate_aprbs_abs(N_steps, hold_steps, U_MIN, U_MAX);
            sig_solo = generate_aprbs_abs(N_steps, hold_steps, U_MIN, U_MAX);
            u_matrix = [sig_maj, sig_maj, sig_maj, sig_maj];
            idx_solo = randi(4); 
            u_matrix(:, idx_solo) = sig_solo;
            
        case 4 % u0 commun + micro-perturbations indépendantes
            u0 = u0_min + (u0_max - u0_min) * rand(); % Tirage du u0 d'équilibre
            sig1 = generate_aprbs_pert(N_steps, hold_steps, u0, amp_small);
            sig2 = generate_aprbs_pert(N_steps, hold_steps, u0, amp_small);
            sig3 = generate_aprbs_pert(N_steps, hold_steps, u0, amp_small);
            sig4 = generate_aprbs_pert(N_steps, hold_steps, u0, amp_small);
            u_matrix = [sig1, sig2, sig3, sig4];
    end
    
    % Envoi des 4 signaux individuels au Workspace
    assignin('base', 'sim_u1', timeseries(u_matrix(:, 1), t));
    assignin('base', 'sim_u2', timeseries(u_matrix(:, 2), t));
    assignin('base', 'sim_u3', timeseries(u_matrix(:, 3), t));
    assignin('base', 'sim_u4', timeseries(u_matrix(:, 4), t));
    
    % Exécution Simulink
    out = sim(model_name, 'StopTime', num2str(T_segment), 'FixedStep', num2str(Ts));
    
    % Extraction robuste
    if isa(out.simout, 'timeseries')
        all_outputs = out.simout.Data;
    elseif isstruct(out.simout) && isfield(out.simout, 'signals')
        all_outputs = out.simout.signals.values;
    elseif isa(out.simout, 'Simulink.SimulationData.Dataset')
        all_outputs = out.simout{1}.Values.Data;
    else
        all_outputs = out.simout; 
    end
    
    % Alignement des tailles en cas d'arrêt prématuré de Simulink
    N_min = min(N_steps, size(all_outputs, 1));
    
    % Écriture dans le CSV des entrées
    fid_in = fopen(file_in, 'a'); 
    fprintf(fid_in, '%f,', reshape(u_matrix(1:N_min,:)', 1, [])); 
    fprintf(fid_in, '\n'); 
    fclose(fid_in);
    
    % Écriture dans le CSV des sorties
    fid_out = fopen(file_out, 'a'); 
    fprintf(fid_out, '%f,', reshape(all_outputs(1:N_min,1:18)', 1, [])); 
    fprintf(fid_out, '\n'); 
    fclose(fid_out);
    
end

disp('Génération DB 4 Quarts terminée avec succès !');


% APRBS Absolu (Quarts 1, 2, 3)

function sig = generate_aprbs_abs(N, hold_steps, u_min, u_max)
    sig = zeros(N, 1);
    current_val = u_min + (u_max - u_min) * rand(); 
    for k = 1:N
        if mod(k-1, hold_steps) == 0
            current_val = u_min + (u_max - u_min) * rand();
        end
        sig(k) = current_val;
    end
end


% APRBS Perturbation (Quart 4)

function sig = generate_aprbs_pert(N, hold_steps, u0, amp)
    sig = zeros(N, 1);
    current_val = u0 + amp * (2*rand() - 1); 
    for k = 1:N
        if mod(k-1, hold_steps) == 0
            current_val = u0 + amp * (2*rand() - 1);
        end
        sig(k) = current_val;
    end
end