% script de test de la configuration en X
% avec un premier test, on connaissait le thrust et le yaw
% ccl : sur les 24 configs, on
% obtient aucune config découplé. sys couplé plus compliqué

%% Initialize random number stream
s = RandStream('mt19937ar','Seed','shuffle');
RandStream.setGlobalStream(s);
Ts = 0.05; % Maybe not the right value...
g_noise = 0.01; % Global noise gain

% Generate random seeds for each state variable to ensure variability in simulations for data generation
seed_x = randi(s, 10000);      seed_y = randi(s, 10000);      seed_z = randi(s, 10000);
seed_vx = randi(s, 10000);     seed_vy = randi(s, 10000);     seed_vz = randi(s, 10000);
seed_ax = randi(s, 10000);     seed_ay = randi(s, 10000);     seed_az = randi(s, 10000);
seed_phi = randi(s, 10000);    seed_psi = randi(s, 10000);    seed_theta = randi(s, 10000);
seed_wx = randi(s, 10000);     seed_wy = randi(s, 10000);     seed_wz = randi(s, 10000);
seed_w_dot_x = randi(s, 10000);seed_w_dot_y = randi(s, 10000);seed_w_dot_z = randi(s, 10000);

model_name = 'EINT_system_simulation';
Ts = 0.05;              
T_fin = 16.0; % Simulation allongée à 16s pour intégrer les pauses
t = (0:Ts:T_fin)';      
N_steps = length(t);

% Initialisation des commandes virtuelles 
U_thrust = zeros(N_steps, 1);
U_roll   = zeros(N_steps, 1);
U_pitch  = zeros(N_steps, 1);
U_yaw    = zeros(N_steps, 1);


% CONSTRUCTION DU SCÉNARIO DE TEST 

% 1. Hover (Stationnaire)
Hover_voltage = 0.5; 
U_thrust(:) = Hover_voltage; 

% 2. TEST DE LA POUSSÉE (2s à 4s)
U_thrust(t >= 2 & t < 3) = Hover_voltage + 0.5;
U_thrust(t >= 3 & t < 4) = Hover_voltage - 0.5;

% 3. TEST DU ROULIS - Roll (5s à 7s)
U_roll(t >= 5 & t < 6) = 0.2; 
U_roll(t >= 6 & t < 7) = -0.2;

% 4. TEST DU TANGAGE - Pitch (8s à 10s)
U_pitch(t >= 8 & t < 9) = 0.2;
U_pitch(t >= 9 & t < 10) = -0.2;

% 5. TEST DU LACET - Yaw (11s à 13s)
U_yaw(t >= 11 & t < 12) = 0.2;
U_yaw(t >= 12 & t < 13) = -0.2;

% LOGIQUE DES 24 CONFIGURATIONS

% Signes FIXES selon tes instructions :
% u1 : +Thrust, +Yaw
% u2 : +Thrust, -Yaw
% u3 : +Thrust, +Yaw
% u4 : +Thrust, -Yaw
sign_T = [1, 1, 1, 1];
sign_Y = [1, -1, 1, -1];

% Les 4 paires possibles pour (Pitch, Roll)
% 1:(+,+) | 2:(+,-) | 3:(-,+) | 4:(-,-)
paires_PR = [
    1,  1;  
    1, -1;  
   -1,  1;  
   -1, -1   
];

% Génère les 24 ordres possibles (matrice 24x4)
permutations = perms([1 2 3 4]);

% Noms des sorties
noms_sorties = {'x', 'y', 'z', ...
    'v_x', 'v_y', 'v_z', ...
    'ax', 'ay', 'az', ...
    '\phi (Roulis)', '\theta (Tangage)', '\psi (Lacet)', ...
    'wx', 'wy', 'wz', ...
    'dwx/dt', 'dwy/dt', 'dwz/dt'};

disp('Lancement des 24 simulations...');

for config = 1:24
    ordre = permutations(config, :);
    u_matrix = zeros(N_steps, 4);
    titre_signes = '';
    
    % Construction de la matrice pour cette configuration
    for m = 1:4
        idx_paire = ordre(m);
        sign_P = paires_PR(idx_paire, 1);
        sign_R = paires_PR(idx_paire, 2);
        
        u_matrix(:, m) = sign_T(m) * U_thrust ...
                       + sign_P * U_pitch ...
                       + sign_R * U_roll ...
                       + sign_Y(m) * U_yaw;
                       
        % Construction du titre pour savoir quels signes sont sur quel moteur
        char_P = '+'; if sign_P < 0; char_P = '-'; end
        char_R = '+'; if sign_R < 0; char_R = '-'; end
        titre_signes = [titre_signes, sprintf('u%d(%c%c) ', m, char_P, char_R)];
    end
    
    % Sécurité anti-saturation
    u_matrix = max(-3, min(3, u_matrix));
    
    assignin('base', 'sim_u1', timeseries(u_matrix(:,1), t));
    assignin('base', 'sim_u2', timeseries(u_matrix(:,2), t));
    assignin('base', 'sim_u3', timeseries(u_matrix(:,3), t));
    assignin('base', 'sim_u4', timeseries(u_matrix(:,4), t));
    
    % Lancement simulation
    out = sim(model_name, 'StopTime', num2str(T_fin), 'FixedStep', num2str(Ts));
    
    % Extraction robuste
    if isa(out.simout, 'timeseries')
        out_t = out.simout.Time;
        all_outputs = out.simout.Data;
    elseif isstruct(out.simout) && isfield(out.simout, 'signals')
        out_t = out.tout;
        all_outputs = out.simout.signals.values;
    elseif isa(out.simout, 'Simulink.SimulationData.Dataset')
        out_t = out.simout{1}.Values.Time;
        all_outputs = out.simout{1}.Values.Data;
    else
        out_t = out.tout;
        all_outputs = out.simout; 
    end
    
    % Affichage
    fig_titre = sprintf('Config %d | %s [Pitch, Roll]', config, titre_signes);
    figure('Name', fig_titre, 'NumberTitle', 'off', 'Units', 'normalized', 'Position', [0 0 1 1]);
    
    for i = 1:18
        subplot(3, 6, i);
        plot(out_t, all_outputs(:, i), 'LineWidth', 1.5, 'Color', '#D95319');
        hold on;
        xline([2 5 8 11], '--r', {'Test Thrust', 'Test Roll', 'Test Pitch', 'Test Yaw'}, 'LabelVerticalAlignment', 'bottom');
        grid on;
        title(noms_sorties{i}, 'FontWeight', 'bold');
        if i > 12; xlabel('Temps (s)'); end
    end
    
    drawnow; % Afficher immédiatement la fenêtre avant la prochaine boucle
end
disp('Les 24 simulations sont terminées !');