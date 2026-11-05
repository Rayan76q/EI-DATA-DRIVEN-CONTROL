function out = EINT_RL_eval(export_dir, noise_gain, results_dir)
%EINT_RL_EVAL Replay a Python export, then fly its policy on the Simulink plant.
%   addpath('src/drone/closed_loop');
%   out = EINT_RL_eval('results/altitude/ppo/my_run', 0.0);
% The export folder must contain policy.mat and traj_model.mat from Python.
% Plant logs and scope plots go to closed_loop/results/<algorithm>/<run>/.
% Pass a third argument to choose a different plant-results folder.
if nargin < 1 || isempty(export_dir)
    error('Pass the export folder containing policy.mat and traj_model.mat.');
end
if nargin < 2
    noise_gain = 0.1;
end
validateattributes(noise_gain, {'numeric'}, {'scalar', 'real', 'finite', 'nonnegative'});
[exists, export_info] = fileattrib(export_dir);
if ~exists || ~export_info.directory
    error('Export folder does not exist: %s', export_dir);
end
export_dir = export_info.Name;
here = fileparts(mfilename('fullpath'));
plant_dir = fullfile(here, '..', 'plant');
addpath(here, plant_dir);

previous_export = getenv('DRONE_POLICY_EXPORT_DIR');
restore_export = onCleanup(@() setenv('DRONE_POLICY_EXPORT_DIR', previous_export));
setenv('DRONE_POLICY_EXPORT_DIR', export_dir);
check_controller(export_dir);
P = load(fullfile(export_dir, 'policy.mat'));
required = {'Ts', 'initial_az', 'g', 'horizon', 'mission_t', 'mission_z_ref'};
for k = 1:numel(required)
    if ~isfield(P, required{k})
        error('Export is missing %s: export again with src/policy/export_policy.py.', required{k});
    end
end
validateattributes(P.Ts, {'numeric'}, {'scalar', 'real', 'finite', 'positive'});
validateattributes(P.horizon, {'numeric'}, {'scalar', 'real', 'finite', 'positive'});
validateattributes(P.initial_az, {'numeric'}, {'scalar', 'real', 'finite'});
if numel(P.action_low) ~= 1 || size(P.W1, 2) ~= 3
    error('This entrypoint requires an altitude policy with 3 observations and 1 action.');
end
if nargin < 3 || isempty(results_dir)
    switch strtrim(P.output)
        case 'discrete', algorithm = 'dqn';
        case 'tanh_scale', algorithm = 'sac';
        case 'scale', algorithm = 'td3';
        otherwise, algorithm = 'ppo';
    end
    [parent_dir, run_name] = fileparts(export_dir);
    if any(strcmp(run_name, {'training', 'matlab'}))
        [~, run_name] = fileparts(parent_dir);
    end
    results_dir = fullfile(here, 'results', algorithm, run_name);
end

random_init;                          % initialize the plant's 18 noise seeds
Ts = double(P.Ts);                    % use the same period and horizon as Python
T_end = double(P.horizon);
g_noise = double(noise_gain);
g = double(P.g);
model_name = 'EINT_RL_closed_loop';
was_loaded = bdIsLoaded(model_name);
load_system(fullfile(here, [model_name '.slx']));
if ~was_loaded
    close_model = onCleanup(@() close_system(model_name, 0));
end
% The diagram has no Scope blocks. Log its existing 18-state output without
% adding blocks or saving changes to either SLX file.
ports = get_param([model_name '/UAV'], 'PortHandles');
state_port = ports.Outport(19);
logging_names = {'DataLogging', 'DataLoggingNameMode', 'DataLoggingName', ...
    'DataLoggingLimitDataPoints', 'DataLoggingDecimateData'};
previous_logging = cell(size(logging_names));
for k = 1:numel(logging_names)
    previous_logging{k} = get_param(state_port, logging_names{k});
end
previous_override = get_param(model_name, 'DataLoggingOverride');
previous_dirty = get_param(model_name, 'Dirty');
restore_logging = onCleanup(@() restore_plant_logging(model_name, state_port, ...
    logging_names, previous_logging, previous_override, previous_dirty));
set_param(state_port, 'DataLogging', 'on', 'DataLoggingNameMode', 'Custom', ...
    'DataLoggingName', 'plant_state', 'DataLoggingLimitDataPoints', 'off', ...
    'DataLoggingDecimateData', 'off');
logging_override = Simulink.SimulationData.ModelLoggingInfo(model_name);
logging_override.LoggingMode = 'LogAllAsSpecifiedInModel';
set_param(model_name, 'DataLoggingOverride', logging_override);
% SimulationInput keeps parameters local and leaves both SLX files unchanged.
sim_input = Simulink.SimulationInput(model_name);
sim_input = sim_input.setModelParameter('SignalLogging', 'on', ...
    'SignalLoggingName', 'plant_log');
sim_input = sim_input.setVariable('Ts', Ts);
sim_input = sim_input.setVariable('T_end', T_end);
sim_input = sim_input.setVariable('g_noise', g_noise);
sim_input = sim_input.setVariable('g', g);
seed_names = {'seed_x', 'seed_y', 'seed_z', 'seed_vx', 'seed_vy', 'seed_vz', ...
    'seed_ax', 'seed_ay', 'seed_az', 'seed_phi', 'seed_psi', 'seed_theta', ...
    'seed_wx', 'seed_wy', 'seed_wz', 'seed_w_dot_x', 'seed_w_dot_y', 'seed_w_dot_z'};
for k = 1:numel(seed_names)
    sim_input = sim_input.setVariable(seed_names{k}, eval(seed_names{k}));
end
% The first Python observation contains initial_az; match the live delay state.
sim_input = sim_input.setBlockParameter([model_name '/Delay_az'], ...
    'InitialCondition', sprintf('%.17g', double(P.initial_az)));
clear rl_controller
out = sim(sim_input);
clear rl_controller
clear restore_logging
L = out.rl_log;                       % t, z, vz, az, u, z_ref
if isa(L, 'timeseries')
    L = L.Data;
elseif isstruct(L)
    L = L.signals.values;
end
t = L(:, 1); z = L(:, 2); u = L(:, 5); z_ref = L(:, 6);
rmse = sqrt(mean((z_ref - z).^2));
plant_logged = out.plant_log.getElement('plant_state');
plant_signal = plant_logged.Values;
plant_t = plant_signal.Time(:);
if plant_signal.IsTimeFirst
    plant_state = reshape(plant_signal.Data, numel(plant_t), []);
else
    plant_state = reshape(plant_signal.Data, [], numel(plant_t))';
end
% UAV_model's all_outputs mux has acceleration before angles. Its individual
% outports use a different order; use this mux order for CSV and plots.
state_names = {'x', 'y', 'z', 'vx', 'vy', 'vz', 'ax', 'ay', 'az', ...
    'phi', 'theta', 'psi', 'wx', 'wy', 'wz', 'w_dot_x', 'w_dot_y', 'w_dot_z'};
if size(plant_state, 2) ~= numel(state_names)
    error('Expected 18 values from UAV/all_outputs; received %d.', size(plant_state, 2));
end
[sample_t, sample_index] = unique(plant_t, 'last');
control_state = interp1(sample_t, plant_state(sample_index, :), t, 'linear');
trajectory = array2table([t, control_state, u, z_ref], ...
    'VariableNames', [{'t'}, state_names, {'u_base', 'z_ref'}]);
if ~isfolder(results_dir)
    mkdir(results_dir);
end
% Keep raw solver-time states as well as the control-time CSV. All of these
% values come from the Simulink plant; traj_model.mat remains the RNN replay.
source = 'Simulink plant: EINT_RL_closed_loop/UAV (UAV_model)';
units = {'m', 'm', 'm', 'm/s', 'm/s', 'm/s', 'm/s^2', 'm/s^2', 'm/s^2', ...
    'rad', 'rad', 'rad', 'rad/s', 'rad/s', 'rad/s', 'rad/s^2', 'rad/s^2', 'rad/s^2'};
save(fullfile(results_dir, 'traj_simulink.mat'), 'out', 'L', 'plant_t', ...
    'plant_state', 'state_names', 'units', 'trajectory', 'Ts', 'T_end', ...
    'g_noise', 'g', 'rmse', 'source', 'export_dir', '-v7.3');
writetable(trajectory, fullfile(results_dir, 'trajectory.csv'));

fig = figure('Name', 'Altitude policy on the Simulink plant');
subplot(2, 1, 1); hold on; grid on;
plot(t, z_ref, '--', t, z, 'LineWidth', 1.2); yline(-0.3, ':');
ylabel('z [m]'); legend('reference', 'z', 'landing platform');
title(sprintf('Noise gain %g: RMSE %.2f m, min z %.2f m', g_noise, rmse, min(z)));
subplot(2, 1, 2); stairs(t, u); grid on; ylabel('u'); xlabel('t [s]');
exportgraphics(fig, fullfile(results_dir, 'simulink_altitude.png'));
savefig(fig, fullfile(results_dir, 'simulink_altitude.fig'));

fig = figure('Name', 'Simulink plant angles');
for k = 1:3
    subplot(3, 1, k); plot(plant_t, plant_state(:, 9 + k)); grid on;
    yline(0, '--'); ylabel([state_names{9 + k} ' [rad]']);
end
xlabel('t [s]');
exportgraphics(fig, fullfile(results_dir, 'scope_angles.png'));
savefig(fig, fullfile(results_dir, 'scope_angles.fig'));

fig = figure('Name', 'Simulink plant states', 'Position', [100, 100, 1100, 800]);
group_names = {'Position [m]', 'Velocity [m/s]', 'Acceleration [m/s^2]', ...
    'Angles [rad]', 'Angular velocity [rad/s]', 'Angular acceleration [rad/s^2]'};
for k = 1:6
    columns = (3 * k - 2):(3 * k);
    subplot(3, 2, k); plot(plant_t, plant_state(:, columns)); grid on;
    ylabel(group_names{k}); xlabel('t [s]'); legend(state_names(columns));
end
exportgraphics(fig, fullfile(results_dir, 'scope_states.png'));
savefig(fig, fullfile(results_dir, 'scope_states.fig'));
fprintf('Simulink plant results saved to %s (RMSE %.3f m).\n', results_dir, rmse);
end


function restore_plant_logging(model_name, state_port, names, values, override, dirty)
% Restore a model that was already open, including its unsaved-change flag.
if ~bdIsLoaded(model_name)
    return
end
for k = 1:numel(names)
    set_param(state_port, names{k}, values{k});
end
set_param(model_name, 'DataLoggingOverride', override, 'Dirty', dirty);
end


function check_controller(export_dir)
% Replays the measurements of traj_model.mat (the Figure 1 mission flown on your model, written by
% export_for_matlab in track_b.ipynb) through rl_controller.m, step by step as in Simulink: the
% mission, the observation and the command must be those of Python.
if ~isfile(fullfile(export_dir, 'policy.mat')) || ~isfile(fullfile(export_dir, 'traj_model.mat'))
    error('Missing policy.mat or traj_model.mat in %s. Export the policy from Python first.', export_dir);
end
T = load(fullfile(export_dir, 'traj_model.mat'));
clear rl_controller
for k = 1:numel(T.t)
    [u, z_ref, obs] = rl_controller(T.t(k), T.z(k), T.vz(k), T.az_prev(k), T.u_prev(k));
    where = sprintf('step %d (t = %.2f s)', k - 1, T.t(k));
    if abs(z_ref - T.z_ref(k)) > 1e-9
        error('%s: z_ref = %g in rl_controller.m, %g in Python: mission() differs.', where, z_ref, T.z_ref(k));
    end
    if numel(obs) ~= size(T.obs, 2)
        error('observation() returns %d values in rl_controller.m, %d in Python.', numel(obs), size(T.obs, 2));
    end
    [e, i] = max(abs(obs(:)' - T.obs(k, :)) ./ max(1, abs(T.obs(k, :))));
    if e > 1e-5
        error('%s: component %d of the observation is %g in rl_controller.m, %g in Python.', ...
              where, i, obs(i), T.obs(k, i));
    end
    if abs(u - T.u(k)) > 1e-4
        error('%s: u = %g in MATLAB, %g in Python, with the same observation: export again.', where, u, T.u(k));
    end
end
clear rl_controller
fprintf('Check OK: rl_controller.m flies the policy as in Python (%d steps of traj_model.mat).\n', numel(T.t));
end
