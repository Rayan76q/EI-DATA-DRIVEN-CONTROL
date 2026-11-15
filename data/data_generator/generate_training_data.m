function files = generate_training_data(mode, output_dir, n_series, horizon, seed)
%GENERATE_TRAINING_DATA Reproducible motor inputs and all 18 plant states.
% One trajectory per CSV row. Use train.py --input-format motors --output-format plant.
if nargin < 1, mode = 'z'; end
if nargin < 2, output_dir = fullfile(fileparts(fileparts(mfilename('fullpath'))),'generated_data'); end
if nargin < 3, n_series = 50; end
if nargin < 4, horizon = 8; end
if nargin < 5, seed = 0; end
assert(any(strcmp(mode,{'z','z_psi'})), 'Choose z or z_psi.');
validateattributes(n_series,{'numeric'},{'scalar','integer','positive'});
validateattributes(horizon,{'numeric'},{'scalar','finite','positive'});
here = fileparts(mfilename('fullpath'));
root = fileparts(fileparts(here));
plant = fullfile(root,'src','drone','plant');
addpath(plant);
random_init;
rng(seed,'twister');
Ts = 0.05; g_noise = 0; g = 9.816;
t = (0:Ts:horizon)';
assert(numel(t) >= 2, 'Horizon must cover at least one control interval.');
if ~isfolder(output_dir), mkdir(output_dir); end
files = struct('inputs',fullfile(output_dir,[mode '_inputs.csv']), ...
               'outputs',fullfile(output_dir,[mode '_outputs.csv']));
input_file = fopen(files.inputs,'w'); output_file = fopen(files.outputs,'w');
assert(input_file >= 0 && output_file >= 0, 'Cannot create training CSVs.');
close_files = onCleanup(@() close_pair(input_file,output_file));
name = 'EINT_system_simulation';
was_loaded = bdIsLoaded(name);
load_system(fullfile(plant,[name '.slx']));
if ~was_loaded, cleanup = onCleanup(@() close_system(name,0)); end
seeds = {'seed_x','seed_y','seed_z','seed_vx','seed_vy','seed_vz','seed_ax','seed_ay','seed_az', ...
         'seed_phi','seed_psi','seed_theta','seed_wx','seed_wy','seed_wz','seed_w_dot_x','seed_w_dot_y','seed_w_dot_z'};
excitation = zeros(n_series,2);
for episode = 1:n_series
    hold_samples = max(1,round(.5/Ts));
    knots = ceil(numel(t)/hold_samples);
    thrust = repelem(.25 + .5*rand(knots,1),hold_samples);
    yaw = zeros(size(thrust));
    if strcmp(mode,'z_psi'), yaw = repelem(.15*(2*rand(knots,1)-1),hold_samples); end
    thrust = thrust(:); yaw = yaw(:);
    commands = [thrust(1:numel(t)),yaw(1:numel(t))];
    motors = commands * [1,1;1,-1;1,1;1,-1]';
    sim_input = Simulink.SimulationInput(name);
    sim_input = sim_input.setModelParameter('StopTime',num2str(t(end),17), ...
        'SaveTime','on','TimeSaveName','tout','ReturnWorkspaceOutputs','on');
    sim_input = sim_input.setVariable('Ts',Ts).setVariable('g_noise',g_noise).setVariable('g',g);
    for k = 1:4
        sim_input = sim_input.setVariable(sprintf('sim_u%d',k),timeseries(motors(:,k),t));
    end
    for k = 1:numel(seeds), sim_input = sim_input.setVariable(seeds{k},randi(10000)); end
    result = sim(sim_input);
    signal = result.simout;
    if isa(signal,'Simulink.SimulationData.Dataset'), signal = signal{1}.Values; end
    if isa(signal,'timeseries')
        time = signal.Time(:); states = reshape(signal.Data,numel(time),[]);
    elseif isstruct(signal)
        time = signal.time(:); states = signal.signals.values;
    else
        time = result.tout(:); states = signal;
    end
    assert(size(states,2)==18, 'Plant must log all 18 states in the documented order.');
    [time,index] = unique(time,'last');
    states = interp1(time,states(index,:),t,'linear');
    assert(all(isfinite(states),'all'), 'Plant generated non-finite states.');
    excitation(episode,:) = states(end,[9,18]);
    fprintf(input_file,'%.9g,',reshape(motors',1,[])); fprintf(input_file,'\n');
    fprintf(output_file,'%.9g,',reshape(states',1,[])); fprintf(output_file,'\n');
end
if n_series > 1
    assert(std(excitation(:,1)) > 1e-6, 'drone:DisconnectedInputs', ...
        'Motor commands did not excite acceleration; inspect the standalone plant wiring.');
    if strcmp(mode,'z_psi')
        assert(std(excitation(:,2)) > 1e-6, 'Yaw commands did not excite yaw acceleration.');
    end
end
metadata = struct('mode',mode,'dt',Ts,'series',n_series,'horizon',t(end),'seed',seed, ...
    'input_order',{{'motor1','motor2','motor3','motor4'}}, ...
    'output_order',{{'x','y','z','vx','vy','vz','ax','ay','az','phi','theta','psi','wx','wy','wz','wdotx','wdoty','wdotz'}});
file = fopen(fullfile(output_dir,[mode '_dataset.json']),'w');
fprintf(file,'%s\n',jsonencode(metadata)); fclose(file);
end

function close_pair(a,b)
fclose(a); fclose(b);
end
