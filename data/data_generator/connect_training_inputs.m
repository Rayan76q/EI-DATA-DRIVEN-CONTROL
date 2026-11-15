function connect_training_inputs(model_file)
%CONNECT_TRAINING_INPUTS Connect sim_u1..sim_u4 to the standalone plant.
% This is an explicit model migration; saves a backward-compatible R2024a SLX.
[folder,name] = fileparts(model_file);
assert(~bdIsLoaded(name), 'Close the model before migrating its input wiring.');
addpath(folder);
load_system(model_file);
cleanup = onCleanup(@() close_if_loaded(name));
plant = [name '/Subsystem Reference'];
ports = get_param(plant,'PortHandles');
assert(numel(ports.Inport)==4,'Expected four motor inputs on the plant subsystem.');
for k = 1:4
    source = [name '/motor_input_' num2str(k)];
    if getSimulinkBlockHandle(source) < 0
        add_block('simulink/Sources/From Workspace',source, ...
            'VariableName',sprintf('sim_u%d',k),'SampleTime','Ts', ...
            'Position',[30,50+80*k,170,80+80*k]);
    end
    set_param(source,'Interpolate','off','OutputAfterFinalValue','Holding final value');
    line = get_param(ports.Inport(k),'Line');
    if line >= 0 && get_param(line,'SrcBlockHandle')~=getSimulinkBlockHandle(source)
        % The supplied diagram drives all motors from its fixed Step+Step1 demo.
        assert(strcmp(getfullname(get_param(line,'SrcBlockHandle')),[name '/Sum']), ...
            'Motor input %d has an unexpected source; inspect the model.',k);
        delete_line(name,'Sum/1',['Subsystem Reference/' num2str(k)]);
        line = -1;
    end
    if line < 0
        add_line(name,['motor_input_' num2str(k) '/1'],['Subsystem Reference/' num2str(k)],'autorouting','on');
    end
end
if getSimulinkBlockHandle([name '/Sum']) >= 0
    delete_line(name,'Sum/1','Input/1');
    add_line(name,'motor_input_1/1','Input/1','autorouting','on');
    delete_block([name '/Sum']);
    delete_block([name '/Step']);
    delete_block([name '/Step1']);
end
% Deleting the old demo blocks can leave their purely graphical wire stubs.
lines = find_system(name,'FindAll','on','SearchDepth',1,'Type','line');
for line = lines(:)'
    if get_param(line,'SrcPortHandle') < 0 && all(get_param(line,'DstPortHandle') < 0)
        delete_line(line);
    end
end
% Export to the team's original release rather than upgrading their source format.
folder_out = tempname; mkdir(folder_out);
target = fullfile(folder_out,[name '.slx']);
Simulink.exportToVersion(name,target,'R2024a');
close_system(name,0);
movefile(target,model_file,'f');
rmdir(folder_out);
fprintf('Connected all four training motor inputs: %s\n',model_file);
end

function close_if_loaded(name)
if bdIsLoaded(name), close_system(name,0); end
end
