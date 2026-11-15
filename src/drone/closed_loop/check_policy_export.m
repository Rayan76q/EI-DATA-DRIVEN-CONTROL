function report = check_policy_export(export_dir)
%CHECK_POLICY_EXPORT Compare MATLAB actions and observations with Python samples.
P = load(fullfile(export_dir, 'policy.mat'));
T = load(fullfile(export_dir, 'traj_model.mat'));
previous = getenv('DRONE_POLICY_EXPORT_DIR');
restore = onCleanup(@() setenv('DRONE_POLICY_EXPORT_DIR', previous));
setenv('DRONE_POLICY_EXPORT_DIR', export_dir);
clear rl_controller rl_controller2

maximum_error = 0;
for k = 1:size(P.validation_obs, 1)
    actual = drone_policy_action(P, P.validation_obs(k, :));
    expected = P.validation_actions(k, :)';
    maximum_error = max(maximum_error, max(abs(actual - expected)));
    assert(all(abs(actual - expected) <= 1e-4 + 1e-5 * abs(expected)), ...
        'Exported actor disagrees with SB3 observation %d.', k);
end

for k = 1:numel(T.t)
    if size(T.obs, 2) == 6
        [u1, u2, u3, u4, z_ref, obs] = rl_controller2( ...
            T.t(k), T.z(k), T.vz(k), T.az_prev(k), ...
            0, 0, T.psi(k), 0, 0, T.wz(k), 0, 0, T.wdotz_prev(k), 0, 0, 0, 0);
        assert(max(abs([u1, u2, u3, u4] - T.motors(k, :))) < 1e-4, ...
            'Motor mixer disagrees with Python at sample %d.', k);
        [~, psi_ref] = drone_policy_reference(P, T.t(k));
        assert(abs(psi_ref - T.psi_ref(k)) < 1e-9, 'Yaw reference differs at sample %d.', k);
    else
        [u, z_ref, obs] = rl_controller(T.t(k), T.z(k), T.vz(k), T.az_prev(k), T.u_prev(k));
        assert(abs(u - T.u(k)) < 1e-4, 'Altitude action differs at sample %d.', k);
    end
    assert(abs(z_ref - T.z_ref(k)) < 1e-9, 'Altitude reference differs at sample %d.', k);
    relative_error = abs(obs(:)' - T.obs(k, :)) ./ max(1, abs(T.obs(k, :)));
    assert(max(relative_error) < 1e-5, 'Observation differs at sample %d.', k);
end
clear rl_controller rl_controller2

report = struct('status', 'passed', 'samples', numel(T.t), ...
    'independent_probes', size(P.validation_obs, 1), 'max_action_error', maximum_error);
fprintf('MATLAB policy check passed: %d samples, %d probes, max error %.3g.\n', ...
    report.samples, report.independent_probes, report.max_action_error);
end
