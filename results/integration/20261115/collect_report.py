"""Collect the existing integration artifacts; run from any working directory."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
sys.path.insert(0, str(PROJECT/'src'))
from policy.matlab import summarize_plant


def read(path):
    return json.loads((ROOT/path).read_text())


def file_identity(path):
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT/path
    return {'path': str(path.relative_to(PROJECT)),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


plant = {}
for name, folder, policy, rotation in [
    ('z_ppo_default', 'main_z/plant', read('main_z/metrics.json')['policy_path'], False),
    ('z_ppo_bundled', 'plant_z_ppo', 'src/policy/checkpoints/ppo_drone_rnn.zip', False),
    ('z_psi_ppo', 'main_z_psi/plant', read('main_z_psi/metrics.json')['policy_path'], True),
    ('z_dqn', 'plant_z_dqn', read('main_z_dqn/metrics.json')['policy_path'], False),
    ('z_sac', 'plant_z_sac', read('main_z_sac/metrics.json')['policy_path'], False),
]:
    plant[name] = dict(summarize_plant(ROOT/folder/'trajectory.csv', rotation=rotation),
                       policy=file_identity(policy), trajectory=f'{folder}/trajectory.csv')
older_sac = {}
for name in ['20261105_202000_ztol02', '20261105_193731_full']:
    older_sac[name] = dict(summarize_plant(ROOT/'sac_transfer'/name/'plant/trajectory.csv'),
        policy=file_identity(f'results/altitude/sac/{name}/training/best_model.zip'))
assert all(plant[name]['success'] for name in ['z_ppo_default','z_ppo_bundled','z_psi_ppo','z_dqn'])
assert not plant['z_sac']['success'] and not any(item['success'] for item in older_sac.values())
for mode in ['z', 'z_psi']:
    assert read(f'main_{mode}/metrics.json')['simulink']['status'] == 'completed'
report = {
    'date': '2026-11-15',
    'branch': 'final',
    'integration_status': 'Primary z and z+psi pipelines passed; existing SAC plant transfer fails',
    'merge': {'local_preservation_commit': '9ff94d9', 'team_merge_commit': 'a505ddf',
              'origin_final': read('preservation.json')['refs']['origin/final'],
              'main_unchanged': read('preservation.json')['refs']['main'],
              'main_asset_import': 'src/drone/plant/EINT_system_simulationPID.slx'},
    'policy_preservation': 'No RL weights, configs, rewards or training hyperparameters changed during integration',
    'matlab_numerical_validation': read('matlab_policy_validation.json'),
    'plant': plant,
    'older_sac_plant_checks': older_sac,
    'main_cli': {mode: read(f'main_{mode}/metrics.json')['simulink'] for mode in ['z','z_psi']},
    'data_generation_to_training': read('generated_training_validation.json'),
    'provided_data_training': {mode: {'epochs': 1, 'max_trajectories': 8,
        'last_epoch': read(f'rnn_train_{mode}/training.json')['history'][-1],
        'input': read(f'rnn_train_{mode}/training.json')['metadata']['inputs'],
        'output': read(f'rnn_train_{mode}/training.json')['metadata']['outputs']}
        for mode in ['z','z_psi']},
    'rl_training_smoke': read('rl_training_smoke/smoke_report.json'),
    'unit_tests': {'passed': 8, 'log': 'unit_tests.log'},
    'static_validation': read('static_validation.json'),
    'notebook_execution': {'status': 'passed', 'max_trajectories': 8,
        'notebooks': ['dataset_cleaning.ipynb','test_model_az.ipynb','test_model_az_wdotz.ipynb'],
        'scope': 'Data loading, preparation, normalization, model inference and export cells; outputs preserved'},
    'preservation_report': 'preservation.json',
    'acceptance': {'control_period_s': .05, 'landing_altitude_m': -.3,
        'max_abs_touchdown_vz_m_s': .5, 'altitude_tolerance_m': .2,
        'min_cumulative_positive_reference_occupancy_s': .5,
        'yaw_error_tolerance_deg': 10, 'max_abs_touchdown_yaw_rate_rad_s': .5,
        'yaw': 'unwrapped, full 2*pi turn', 'plant_noise_gain': 0},
    'limitations': [
        'SAC actor/export checks pass, but selected and two older policies violate plant landing criteria. User chose preservation over tuning.',
        'Short RNN and RL training runs verify execution and contracts, not convergence.',
        'Plant success uses the first sampled landing threshold event; the continuous plant has no ground/contact model and runs after touchdown.',
        'Zero-noise nominal missions only; no robustness claim for noise or varied initial conditions.',
        'Hover occupancy is cumulative, not a continuous settling requirement.',
    ],
}
(ROOT/'plant_metrics.json').write_text(json.dumps(plant, indent=2)+'\n')
(ROOT/'report.json').write_text(json.dumps(report, indent=2)+'\n')
print('Wrote report.json:', {name: item['success'] for name,item in plant.items()})
