"""Discover MATLAB and validate/fly a Python policy export in one batch run."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import numpy as np


def summarize_plant(csv_path, rotation=False, landing_altitude=-.3):
    """Apply the environment's touchdown criteria to actual plant states."""
    data = np.genfromtxt(csv_path, delimiter=',', names=True)
    contact = np.flatnonzero((data['z_ref'] < 0) & (data['z'] <= landing_altitude))
    result = {'success': False, 'peak_altitude_m': float(data['z'].max()),
              'altitude_mae_m': float(np.mean(np.abs(data['z']-data['z_ref']))),
              'sample_period_s': float(np.median(np.diff(data['t']))),
              'touchdown': None}
    if len(contact):
        k = int(contact[0])
        dt = float(np.median(np.diff(data['t'])))
        hold_time = float(np.sum((data['z_ref'][:k] > 0) &
            (np.abs(data['z'][:k]-data['z_ref'][:k]) <= .2))*dt)
        touchdown = {'time_s': float(data['t'][k]), 'z_m': float(data['z'][k]),
                     'vz_m_s': float(data['vz'][k]), 'altitude_hold_time_s': hold_time}
        success = abs(data['vz'][k]) <= .5 and hold_time + 1e-9 >= .5
        if rotation:
            yaw_error = float(data['psi'][k]-data['psi_ref'][k])
            touchdown.update(psi_rad=float(data['psi'][k]), yaw_error_rad=yaw_error,
                             yaw_rate_rad_s=float(data['wz'][k]))
            success = success and abs(yaw_error) <= np.deg2rad(10) and abs(data['wz'][k]) <= .5
        result.update(success=bool(success),touchdown=touchdown)
    return result


def find_matlab(executable=None):
    candidates = [executable, os.environ.get('MATLAB_EXECUTABLE'), shutil.which('matlab'),
                  Path.home()/'bin/matlab']
    for candidate in candidates:
        if candidate is None:
            continue
        path = Path(candidate).expanduser()
        if path.is_file() and os.access(path,os.X_OK):
            return path.resolve()
    return None


def _literal(value):
    return "'"+str(value).replace("'","''")+"'"


def evaluate_export(export_dir, output_dir, *, executable=None, noise_gain=0.):
    """Check actor equivalence, then run Simulink when its runtime is installed.

    Always return an explicit status; a missing toolbox cannot count as a plant pass.
    MATLAB's report and complete log remain beside the Python results.
    """
    export_dir=Path(export_dir).resolve()
    output_dir=Path(output_dir).resolve()
    output_dir.mkdir(parents=True,exist_ok=True)
    matlab=find_matlab(executable)
    if matlab is None:
        return {'status':'unavailable','reason':'MATLAB executable not found; set MATLAB_EXECUTABLE or --matlab'}
    closed_loop=Path(__file__).resolve().parents[1]/'drone/closed_loop'
    report_path=output_dir/'matlab_status.json'
    script_path=output_dir/'run_matlab.m'
    script=f"""% Generated reproducible MATLAB validation and plant evaluation.
addpath({_literal(closed_loop)});
report = struct('status','failed');
try
    report.policy = check_policy_export({_literal(export_dir)});
    if isfile(fullfile({_literal(export_dir)},'rnn.mat'))
        report.rnn = check_rnn_export({_literal(export_dir)});
    end
    if exist('sim','file') > 0 && exist('load_system','file') > 0
        Simulink.fileGenControl('set','CacheFolder',{_literal(output_dir/'simulink_cache')}, ...
            'CodeGenFolder',{_literal(output_dir/'simulink_codegen')},'createDir',true);
        EINT_RL_eval({_literal(export_dir)}, {float(noise_gain):.17g}, {_literal(output_dir/'plant')});
        report.status = 'completed';
    else
        report.status = 'simulink_unavailable';
        report.reason = 'MATLAB actor checks passed; install Simulink to run the plant.';
    end
catch exception
    report.status = 'failed';
    report.reason = exception.message;
    report.identifier = exception.identifier;
    disp(getReport(exception,'extended'));
end
fid = fopen({_literal(report_path)},'w');
fprintf(fid,'%s\\n',jsonencode(report));
fclose(fid);
if strcmp(report.status,'failed'), error('drone:ValidationFailed','%s',report.reason); end
"""
    script_path.write_text(script)
    environment=os.environ.copy()
    environment.setdefault('QT_QPA_PLATFORM','offscreen')
    environment.setdefault('MATLAB_PREFDIR',str(output_dir/'matlab_preferences'))
    log=output_dir/'matlab.log'
    with log.open('w') as handle:
        process=subprocess.run([str(matlab),'-batch',f'run({_literal(script_path)})'],
                               stdout=handle,stderr=subprocess.STDOUT,env=environment)
    if report_path.is_file():
        result=json.loads(report_path.read_text())
    else:
        result={'status':'failed','reason':'MATLAB did not write its validation report'}
    result.update(returncode=process.returncode,log=str(log),executable=str(matlab))
    if process.returncode and result['status']!='failed':
        result.update(status='failed',reason='MATLAB batch process failed; inspect matlab.log')
    csv_path=output_dir/'plant/trajectory.csv'
    if result['status']=='completed' and csv_path.is_file():
        from scipy.io import loadmat
        policy=loadmat(export_dir/'policy.mat',squeeze_me=True)
        result['plant']=summarize_plant(csv_path,rotation=policy.get('mode')=='z_psi')
        if not result['plant']['success']:
            result.update(status='mission_failed',reason='Plant ran, but the touchdown criteria were not met')
        report_path.write_text(json.dumps(result,indent=2)+'\n')
    return result
