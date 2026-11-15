"""Regression tests for the interfaces that join RNN, RL and MATLAB stages."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch
from stable_baselines3 import PPO

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from model.rnn_loader import load_rnn_model
from model.training.train import prepare_sequences,train
from policy.lib.droneEnv import DroneEnv
from policy.lib.droneEnvRotation import DroneEnv as Rotation,ALTITUDE_HOLD_TIME
from policy.export_policy import _policy_matrices,_predict_matrices,export_for_matlab
from policy.training import ScaledControlActions,ref_mission
from policy.matlab import summarize_plant


class PipelineContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.z=load_rnn_model(ROOT/'src/model/checkpoints/rnn_z.pt')
        cls.psi=load_rnn_model(ROOT/'src/model/checkpoints/rnn_model_thrust_speeds.pt')
        cls.agent=PPO.load(ROOT/'src/policy/checkpoints/ppo_drone_thrust_speeds.zip',device='cpu')

    def test_recurrent_sequence_matches_streaming(self):
        for model in [self.z,self.psi]:
            x=torch.from_numpy(np.random.default_rng(7).normal(size=(1,17,model.rnn.input_size)).astype(np.float32))
            with torch.inference_mode():
                sequence,final=model(x)
                hidden=None;outputs=[]
                for k in range(x.shape[1]):
                    y,hidden=model(x[:,k:k+1],hidden);outputs.append(y)
            np.testing.assert_allclose(sequence.numpy(),torch.cat(outputs,dim=1).numpy(),atol=1e-6)
            np.testing.assert_allclose(final.numpy(),hidden.numpy(),atol=1e-6)

    def test_environments_do_not_share_recurrent_state(self):
        a,b=DroneEnv(rnn_model=self.z),DroneEnv(rnn_model=self.z)
        try:
            a.reset(seed=0);b.reset(seed=0)
            a.step([.2]);b.step([.8])
            self.assertFalse(np.array_equal(a.state,b.state))
            self.assertNotEqual(a.hidden_state.data_ptr(),b.hidden_state.data_ptr())
            a.reset(seed=0);b.reset(seed=0)
            np.testing.assert_array_equal(a.step([.4])[0],b.step([.4])[0])
        finally:a.close();b.close()

    def test_two_axis_motor_layout_and_output_order(self):
        motors=np.array([[[.7,.3,.7,.3],[.6,.4,.6,.4]]],dtype=np.float32)
        states=np.arange(36,dtype=np.float32).reshape(1,2,18)
        x,y=prepare_sequences(motors.reshape(1,-1),states.reshape(1,-1),'z_psi','motors','plant')
        np.testing.assert_allclose(x[0,0],[.5,0,0,.2],atol=1e-7)
        np.testing.assert_array_equal(y,states[:,:,[8,17]])
        with self.assertRaises(ValueError):
            prepare_sequences(motors.reshape(1,-1),states.reshape(1,-1),'z','motors','plant')

    def test_export_includes_saved_observation_and_action_scales(self):
        policy=_policy_matrices(self.agent)
        probes=np.vstack([np.zeros(6),np.eye(6),-np.eye(6),100*np.eye(6)])
        for obs in probes:
            action=self.agent.predict(obs.astype(np.float32),deterministic=True)[0]
            raw=np.asarray(self.agent.action_offset)+np.asarray(self.agent.action_scale)*action
            np.testing.assert_allclose(_predict_matrices(policy,obs),raw,atol=1e-5)
        env=ScaledControlActions(Rotation(rnn_model=self.psi),[1,.1],[.4,0])
        try:
            for a,b in [([-1,-1],[-.6,-.1]),([0,0],[.4,0]),([1,1],[1.4,.1])]:
                np.testing.assert_allclose(env.action(a),b,atol=1e-7)
        finally:env.close()

    def test_full_turn_is_distinct_from_zero_and_requires_altitude(self):
        env=Rotation(rnn_model=self.psi)
        try:
            env.reset();env.state[0]=-.3;env.state[1]=0;env.t=12
            env.state[11]=0;env.state[16]=0;env.altitude_hold_time=ALTITUDE_HOLD_TIME
            self.assertEqual(env.outcome(),('landed','no_turn'))
            env.state[11]=np.float32(2*np.pi)
            self.assertEqual(env.outcome(),('landed','has_turned'))
            env.altitude_hold_time=0
            self.assertEqual(env.outcome(),('missed_altitude','has_turned'))
        finally:env.close()

    def test_new_training_bundle_carries_its_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)
            x=np.random.default_rng(0).uniform(.1,.8,size=(5,9)).astype(np.float32)
            y=2*x-1
            np.savetxt(p/'x.csv',x,delimiter=',');np.savetxt(p/'y.csv',y,delimiter=',')
            model,path=train(p/'x.csv',p/'y.csv',p/'trained',epochs=1,hidden_size=4,device='cpu')
            loaded=load_rnn_model(path)
            with torch.inference_mode():
                sample=torch.zeros(1,3,1)
                np.testing.assert_array_equal(model(sample)[0].numpy(),loaded(sample)[0].numpy())
            env=DroneEnv(rnn_model=loaded,mission=ref_mission)
            try:
                self.assertEqual(env.p.mean_az,loaded.training_metadata['normalization']['output_mean'][0])
                self.assertEqual(env.p.sigma_u,loaded.training_metadata['normalization']['input_scale'][0])
            finally:env.close()

    def test_complete_yaw_export_keeps_physical_actions(self):
        from scipy.io import loadmat
        with tempfile.TemporaryDirectory() as directory:
            env=ScaledControlActions(Rotation(rnn_model=self.psi),self.agent.action_scale,self.agent.action_offset)
            try:
                policy_path,trajectory_path=export_for_matlab(self.agent,env,directory)
            finally:env.close()
            policy=loadmat(policy_path,squeeze_me=True)
            trajectory=loadmat(trajectory_path,squeeze_me=True)
            self.assertEqual(policy['observation_order'],'z-z_ref,vz,az,psi-psi_ref,wz,wdotz')
            self.assertEqual(trajectory['u'].shape[1],2)
            np.testing.assert_allclose(trajectory['u']@policy['motor_mixer'].T,trajectory['motors'])
            self.assertGreater(trajectory['psi'][-1],6)

    def test_plant_verdict_uses_first_contact_and_unwrapped_yaw(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'trajectory.csv'
            t=np.arange(25)*.05
            z=np.where(t<1,2.,-.31)
            zref=np.where(t<1,2.,-.3)
            vz=np.zeros_like(t);vz[20]=-1.
            psi=np.full_like(t,2*np.pi)
            values=np.column_stack([t,z,zref,vz,psi,psi,np.zeros_like(t)])
            def verdict():
                np.savetxt(path,values,delimiter=',',header='t,z,z_ref,vz,psi,psi_ref,wz',comments='')
                return summarize_plant(path,rotation=True)
            self.assertFalse(verdict()['success'])  # Later slow samples cannot rescue a hard landing.
            values[20,3]=-.1
            self.assertTrue(verdict()['success'])
            values[:,4]=0
            self.assertFalse(verdict()['success'])  # Same wrapped heading is not a full turn.


if __name__=='__main__':unittest.main()
