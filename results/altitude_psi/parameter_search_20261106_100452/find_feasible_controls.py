import sys, numpy as np, torch, json
sys.path.insert(0,'/home/rayan768/CS/drone/EI-DATA-DRIVEN-CONTROL/src')
from model.rnn_loader import load_rnn_model
from policy.lib.droneEnvRotation import DroneEnv, run_episode
from torch.nn import functional as F
rnn=load_rnn_model('/home/rayan768/CS/drone/EI-DATA-DRIVEN-CONTROL/src/model/checkpoints/rnn_model_thrust_speeds.pt'); torch.set_num_threads(1)
N=200; dt=.05
t=torch.arange(1,N+1)*dt
zref=torch.where(t<2,0.,torch.where(t<8,2.,-.3))
psiref=(t-5).clamp(0,3)*(2*np.pi/3)
means=torch.tensor([-.0205,.0014,0.,0.]); scales=torch.tensor([6.4559,.4590,1.,1.])
outsmean=torch.tensor([-9.9011,.0310]); outsscale=torch.tensor([20.6719,10.7658])
a=torch.zeros(N,2); a[:,0]=float(np.arctanh(1.52/3)); a[:9,0]=2.
a=torch.from_numpy(np.arctanh(np.clip(np.load('/tmp/rotation_feasibility_safe.npz')['u']/3,-.999,.999)))
raw=torch.nn.Parameter(a)
optimizer=torch.optim.Adam([raw],lr=.01)
best=1e99
for it in range(2400):
 optimizer.zero_grad(); u=3*raw.tanh()
 inputs=torch.stack([u[:,0],torch.zeros(N),torch.zeros(N),u[:,1]],dim=-1)
 out,_=rnn(((inputs-means)/scales).unsqueeze(0))
 accel=out[0]*outsscale+outsmean
 vel=accel.cumsum(0)*dt; pos=vel.cumsum(0)*dt
 z,psi=pos.T; vz,wz=vel.T
 # Plan safe motion with terminal settling, ignore abrupt landing reference until final descent.
 ztrajectory=torch.where(t<2,0.,torch.where(t<8,2.,2.-1.15*(t-8)))
 loss=2*(z-ztrajectory).square().mean()+1.5*(psi-psiref).square().mean()+.03*vel.square().mean()+.005*(u[1:]-u[:-1]).square().mean()
 loss=loss+200000*F.relu(-.287-z[:-1]).square().mean()+5000*(z[-1]+.302).square()+1000*(vz[-1]+.3).square()+50*(psi[-1]-2*np.pi).square()+50*wz[-1].square()
 loss.backward(); optimizer.step()
 if loss.item()<best:
  best=loss.item(); ubest=u.detach().numpy().copy(); sbest=torch.stack([z,vz,psi,wz],dim=-1).detach().numpy()
 if it%200==0: print(it,loss.item(),'zmin',z.min().item(),'zend',z[-1].item(),'vz',vz[-1].item(),'psi',psi[-1].item(),'wz',wz[-1].item(),flush=True)
np.savez('/tmp/rotation_feasibility_success.npz',u=ubest,s=sbest,t=t.numpy(),loss=best)
print('BEST',best,'ranges',ubest.min(0),ubest.max(0),'final',sbest[-1],flush=True)
def mission(t):
 return (0. if t<2 else 2. if t<8 else -.3, float(np.clip(t-5,0,3)*2*np.pi/3))
env=DroneEnv(rnn_model=rnn,mission=mission)
idx=[0]
def pol(obs):
 u=ubest[idx[0]];idx[0]+=1;return u
s,a,total,outcome=run_episode(env,pol)
print('ENV',len(a),total,outcome,'min',s[:,0].min(),'max',s[:,0].max(),'final',[float(s[-1,i]) for i in [0,1,11,16]],flush=True)
