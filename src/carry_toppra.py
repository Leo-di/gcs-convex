import numpy as np
from toppra.interpolator import AbstractGeometricPath
from toppra import algorithm,constraint
from .timed_carry_bezier import LIMITS

class BezierPath(AbstractGeometricPath):
 def __init__(self,segments):
  self.n=(len(segments[0]['x'])-4)//4
  self.controls=np.array([s['x'][:4*self.n] for s in segments]).reshape(-1,4,self.n)
 @property
 def dof(self):return self.n
 @property
 def path_interval(self):return np.array([0.,len(self.controls)])
 def __call__(self,positions,order=0):
  raw=np.asarray(positions);s=np.atleast_1d(raw);i=np.minimum(np.floor(s).astype(int),len(self.controls)-1);u=s-i;p=self.controls[i]
  if order==0:b=np.array([(1-u)**3,3*u*(1-u)**2,3*u*u*(1-u),u**3]).T
  elif order==1:b=3*np.array([(1-u)**2,2*u*(1-u),u*u]).T;p=np.diff(p,axis=1)
  elif order==2:b=6*np.array([1-u,u]).T;p=np.diff(p,n=2,axis=1)
  else:raise ValueError(order)
  values=np.einsum('ni,nij->nj',b,p);return values[0] if raw.ndim==0 else values


def retime(segments,velocity=LIMITS,acceleration=None):
 """TOPPRA exact cubic geometry, sampled bounds with uniform slowdown.

 Research caps, not hardware certification. Does not recheck collisions.
 """
 if acceleration is None:acceleration=np.array([.5,.5,.8]+[1.]*7)
 path=BezierPath(segments)
 instance=algorithm.TOPPRA([
  constraint.JointVelocityConstraint(velocity),
  constraint.JointAccelerationConstraint(acceleration,discretization_scheme=constraint.DiscretizationType.Interpolation)
 ],path,gridpoints=np.linspace(0,len(path.controls),1000*len(path.controls)+1),
 solver_wrapper='seidel',parametrizer='ParametrizeConstAccel')
 trajectory=instance.compute_trajectory(0,0)
 if trajectory is None:return {'solved':False}
 ts=np.unique(np.clip(np.r_[np.linspace(0,trajectory.duration,20001),trajectory._ts-1e-8,trajectory._ts+1e-8],0,trajectory.duration))
 vr=float(np.max(abs(trajectory(ts,1))/velocity))
 ar=float(np.max(abs(trajectory(ts,2))/acceleration))
 scale=max(1.,vr,np.sqrt(ar))*1.001
 return {'solved':True,'raw_duration':float(trajectory.duration),'scaled_duration':float(scale*trajectory.duration),
 'slowdown':scale,'raw_velocity_ratio':vr,'raw_acceleration_ratio':ar,
 'checked_velocity_ratio':vr/scale,'checked_acceleration_ratio':ar/scale**2,'samples':len(ts),
 'scope':'Dense sampled derivative validation, not continuous certification; original spatial path preserved.'}
