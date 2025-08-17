from __future__ import annotations

import numpy as np
from ..toolkit import LinePlot
import matplotlib.pyplot as plt
from .lowlevelsupport import *


class CartesianOrbitBase2D:

    @property
    def t(self)->np.ndarray:
        pass

    @property
    def x(self)->np.ndarray:
        pass

    @property
    def y(self)->np.ndarray:
        pass

    @property
    def xdot(self):
        '''
        override
        '''
        return np.gradient(self.x, self.t)
    
    @property
    def ydot(self):
        '''
        override
        '''
        return np.gradient(self.y, self.t)
    
    def lineplot(self, **kwargs):
        return LinePlot(x=self.x, y=self.y, **kwargs)
    
    def plot(self):
        fig, ax = plt.subplots()
        ax.plot(self.x, self.y)
        return fig, ax
    

class CartesianVariationalOrbitBase2D(CartesianOrbitBase2D):

    @property
    def delx(self)->np.ndarray:
        pass

    @property
    def dely(self)->np.ndarray:
        pass

    @property
    def delx_dot(self)->np.ndarray:
        '''
        override
        '''
        return np.gradient(self.delx, self.t)

    @property
    def dely_dot(self)->np.ndarray:
        '''
        override
        '''
        return np.gradient(self.dely, self.t)
    

class CartesianOrbit2D(LowLevelODE, CartesianOrbitBase2D):

    @property
    def x(self):
        return self.q[:, 0]
    
    @property
    def y(self):
        return self.q[:, 1]
    

class CartesianVariationalOrbit2D(VariationalLowLevelODE, CartesianVariationalOrbitBase2D):

    @property
    def x(self):
        return self.q[:, 0]
    
    @property
    def y(self):
        return self.q[:, 1]
    
    @property
    def delx(self):
        return self.q[:, 2]

    @property
    def dely(self):
        return self.q[:, 3]
    

class HamiltonianOrbit2D(CartesianOrbit2D):

    @property
    def xdot(self):
        return self.q[:, 2]
    
    @property
    def ydot(self):
        return self.q[:, 3]
    

class HamiltonianVariationalOrbit2D(CartesianVariationalOrbit2D):

    @property
    def xdot(self):
        return self.q[:, 4]
    
    @property
    def ydot(self):
        return self.q[:, 5]
    
    @property
    def delx_dot(self):
        return self.q[:, 6]
    
    @property
    def dely_dot(self):
        return self.q[:, 7]
    

class HamiltonianSystem2D(OdeSystem):

    def __new__(cls, V: Expr, t: Symbol, x, y, px, py, args = (), events=())->HamiltonianSystem2D:
        q = [x, y, px, py]
        f = [px, py] + [-V.diff(x), -V.diff(y)]
        obj = object.__new__(cls)
        return cls._process_args(obj, f, t, *q, args=args, events=events)
    

    def get_orbit(self, t0: float, q0: np.ndarray, *, rtol=1e-6, atol=1e-12, min_step=0., max_step=np.inf, first_step=0., args=(), method="RK45"):
        if len(q0) != self.Nsys:
            raise ValueError(f"The size of the initial conditions provided is {len(q0)} instead of {self.Nsys}")
        return HamiltonianOrbit2D(self.lowlevel_odefunc, jac=self.lowlevel_jac, t0=t0, q0=q0, rtol=rtol, atol=atol, min_step=min_step, max_step=max_step, first_step=first_step, args=args, method=method, events=self.true_events)
    


class HamiltonianVariationalSystem2D(HamiltonianSystem2D):

    def __new__(cls, V: Expr, t: Symbol, x, y, px, py, delx, dely, delpx, delpy, args = (), events=()):
        q = [x, y, px, py]
        f = [px, py] + [-V.diff(x), -V.diff(y)]
        delq = [delx, dely, delpx, delpy]

        n = 4
        var_odesys = []
        for i in range(n):
            var_odesys.append(sum([f[i].diff(q[j])*delq[j] for j in range(n)]))
        
        new_sys = f + var_odesys

        obj = object.__new__(cls)
        return cls._process_args(obj, new_sys, t, *[*q, *delq], args=args, events=events)
    

    def get_variational_orbit(self, t0: float, q0: np.ndarray, period: float, *, rtol=1e-6, atol=1e-12, min_step=0., max_step=np.inf, first_step=0., args=(), method="RK45"):
        if len(q0) != self.Nsys:
            raise ValueError(f"The size of the initial conditions provided is {len(q0)} instead of {self.Nsys}")
        return HamiltonianVariationalOrbit2D(self.lowlevel_odefunc, jac=self.lowlevel_jac, t0=t0, q0=q0, period=period, rtol=rtol, atol=atol, min_step=min_step, max_step=max_step, first_step=first_step, args=args, method=method, events=self.true_events)
