from __future__ import annotations
from ..toolkit import Template
import numpy as np
from .odes import *



class Orbit(Template):

    ode: LowLevelODE
    data: np.ndarray
    diverges: bool
    is_stiff: bool
    
    def __init__(self, ode: SymbolicOde, lowlevel=True, stack=True):
        Orbit._init(self, ode, lowlevel=lowlevel, stack=stack)

    def _init(self, ode: SymbolicOde, lowlevel: bool, stack: bool, **kwargs):
        _ode = ode.ode(lowlevel=lowlevel, stack=stack, variational=self.is_variational)
        
        nsys = 2*ode.Nsys if self.is_variational else ode.Nsys
        Template.__init__(self, ode=_ode, data=np.empty((0, nsys+1), dtype=np.float64), diverges=False, is_stiff=False, **kwargs)
    
    @property
    def dof(self)->int:
        return self.data.shape[1]-1
    
    @property
    def t(self)->np.ndarray:
        return self.data[:, 0]

    @property
    def f(self)->np.ndarray:
        return self.data[:, 1:]
    
    @property
    def is_variational(self):
        return isinstance(self, VariationalOrbit)

    def newcopy(self):
        obj = self.clone()
        obj.clear()
        return obj
    
    def clear(self):
        self._set(data=self._empty(), diverges=False, is_stiff=False)

    def reset(self):
        if self.data.shape[0] > 0:
            self._remake(self.data[0, 0], self.data[0, 1:])
    
    def set_ics(self, t0: float, f0: np.ndarray):
        f0 = np.array(f0)
        if f0.shape != (self.dof,):
            raise ValueError(f"Initial conditions need to be a 1D array of size {self.dof}")
        self._remake(t0, f0)

    def now(self, **kwargs):
        return self._parse_ics((self.data[-1, 0], self.data[-1, 1:]), **kwargs)

    def integrate(self, Delta_t, dt, **kwargs):
        if self.diverges or self.is_stiff:
            return None
            # return OdeResult(self.t[-1:], self.f[-1:, :], diverges=self.diverges, is_stiff=self.is_stiff, runtime=0.)
        elif Delta_t<0 or dt<0:
            raise ValueError('Invalid Delta_t or dt inserted')
        elif Delta_t < dt:
            raise ValueError('Delta_t must be greater than dt')
        
        if self.data.shape[0] == 0:
            raise RuntimeError('No initial conditions have been set')
        
        func = kwargs.pop('func', 'solve')
        renorm = kwargs.pop('renorm', False)
        ics = self._parse_ics((self.data[-1, 0], self.data[-1, 1:]), renorm=renorm)
        
        res: OdeResult = getattr(self.ode, func)(ics, self.data[-1, 0]+Delta_t, dt, **kwargs)
        self._absorb_oderes(res, renorm=renorm)
        return res

    def _parse_ics(self, ics, **kwargs):
        return (float(ics[0]), list(ics[1]))
    
    def _absorb_oderes(self, res: OdeResult, **kwargs):
        tarr, farr = res.t, res.y
        
        newdata = np.column_stack((tarr, farr))
        data = np.concatenate((self.data, newdata[1:]))
        self._set(data=data, diverges=res.diverges, is_stiff=res.is_stiff)
        return res

    def _empty(self):
        return np.empty((0, self.dof+1), dtype=np.float64)

    def _remake(self, t0, f0):
        data = np.array([[t0, *f0]], dtype=np.float64)
        if data.shape != (1, self.dof+1):
            raise ValueError(f"The provided initial conditions have data shape {data.shape} instead of {(1, self.dof+1)}")
        self._set(data=data, diverges=False, is_stiff=False)

from scipy.integrate import solve_ivp

class VariationalOrbit(Orbit):

    _logksi: list[float]

    def __init__(self, ode: SymbolicOde, lowlevel=True, stack=True):
        VariationalOrbit._init(self, ode, lowlevel=lowlevel, stack=stack, _logksi=[])

    @property
    def q(self):
        return self.data[:, 1:1+self.dof//2].transpose()

    @property
    def delq(self):
        return self.data[:, 1+self.dof//2:].transpose()
    
    @property
    def ksi(self):
        return np.linalg.norm(self.delq, axis=1)
    
    @property
    def lyapunov(self):
        res = np.array(self._logksi[1:])/(self.t[1:]-self.t[0])
        res = np.concatenate((np.array([0.]), res))
        return self.t, res
    
    def reset(self):
        Orbit.reset(self)
        if self._logksi:
            self._set(_logksi = [self._logksi[0]])

    def clear(self):
        Orbit.clear(self)
        self._set(_logksi=[])

    def set_ics(self, t0, f0):
        self._set(_logksi=[0.])
        q0, delq0 = f0[:self.dof//2], f0[self.dof//2:]
        delq0 = np.array(delq0)/np.linalg.norm(delq0)
        f0 = [*q0, *delq0]
        Orbit.set_ics(self, t0, f0)
    
    def get(self, Delta_t, dt, split=100, renorm=False, **odekw):

        for _ in range(split):
            self.integrate(Delta_t/split, dt, renorm=renorm, **odekw)
    
    def _parse_ics(self, ics, **kwargs):
        t0, f0 = ics
        q0 = f0[:self.dof//2]
        delq0 = f0[self.dof//2:]
        if kwargs.pop('renorm', False):
            ksi = np.linalg.norm(delq0)
            delq0 = delq0/ksi
        qnew = np.concatenate((q0, delq0))
        return (t0, qnew)
    
    def _absorb_oderes(self, res, renorm=False):
        Orbit._absorb_oderes(self, res)
        self._absorb_ksi(res, renorm=renorm)

    def _absorb_ksi(self, res: OdeResult, renorm=False):
        ksi = np.linalg.norm(res.y[:, self.dof//2:], axis=1)
        logksi = np.log(ksi)
        if renorm:
            logksi += self._logksi[-1]
        self._set(_logksi=self._logksi+list(logksi[1:]))


class HamiltonianOrbit(Orbit):

    def __init__(self, potential: sym.Expr, variables: tuple[sym.Variable, ...], args: tuple[sym.Variable, ...] = (), lowlevel=True, stack=True):
        HamiltonianOrbit._init(self, potential=potential, variables=variables, args=args, lowlevel=lowlevel, stack=stack)
    
    def _init(self, potential: sym.Expr, variables: tuple[sym.Variable, ...], args: tuple[sym.Variable, ...], lowlevel: bool, stack: bool, **kwargs):
        hs = HamiltonianSystem(potential, *variables, args=args)
        return Orbit._init(self, hs.symbolic_ode, lowlevel, stack, **kwargs)

    @property
    def nd(self):
        if self.is_variational:
            return self.dof//4
        else:
            return self.dof//2

    @property
    def x(self):
        return self.data[:, 1:1+self.nd].transpose()

    @property
    def p(self):
        return self.data[:, 1+self.nd:1+2*self.nd].transpose()


class VariationalHamiltonianOrbit(VariationalOrbit, HamiltonianOrbit):

    def __init__(self, potential: sym.Expr, variables: tuple[sym.Variable, ...], args: tuple[sym.Variable, ...] = (), lowlevel=True, stack=True):
        HamiltonianOrbit._init(self, potential=potential, variables=variables, args=args, lowlevel=lowlevel, stack=stack, _logksi=[])
    
    @property
    def delx(self):
        return self.data[:, 1+2*self.nd:1+3*self.nd].transpose()

    @property
    def delp(self):
        return self.data[:, 1+3*self.nd:].transpose()


class FlowOrbit(Orbit):

    def __init__(self, ode: SymbolicOde, lowlevel=True, stack=True):
        Orbit.__init__(self, ode, lowlevel, stack)

    @property
    def nd(self):
        if self.is_variational:
            return self.dof//2
        else:
            return self.dof

    @property
    def x(self):
        return self.data[:, 1:].transpose()


class VariationalFlowOrbit(VariationalOrbit, FlowOrbit):

    def __init__(self, ode: SymbolicOde, lowlevel=True, stack=True):
        VariationalOrbit.__init__(self, ode, lowlevel, stack)

    @property
    def x(self):
        return self.data[:, 1:1+self.nd].transpose()
    
    @property
    def delx(self):
        return self.data[:, 1+self.nd:].transpose()


class HamiltonianSystem:

    _instances: list[tuple[tuple[sym.Expr, ...], HamiltonianSystem]] = []

    __args: tuple[sym.Expr, tuple[sym.Variable, ...], tuple[sym.Variable, ...]]

    def __new__(cls, potential: sym.Expr, *variables: sym.Variable, args: tuple[sym.Variable, ...]=()):
        args = (potential, variables, args)
        for i in range(len(cls._instances)):
            if cls._instances[i][0] == args:
                return cls._instances[i][1]
        for v in variables:
            if len(v.name) != 1 or v.name == 't':
                raise ValueError("All variables in the dynamical system need to have exactly one letter, and different from 't'") 
        obj = super().__new__(cls)
        cls._instances.append((args, obj))
        return obj

    def __init__(self, potential: sym.Expr, *variables: sym.Variable, args: tuple[sym.Variable, ...]=()):
        self.__args = (potential, variables, tuple(args))
            
    @property
    def nd(self):
        return len(self.variables)
    
    @property
    def variables(self)->tuple[sym.Variable, ...]:
        return self.__args[1]
    
    @property
    def extras(self):
        return self.__args[2]
    
    @property
    def V(self)->sym.Expr:
        return self.__args[0]
    
    @cached_property
    def H(self):
        p = self.ode_vars[self.nd:]
        T = sum([pi**2 for pi in p])/2
        return T + self.V
    
    @property
    def x(self):
        return self.variables
    
    @property
    def p(self):
        return self.ode_vars[self.nd:]
    
    @cached_property
    def rhs(self):
        xdot = [self.H.diff(pi) for pi in self.p]
        pdot = [-self.H.diff(xi) for xi in self.x]
        return xdot+pdot
    
    @cached_property
    def ode_vars(self)->tuple[sym.Variable, ...]:
        return tuple([*self.variables] + [sym.Variable('p'+xi.name) for xi in self.variables])
    
    @cached_property
    def symbolic_ode(self):
        return SymbolicOde(*self.rhs, symbols=[sym.Variable('t'), *self.ode_vars], args=self.extras)
    
    def ode(self, lowlevel=True, stack=True, variational=False):
        return self.symbolic_ode.ode(lowlevel=lowlevel, stack=stack, variational=variational)

    def new_orbit(self, q0, lowlevel=True, stack=True):
        orb = HamiltonianOrbit(self.V, self.variables, self.extras, lowlevel=lowlevel, stack=stack)
        orb.set_ics(0., q0)
        return orb

    def new_varorbit(self, q0, delq0=None, lowlevel=True, stack=True):
        orb = VariationalHamiltonianOrbit(self.V, self.variables, self.extras, lowlevel=lowlevel, stack=stack)
        if delq0 is None:
            delq0 = [1., *((2*self.nd-1)*[0.])]
        orb.set_ics(0., [*q0, *delq0])
        return orb


def integrate_all(orbits: list[Orbit], Delta_t, dt, threads=-1, **kwargs)->list[OdeResult]:

    ode_data = []
    kw = dict(t=ics[0]+Delta_t, dt=dt)
    kw.update(kwargs)
    renorm = kwargs.pop('renorm', False)
    for orb in orbits:
        ics = orb.now(renorm=renorm)
        kw.update()
        ode_data.append((orb.ode, kw))

    res = LowLevelODE.dsolve_all(ode_data, threads)

    for i in range(len(orbits)):
        orbits[i]._absorb_oderes(res[i], **kwargs)

    
    return res
