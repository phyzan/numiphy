from __future__ import annotations
from typing import Callable
from ..findiffs import grids
from . import bounds
# from .. import symbolic as sym
from . import cached
from ..toolkit import tools
from .. import odesolvers as ods
from . import linalg
import numpy as np
from abc import ABC, abstractmethod


class Propagator:

    f0: np.ndarray
    coefs: np.ndarray

    def __init__(self, eigp: linalg.EigProblem, f0: Callable[..., np.ndarray], coef=1.0):
        self.eigp = eigp
        self._coef = coef
        self.f0 = eigp.bcs.discretize(f0, reduced=True) #asserts bcs are homogeneous
        self.is_constructed = False

    def construct(self):
        if not self.is_constructed:
            self.eigp.solve()
            if self.eigp.isherm:#############change for analytical calculation
                self.coefs = self.eigp.f.transpose().dot(self.f0)
            else:
                self.coefs = np.linalg.solve(self.eigp.f, self.f0)
            self.is_constructed = True

    def apply(self, t: float|np.ndarray)->np.ndarray:
        if hasattr(t, '__iter__'):
            g = np.exp(self._coef * np.outer(t, self.eigp.e)) * self.coefs
            f = np.tensordot(g, self.eigp.f, axes=[1, 1])
        else:
            f = self.eigp.f.dot(np.exp(self._coef * self.eigp.e * t) * self.coefs)
        f = self.eigp.bcs.insert(f)
        return f.flatten().reshape((*self.eigp.grid.shape, *t.shape), order='F')


class IVP(ABC):

    def __init__(self, bcs: bounds.GroupedBcs, grid: grids.Grid, *operators: sym.Expr):
        bcs.apply_grid(grid)
        self.grid = bcs.grid
        self.bcs = bcs
        self.operators = operators
        self.f0 = None

    @abstractmethod
    def dfdt(self, t, f, *ops: cached.CachedOperator)->np.ndarray:
        '''
        ops is a representation of self.operators
        The matrices can be reduced if the bcs are homogeneous.
        Otherwise they are full
        '''
        pass
    
    def set_ics(self, f0: Callable[..., np.ndarray]):
        self.f0 = f0

    def apply_bcs(self, t, f: np.ndarray):
        return self.bcs.apply(f, t)

    def solve(self, t, dt, method='RK4', rtol=0., max_frames=-1, display=True, acc=1, fd='central'):
        '''
        remember to include non-autonomous IVP's
        '''
        ics = (0, self.bcs.discretize(self.f0, 0))
        kwargs = dict(t=t, dt=dt, method=method, rtol=rtol, max_frames=max_frames, display=display)
        if self.bcs.is_homogeneous:
            ops = tuple([cached.cache_operator(op, self.grid, self.bcs, acc, fd) for op in self.operators])
            ics = (ics[0], self.bcs.reduced_array(ics[1]))
            ode = ods.ODE(self.dfdt, ics=ics)
            x, f = ode.solve(**kwargs, args = ops)
            f = self.bcs.insert(f)
        else:
            ops = tuple([cached.cache_operator(op, self.grid, acc=acc, fd=fd) for op in self.operators])
            ode = ods.ODE(self.dfdt, ics=ics)
            x, f = ode.solve(**kwargs, args = ops, mask=self.apply_bcs)
                
        return x, f.flatten().reshape((*self.grid.shape, x.shape[0]), order='F')
    
    def get_ScalarField(self, name: str, t, dt, method='RK4', rtol=0., max_frames=-1, display=True, acc=1, fd='central'):
        '''
        general method, even for non linear problems. I need to get the varnames when i have many operators.
        for linear IVP problems this is easy, it is just self.op.varnames because there is only one op.
        '''
        v: list[sym.Variable] = []
        for arg in self.operators:
            for x in arg.variables:
                if x not in v:
                    v.append(x)
        variables: list[sym.Variable] = tools.sort(v, [x.axis for x in v])[0]
        names = [x.name for x in variables]
        t, f = self.solve(t=t, dt=dt, method=method, rtol=rtol, max_frames=max_frames, display=display, acc=acc, fd=fd)

        return fields.ScalarField(f, self.grid*grids.Unstructured1D(t), name, *names, 't')


class LinearIVP(IVP):

    def __init__(self, bcs: bounds.GroupedBcs, grid: grids.Grid, operator: sym.Expr):
        super().__init__(bcs, grid, operator)
        self.ivp_op = operator

    @abstractmethod
    def dfdt(self, t, f: np.ndarray, op: cached.CachedOperator)->np.ndarray:
        pass


class InhomLinearIVP(LinearIVP):

    def __init__(self, bcs: bounds.GroupedBcs, grid: grids.Grid, operator: sym.Expr, source: Callable[..., np.ndarray]):
        super().__init__(bcs, grid, operator)
        self._src = source

    def src(self, t):
        '''
        if src is independent of time, simplify stuff
        '''
        return self._src(*self.grid.x_mesh(), t).flatten(order='F')

    def dfdt(self, t, f: np.ndarray, op: cached.CachedOperator)->np.ndarray:
        '''
        For autonomous IVP's
        '''
        return op.matrix(self.grid, t).dot(f) + self.src(t)


class HomLinearIVP(LinearIVP):


    def dfdt(self, t, f: np.ndarray, op: cached.CachedOperator)->np.ndarray:
        '''
        For non-autonomous IVP's
        '''
        return op.matrix(self.grid, t).dot(f)


class HomLinearIBVP(HomLinearIVP):

    '''
    Homogeneous linear initial value problem with homogeneous boundary conditions
    '''
    def __init__(self, bcs: bounds.GroupedBcs, grid: grids.Grid, operator: sym.Expr, coef = 1.0):
        if not bcs.is_homogeneous:
            raise ValueError('HomLinearIBVP needs homogeneous boundary conditions')
        super().__init__(bcs, grid, (coef*operator).expand())
        self.eigproblem = linalg.EigProblem(operator, bcs, self.grid)
        self._coef = coef

    def solve(self, t, dt, method='RK4', rtol=0., max_frames=-1, display=True, acc=1, fd='central'):
        if method == 'propagator':
            self.arm_propagator()
            t_arr = np.append(np.arange(0, t, dt), [t])
            return t_arr, self.propagate(t_arr)
        else:
            return super().solve(t, dt, method, rtol, max_frames, display, acc, fd)

    def set_ics(self, f0: Callable[..., np.ndarray]):
        self.f0 = f0
        self.propagator = Propagator(self.eigproblem, f0, self._coef)

    def arm_propagator(self):
        self.propagator.construct()

    def propagate(self, t: float|np.ndarray):
        return self.propagator.apply(t)


class IVPsystem2D(ABC):
    '''
    For a system of 2 IVP's:

    Let du/dt = F(t, u, v)
        dv/dt = G(t, u, v)

        where u = u(x, y, t) and v = v(x, y, t)

        Then we define f as an array with 2 rows, f = [u, v]
        Similarly, dfdt = [dudt, dvdt]

    '''
    def __init__(self, bcs1: bounds.GroupedBcs, bcs2: bounds.GroupedBcs, grid: grids.Grid, *operators: sym.Expr):

        self.bcs1 = bcs1
        self.bcs2 = bcs2
        bcs1.apply_grid(grid)
        bcs2.apply_grid(grid)
        if bcs1.grid != bcs2.grid:
            raise ValueError('The two given bcs have different periodicity in some axis')
        self.grid = bcs1.grid
        self.joker = np.zeros(shape=(2, self.grid.n))
        self.operators = operators

    @abstractmethod
    def dfdt(self, t, f, *ops: cached.CachedOperator)->np.ndarray:
        pass

    def set_ics(self, u0: Callable[..., np.ndarray], v0: Callable[..., np.ndarray]):
        self.u0 = u0
        self.v0 = v0

    def apply_bcs(self, t, f: np.ndarray):
        s = np.zeros(shape=(2, self.grid.n))
        s[0, :] = self.bcs1.apply(f[0], t)
        s[1, :] = self.bcs2.apply(f[1], t)
        return s
    
    def solve(self, t, dt, method='RK4', rtol=0., max_frames=-1, display=True, acc=1, fd='central'):
        f0 = self.joker.copy()
        f0[0, :] = self.bcs1.discretize(self.u0, 0)
        f0[1, :] = self.bcs2.discretize(self.v0, 0)
        ics = (0, f0)
        kwargs = dict(t=t, dt=dt, method=method, rtol=rtol, max_frames=max_frames, display=display)
        ops = tuple([cached.cache_operator(op, self.grid, acc=acc, fd=fd) for op in self.operators])
        ode = ods.ODE(self.dfdt, ics=ics)
        x, f = ode.solve(**kwargs, args=ops, mask = self.apply_bcs)
        u = f[:, 0, :]
        v = f[:, 1, :]

        u = u.flatten().reshape((*self.grid.shape, x.shape[0]), order='F')
        v = v.flatten().reshape((*self.grid.shape, x.shape[0]), order='F')

        return x, np.array([u, v])


class Wave(IVPsystem2D, HomLinearIVP):
    
    def __init__(self, c: sym.Expr, bcs: bounds.GroupedBcs, grid: grids.Grid):
        if grid.nd == 1:
            x = sym.Variable('x')
            Laplacian = sym.Diff(x)**2
        elif grid.nd == 2:
            x, y = sym.variables('x y')
            Laplacian = sym.Diff(x)**2 + sym.Diff(y)**2
        HomLinearIVP.__init__(self, bcs, grid, c**2*Laplacian)
        IVPsystem2D.__init__(self, bcs, bcs.diff(), grid, c**2*Laplacian)

    def dfdt(self, t, f, laplacian: cached.CachedOperator)->np.ndarray:
        return np.array([f[1], laplacian.matrix(self.grid, t).dot(f[0])])
    
    def solve(self, t, dt, method='RK4', rtol=0., max_frames=-1, display=True, acc=1, fd='central'):
        x, (f, df) = IVPsystem2D.solve(self, t, dt, method, rtol, max_frames, display, acc, fd)
        return x, f
