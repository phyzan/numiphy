from __future__ import annotations
from ..findiffs import grids
from .. symlib import geom
from ..findiffs import finitedifferences as fds
from ..toolkit import tools
from ..symlib import symcore as ops
from typing import Callable, Literal, Generator
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from abc import ABC, abstractmethod
from functools import cached_property


_bound = {'lower': 0, 'upper': 1, 0: 'lower', 1: 'upper'}


def reduced_form(m: sp.csr_matrix, nods: list[int])->sp.csr_matrix:
    '''
    This function empties the columns and rows that correspond to boundary conditions.
    If for example, the i-th row contains a boundary condition, all the other rows
    will be subtracted a proper multiple of the i-th row so that their i-th column-element will be zero
    Parameters
    --------------
    m: sparse square matrix
    rows: rows with boundary conditions
    '''
    n = m.shape[0]
    q = m.tolil()
    for i in nods:
        val = m[i, i]
        for j in range(n):
            if i != j:
                if q[j, i] != 0:
                    q[j, :] = q[j, :] - q[j, i]/val * q[i, :]
                    q[j, i] = 0 #explicitly set to avoid roundoff inaccuracies
    mask = np.ones(m.shape[0], dtype=bool)
    mask[nods] = False
    reduced: sp.lil_matrix = q[mask, :][:, mask]

    return reduced.tocsr()


class BoundaryConditions(ABC):

    grid: grids.Grid

    def Lhs(self, op: ops.Expr = ops.S.One, reduced = False, acc=1, fd='central')->sp.csr_matrix:###
        m = op.matrix(self.grid, acc=acc, fd=fd)
        m = self.mask().dot(m) + self.lhs()
        if reduced:
            return self.reduced_matrix(m)
        else:
            return m

    def Rhs(self, arr: np.ndarray, *args)->np.ndarray:
        return self.mask().dot(arr) + self.rhs(*args)

    def mask(self):
        diag = np.ones(self.grid.n, dtype=int)
        for nod in self.nods_generator():
            diag[nod] = 0
        return sp.dia_matrix((diag, 0), shape=(self.grid.n, self.grid.n))

    def rhs(self, *args)->np.ndarray:
        arr = np.zeros(self.grid.n)
        self.fill_rhs(arr, *args)
        return arr
    
    def reserved_nods(self):
        all_nods = list(self.nods_generator())
        res = list(set(all_nods))
        res.sort()
        return res
    
    def reduced_matrix(self, m: sp.csr_matrix):
        return reduced_form(m, self.reserved_nods())
    
    def reduced_array(self, arr: np.ndarray):
        bcs_nods = self.reserved_nods()
        mask = np.ones(self.grid.n, dtype=bool)
        mask[bcs_nods] = False
        return arr[mask]
    
    def bvp_matrix(self):
        return self.Lhs()

    def apply(self, arr: np.ndarray, *args):
        '''
        replace the bcs entries with the right hand side and solve the linear system
        '''
        if self.is_dirichlet and self.is_homogeneous:
            return self.mask().dot(arr)
        elif self.is_dirichlet:
            return self.Rhs(arr, *args)
        else:
            return spl.spsolve(self.bvp_matrix(), self.Rhs(arr, *args))
    
    def expand(self, arr: np.ndarray):
        nods = self.reserved_nods()
        if arr.ndim > 2:
            raise NotImplementedError('Can only expand a reduced array or an array of arrays')
        if arr.shape[-1] != self.grid.n - self.N_nods:
            raise ValueError('The given array(s) do not have the proper length to be expanded')
        nods = self.reserved_nods()
        mask = np.ones(self.grid.n, dtype=bool)
        mask[nods] = False
        if arr.ndim == 1:
            new_arr = np.zeros(self.grid.n, dtype=arr.dtype)
            new_arr[mask] = arr
        else:
            new_arr = np.zeros((arr.shape[0], self.grid.n), dtype=arr.dtype)
            new_arr[:, mask] = arr
        
        return new_arr


    def fill_rhs(self, arr: np.ndarray, *args):
        pass

    def discretize(self, f0: Callable[..., np.ndarray], *args, reduced=False):
        if reduced and not self.is_homogeneous:
            raise NotImplementedError('Can only reduce an array when the boundary conditions are homogeneous')

        f = f0(*self.grid.x_mesh()).flatten(order='F')
        f = self.apply(f, *args)
        if reduced:
            return self.reduced_array(f)
        else:
            return f

    @property #must make faster
    def N_nods(self):
        return len(self.reserved_nods())
    
    @abstractmethod
    def as_homogeneous(self)->BoundaryConditions:
        pass

    @abstractmethod
    @cached_property
    def is_dirichlet(self)->bool:
        pass

    @abstractmethod
    def nods_generator(self)->Generator[int]:###
        pass

    @abstractmethod
    @cached_property
    def is_homogeneous(self):
        pass
    
    @abstractmethod
    def apply_grid(self, grid: grids.Grid):###
        pass

    @abstractmethod
    def lhs(self)->sp.csr_matrix:###
        pass

    @abstractmethod
    def newcopy(self):
        '''
        Creates new copy that has no grid yet
        '''
        pass

    @abstractmethod
    def diff(self)->BoundaryConditions:
        pass


class RobinBcs(BoundaryConditions):

    coefs: tuple[float]

    @cached_property
    def is_dirichlet(self):
        return self.coefs[0] == 1 and all([c == 0 for c in self.coefs[1:]])


class ExteriorBoundaryCondition(RobinBcs):

    loc: Literal['upper', 'lower']
    axis: int

    def __init__(self, a: float, b: float, loc: Literal['lower', 'upper'], axis=0):
        self.coefs = (a, b)
        self.loc = loc
        self.axis = axis

    def apply_grid(self, grid: grids.Grid):
        if grid.periodic[self.axis]:
            raise ValueError('Grid is periodic in the axis of an exterior boundary condition')
        self.grid = grid

    def nods_generator(self):
        for node in self.grid.edge(self.axis, _bound[self.loc], corners=True):
            yield self.grid.flatten_index(node)

    def lhs(self):
        a, b = self.coefs
        n = self.grid.shape[self.axis]
        nod = _bound[self.loc]*(n-1)
        dirichlet1D = tools.eye([nod], n)
        dirichlet = tools.generalize_operation(self.axis, self.grid.shape, matrix=dirichlet1D)
        neumann1D = self.grid[self.axis].findiff.operator_element(nod, 1, 1)
        neumann = tools.generalize_operation(self.axis, self.grid.shape, matrix=neumann1D)
        
        return a*dirichlet + b*neumann
    
    def as_homogeneous(self):
        return HomExteriorBoundaryCondition(*self.coefs, self.loc, self.axis)
    


class HomExteriorBoundaryCondition(ExteriorBoundaryCondition):########

    '''
    a*f(x, y) + b*d/dn f(x, y) = 0
    '''
    @cached_property
    def is_homogeneous(self):
        return True

    def newcopy(self, axis: int = None):
        if axis is None:
            axis = self.axis
        return HomExteriorBoundaryCondition(*self.coefs, self.loc, axis)
    
    def diff(self):
        return self.newcopy()


class InhomExteriorBoundaryCondition(ExteriorBoundaryCondition):########

    def __init__(self, a: float, b: float, loc: Literal['lower', 'upper'], value: Callable[..., np.ndarray], jac = None, axis=0):
        '''
        value must be callable and vectorized. a numpy meshgrid will be passed into the function
        '''
        super().__init__(a, b, loc, axis)
        self.func = value
        self.loc = loc
        self.jac = jac

    def fill_rhs(self, arr: np.ndarray, *args):
        arr[self.reserved_nods()] += self.func(*self.grid.x_mesh(self.axis), *args).flatten(order='F')

    @cached_property
    def is_homogeneous(self):
        return False
    
    def newcopy(self, axis: int = None):
        if axis is None:
            axis = self.axis
        return InhomExteriorBoundaryCondition(*self.coefs, self.loc, self.func, self.jac, axis)

    def diff(self):
        return InhomExteriorBoundaryCondition(*self.coefs, self.loc, value = self.jac, axis=self.axis)


class InteriorBoundaryCondition(RobinBcs):

    obj: geom.GeomObject

    def __init__(self, obj: geom.GeomObject, a, b, c):
        self.obj = obj
        self.coefs = (a, b, c)

    def apply_grid(self, grid: grids.Grid):
        self.grid = grid
        self.nods = self.obj.nods(grid).tolist()

    def lhs(self):
        a, b, c = self.coefs
        
        Id = self.obj.identity_matrix(self.grid)
        dn_up   = self.obj.normaldiff_matrix(self.grid, order=1, fd='forward', acc=1)
        dn_down = self.obj.normaldiff_matrix(self.grid, order=1, fd='backward', acc=1)
        return a*Id + b*dn_up + c*dn_down

    def is_symmetric(self):
        return self.coefs[1] + self.coefs[2] == 0

    def nods_generator(self):
        for nod in self.nods:
            yield nod

    def as_homogeneous(self):
        return HomInteriorBoundaryCondition(self.obj, *self.coefs)


class HomInteriorBoundaryCondition(InteriorBoundaryCondition):#############

    @cached_property
    def is_homogeneous(self):
        return True
    
    def newcopy(self):
        return HomInteriorBoundaryCondition(self.obj, *self.coefs)

    def diff(self):
        return self.newcopy()


class InhomInteriorBoundaryCondition(InteriorBoundaryCondition):###############
    def __init__(self, obj: geom.GeomObject, a, b, c, value: Callable[..., np.ndarray], jac = None):
        super().__init__(obj, a, b, c)
        self.func = value
        self.jac = jac

    def apply_grid(self, grid: grids.Grid):
        super().apply_grid(grid)
        self.params = self.obj.param_values(grid)

    def fill_rhs(self, arr: np.ndarray, *args):
        arr[self.reserved_nods()] += np.array([self.func(*p, *args) for p in self.params]).flatten(order='F')

    @cached_property
    def is_homogeneous(self):
        return False
    
    def newcopy(self):
        return InhomInteriorBoundaryCondition(self.obj, *self.coefs, self.func, self.jac)

    def diff(self):
        return InhomInteriorBoundaryCondition(self.obj, *self.coefs, value=self.jac)


class AxisBcs(BoundaryConditions):
    axis: int
    pass


class SingleAxisBcs(AxisBcs):

    def __init__(self, cond: StandardCondition, loc: str, axis=0):
        self.cond = cond.apply_exterior(axis, loc)
        self.axis = axis
        self._cond = cond
        self.loc = loc

    def apply_grid(self, grid: grids.Grid):
        if grid.periodic[self.axis]:
            grid = grid.as_not_periodic(self.axis)
        
        self.cond.apply_grid(grid)
        self.grid = grid

    def lhs(self):
        return self.cond.lhs()
    
    def nods_generator(self):
        for nod in self.cond.nods_generator():
            yield nod

    def fill_rhs(self, arr: np.ndarray, *args):
        self.cond.fill_rhs(arr, *args)

    def newcopy(self, axis:int = None):
        if axis is None:
            axis = self.axis
        return SingleAxisBcs(self._cond, self.loc, axis)
    
    @cached_property
    def is_homogeneous(self):
        return self.cond.is_homogeneous
    
    @cached_property
    def is_dirichlet(self):
        return self.cond.is_dirichlet
    
    def as_homogeneous(self):
        return SingleAxisBcs(self._cond.as_homogeneous(), self.loc, self.axis)
    
    def diff(self):
        return SingleAxisBcs(self._cond.diff(), self.loc, self.axis)


class StandardAxisBcs(AxisBcs):

    low: ExteriorBoundaryCondition
    up: ExteriorBoundaryCondition

    def __init__(self, lower: StandardCondition, upper: StandardCondition=None, axis=0):

        self.low = lower.apply_exterior(axis, 'lower')
        self.up = upper.apply_exterior(axis, 'upper')
        self.axis = axis

        self._lower = lower
        self._upper = upper


    def apply_grid(self, grid: grids.Grid):
        if grid.periodic[self.axis]:
            grid = grid.as_not_periodic(self.axis)
        
        self.low.apply_grid(grid)
        self.up.apply_grid(grid)
        self.grid = grid

    def lhs(self):
        return self.low.lhs() + self.up.lhs()
    
    def nods_generator(self):
        for loc in [self.low, self.up]:
            for nod in loc.nods_generator():
                yield nod

    def fill_rhs(self, arr: np.ndarray, *args):
        self.low.fill_rhs(arr, *args)
        self.up.fill_rhs(arr, *args)

    def newcopy(self, axis:int = None):
        if axis is None:
            axis = self.axis
        return StandardAxisBcs(self._lower, self._upper, axis)
    
    @cached_property
    def is_homogeneous(self):
        return self.low.is_homogeneous and self.up.is_homogeneous
    
    @cached_property
    def is_dirichlet(self):
        return self.low.is_dirichlet and self.up.is_dirichlet
    
    def as_homogeneous(self):
        return StandardAxisBcs(self._lower.as_homogeneous(), self._upper.as_homogeneous(), self.axis)
        
    def diff(self):
        return StandardAxisBcs(self._lower.diff(), self._upper.diff(), self.axis)


class PeriodicBcs(AxisBcs):

    def __init__(self, axis=0):
        self.axis = axis

    def Lhs(self, op: ops.Expr, reduced=False, acc=1, fd='central'):
        return op.matrix(self.grid, acc=acc, fd=fd)
    
    def Rhs(self, arr: np.ndarray, *args):
        return arr
        
    def apply_grid(self, grid: grids.Grid):
        if not grid.periodic[self.axis]:
            self.grid = grid.as_periodic(self.axis)
        else:
            self.grid = grid

    def lhs(self):
        return self.grid.empty_matrix()
    
    def nods_generator(self):
        return iter([])
    
    @cached_property
    def is_homogeneous(self):
        return True
    
    @cached_property
    def is_dirichlet(self):
        return False
    
    def fill_rhs(self, arr: np.ndarray, *args):
        pass

    def newcopy(self, axis: int = None):
        if axis is None:
            axis = self.axis
        return PeriodicBcs(axis)
    
    def as_homogeneous(self):
        return self.newcopy()
    
    def diff(self):
        return self.newcopy()


class GroupedBcs(BoundaryConditions):
    '''
    If even one condition in time-dependent, all of them need to have that parameter.
    '''
    
    def __init__(self, *bcs: BoundaryConditions):
        if isinstance(bcs[0], ExteriorBoundaryCondition):
            raise ValueError('Boundary conditions for the exterior of the grid must not be given explicitly. They must be given as an AxisBcs object for each axis')
        if not isinstance(bcs[0], AxisBcs):
            raise ValueError('The boundary conditions for the exterior of the grid need to be given before any other internal boundary conditions')
        
        ext_bcs: list[AxisBcs] = []
        int_bcs: list[InteriorBoundaryCondition] = []

        i = 0
        while i < len(bcs):
            if isinstance(bcs[i], AxisBcs):
                if bcs[i].axis != i:
                    raise ValueError('The AxisBcs objects must be given in ascending order with respect to their axis, starting from 0 and increasing by 1.')
                ext_bcs.append(bcs[i])
                i += 1
            else:
                break

        self.nd = i

        for j in range(i, len(bcs)):
            if isinstance(bcs[j], InteriorBoundaryCondition):
                int_bcs.append(bcs[j])
            else:
                raise ValueError('Only interior boundary conditions must be given after all exterior boundary conditions have been given')

        for i in range(len(int_bcs)-1):
            for j in range(i+1, len(int_bcs)):
                if int_bcs[i].obj == int_bcs[j].obj:
                    raise ValueError('Multiple conditions given for the same boundary')
        

        self.bcs = bcs
        self.ext_bcs: tuple[AxisBcs] = tuple(ext_bcs)
        self.int_bcs = tuple(int_bcs)
    

    def apply_grid(self, grid: grids.Grid):
        #needs to change so that grid matches periodicity
        if grid.nd != self.nd:
            raise ValueError(f'This GroupedBcs object contains boundary conditiond for {self.nd} dimensions, while the given grid is {grid.nd}-dimensional')
        
        periodicity = tuple([isinstance(obj, PeriodicBcs) for obj in self.ext_bcs])
        if periodicity != grid.periodic:
            grid = grid.with_periodicity(*periodicity)
        
        self.grid = grid
        for bc in self.bcs:
            bc.apply_grid(grid)

        self._mask = super().mask()
        self._bvp_matrix = super().bvp_matrix()
        self._reserved_nods = super().reserved_nods()

    def bvp_matrix(self):
        return self._bvp_matrix

    def mask(self):
        return self._mask
    
    def reserved_nods(self):
        return self._reserved_nods.copy()
    
    def fill_rhs(self, arr: np.ndarray, *args):
        for cond in self.bcs:
            cond.fill_rhs(arr, *args)

    def lhs(self):
        m = self.grid.empty_matrix()
        for cond in self.bcs:
            m += cond.lhs()
        return m
    
    def nods_generator(self):
        for cond in self.bcs:
            for nod in cond.nods_generator():
                yield nod
    
    def newcopy(self):
        return GroupedBcs(*[cond.newcopy() for cond in self.bcs])

    @cached_property
    def is_homogeneous(self):
        return all([bc.is_homogeneous for bc in self.bcs])
    
    @cached_property
    def is_dirichlet(self):
        return all([bc.is_dirichlet for bc in self.bcs])
    
    def lhs_is_id(self):
        for bc in self.ext_bcs:
            if not isinstance(bc, PeriodicBcs) and not bc.is_dirichlet:
                return False
        
        for bc in self.int_bcs:
            if not bc.is_dirichlet:
                return False
            
        return True
    
    def as_homogeneous(self):
        return StaticBcs(*[bc.as_homogeneous() for bc in self.bcs])
    
    def insert(self, arr: np.ndarray):
        '''
        apply homogeneous boundary conditions on a reduced array
        '''
        
        new_arr = self.expand(arr)
        if self.is_homogeneous and self.is_dirichlet:
            return new_arr
        elif len(arr.shape) == 1:
            return self.apply(new_arr)
        else:
            for i in range(arr.shape[0]):
                new_arr[i, :] = self.apply(new_arr[i, :])
            return new_arr
        
    def diff(self):
        return GroupedBcs(*[cond.diff() for cond in self.bcs])


class StaticBcs(GroupedBcs):

    _rhs: np.ndarray

    def apply_grid(self, grid: grids.Grid):
        super().apply_grid(grid)
        self._rhs = super().rhs()

    def rhs(self, *args):
        return self._rhs
    
    def diff(self):
        return self.as_homogeneous()


'''

Different class for IVP and IVPsystem

'''

class StandardCondition:

    coefs: tuple[float]

    '''
    Not instanciated
    '''

    def __init__(self, value: float|Callable[..., float] = None, jac = None):
        if tools.isnumeric(value):
            def val(*x):
                if x:
                    return 0*x[0] + value
                else:
                    return np.float64(value)
            self.value = val
        else:
            self.value = value
        self.jac = jac

    def apply_exterior(self, axis: int, loc: Literal['lower', 'upper']):
        if self.value is None:
            return HomExteriorBoundaryCondition(*self.coefs[:2], loc=loc, axis=axis)
        else:
            return InhomExteriorBoundaryCondition(*self.coefs[:2], loc=loc, value=self.value, jac=self.jac, axis=axis)

    def apply(self, obj: geom.GeomObject):
        if self.value is None:
            return HomInteriorBoundaryCondition(obj, *self.coefs)
        else:
            return InhomInteriorBoundaryCondition(obj, *self.coefs, self.value, self.jac)
        
    def as_homogeneous(self):...
    
    def diff(self):
        return self.__class__(self.jac)


class Robin(StandardCondition):

    def __init__(self, coefs, value: float|Callable[..., float] = None, jac = None):
        self.coefs = coefs
        super().__init__(value, jac)

    def as_homogeneous(self):
        return Robin(self.coefs)

    def diff(self):
        return Robin(self.coefs, self.jac)


class Dirichlet(StandardCondition):

    coefs = (1, 0, 0)


class Neumann(StandardCondition):

    coefs = (0, 1, 0)




def ivp_operator(x: list[float]):
    '''
    Consider the initial value problem f(0) = a, f'(0) = b, f''(0) = c
    These conditions are sufficient to calculate the function f
    at the first 3 time steps, separated by equal lengths dt.
    Using 1st order forward finite differences, the linear system that should be solved is

    1/dt**0  | 1, 0, 0|   |  f(0) |      | f(0)  |
    1/dt**1  |-1, 1, 0| * | f(dt) | =    | f'(0) |
    1/dt**2  | 1,-2, 1|   |f(2*dt)|      | f''(0)|

    Then, f(0), f(dt), f(2*dt) can be calculated by applying the inverse matrix on the l.h.s
    on the initial conditions. This function calculates that inverse matrix


    Parameters
    --------------
    order: Order of the highest derivative in the initial conditions
    dx: Step of integration
    '''
    order = len(x)
    A = np.zeros((order, order))
    for i in range(order):
        fd = fds.FinDiff(x)
        A[i][:i+1] = fd.coefs(0, i, 1, fd='forward')
    return np.linalg.inv(A)


def Lower(cond: StandardCondition, axis=0):

    return SingleAxisBcs(cond, 'lower', axis)

def Upper(cond: StandardCondition, axis=0):

    return SingleAxisBcs(cond, 'upper', axis)