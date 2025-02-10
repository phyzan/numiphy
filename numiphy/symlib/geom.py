from __future__ import annotations
from ..findiffs import grids
from ..toolkit import tools
from . import expressions as sym
import numpy as np
from typing import Callable
from typing import Iterable
import scipy.sparse as sp
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import math




# class Vector2D:


#     def __init__(self, x, y):
#         self.x = float(x)
#         self.y = float(y)

#     def __neg__(self):
#         return Vector2D(-self.x, -self.y)

#     def __add__(self, other: Vector2D):
#         return Vector2D(self.x+other.x, self.y+other.y)

#     def __sub__(self, other: Vector2D):
#         return Vector2D(self.x-other.x, self.y-other.y)

#     def __mul__(self, other):
#         if isinstance(other, (int, float)):
#             return Vector2D(other*self.x, other*self.y)
#         elif isinstance(other,Vector2D):
#             return self.x*other.x + self.y*other.y
#         else:
#             raise ValueError(f"Cannot multiply Vector2D with object of type {other.__class__}")
    
#     def __rmul__(self, other):
#         return self*other

#     def norm(self):
#         return math.sqrt(self.x**2+self.y**2)

#     def unit(self):
#         norm = self.norm()
#         return Vector2D(self.x/norm, self.y/norm)

#     def cross(self, other: Vector2D):
#         return self.x*other.y - self.y*other.x
    
#     def cross_z(self):
#         return Vector2D(self.y, -self.x)

#     def toarray(self):
#         return np.array([self.x, self.y])

#     def __repr__(self):
#         r = [self.x, self.y]
#         r = [int(i) if int(i)==i else i for i in r]
#         return '({0}, {1})'.format(*r)


class GeomObject(ABC):

    nd: int
    dof: int
    
    @abstractmethod
    def __eq__(self, other: GeomObject):
        pass

    @abstractmethod
    def nodes(self, grid: grids.Grid):
        pass

    @abstractmethod
    def param_values(self, grid: grids.Grid)->list:
        pass
    
    @abstractmethod
    def coords(self, *u)->np.ndarray:
        pass
    
    @abstractmethod
    def normal_vector(self, *u):
        pass

    def nods(self, grid: grids.Grid):
        nods = []
        for node in self.nodes(grid):
            nods.append(grid.flatten_index(node))
        return np.array(nods, dtype=int)

    def identity_matrix(self, grid: grids.Grid):
        m = grid.empty_matrix()
        for node in self.nodes(grid):
            m += grid.identity_element(node)
        return m
    
    def plot(self, grid: grids.Grid):
        self._assert_compatibility(grid)
        fig, ax = plt.subplots()
        ax.set_xlim(*grid.limits[0])
        if grid.nd == 1:
            for node in self.nodes(grid):
                ax.scatter(*grid.coords(node), 0, s=3, c='k')
        if grid.nd == 2:
            ax.set_ylim(*grid.limits[1])
            for node in self.nodes(grid):
                ax.scatter(*grid.coords(node), s=3, c='k')
        else:
            raise NotImplementedError('Plotting in 3D has been implemented yet')

        return fig, ax

    def fill_arr(self, arr, f, grid: grids.Grid, *args):
        '''
        len(arr) should be == grid.n
        '''
        for node, param_value in zip(self.nodes(grid), self.param_values(grid)):
            nod = grid.flatten_index(node)
            arr[nod] += f(*param_value, *args)

    def evaluate_on(self, f, grid: grids.Grid, *args):
        arr = np.zeros(grid.n)
        return self.fill_arr(arr, f, grid, *args)

    def normaldiff_matrix(self, grid: grids.Grid, order: int, fd: str, acc: int)->sp.csr_matrix:
        m = grid.empty_matrix()
        for u in self.param_values(grid):
            nod = grid.node(*self.coords(*u))
            vec = self.normal_vector(*u)
            m += grid.directional_diff_element(node=nod, direction=vec, order=order,fd=fd, acc=acc)
        return m

    def _assert_compatibility(self, grid: grids.Grid):
        if self.nd != grid.nd:
            raise ValueError(f'{self.__class__} and grid lie on different dimensions')



class Point(GeomObject):

    dof = 0

    def __init__(self, *coords):
        if not all([tools.isnumeric(n) for n in coords]):
            raise ValueError('Point arguments must be numerical coordinates')
        self._coords = coords
        self.nd = len(coords)

    def __repr__(self):
        return f'Point{self._coords}'
    
    def __eq__(self, other: GeomObject):
        if isinstance(other, Point):
            return self._coords == other._coords
        else:
            return False

    def nodes(self, grid: grids.Grid):
        self._assert_compatibility(grid)
        return grid.node(*self._coords),
    
    def param_values(self, grid: grids.Grid):
        return [()]

    def coords(self):
        return np.array(self._coords)
    
    def normal_vector(self):
        if self.nd != 1:
            raise ValueError('A point must lie on a 1d line so that it has a uniquely defined normal Vector2D')
        return np.array([1.])


class Line(GeomObject):

    dof = 1

    def __init__(self, x: list[sym.Expr], lims: tuple[float, float]):
        _var = None
        for xi in x:
            if len(xi.variables) > 1:
                raise ValueError("All expressions must have at most one variable")
            elif len(xi.variables) == 1:
                if _var is not None:
                    assert xi.variables[0] == _var
                else:
                    _var = xi.variables[0]
        if _var is None:
            raise ValueError("At least one expression needs to contain exactly one variable")
        
        _x_callable = [xi.lambdify([_var]) for xi in x]
        _xdot_callable = [xi.diff(_var).lambdify([_var]) for xi in x]
        self.Args = (x.copy(), lims, _var, _x_callable, _xdot_callable)

    @property
    def _x(self)->list[sym.Expr]:
        return self.Args[0]
    
    @property
    def lims(self)->tuple[float, float]:
        return self.Args[1]

    @property
    def _var(self)->sym.Variable:
        return self.Args[2]
    
    @property
    def _x_funcs(self):
        return self.Args[3]
    
    @property
    def _xdot_funcs(self):
        return self.Args[4]
    
    @property
    def nd(self)->int:
        return len(self._x)

    def __eq__(self, other: GeomObject):
        if isinstance(other, Line):
            return self.Args == other.Args
        else:
            return False
        
    def __repr__(self):
        return f'Line({self.nd}d-space, u = {self.lims})'

    def coords(self, u):
        return np.array([xi(u) for xi in self._x_funcs])
    
    def tangent(self, u):
        return np.array([xi(u) for xi in self._xdot_funcs])
    
    def param_values(self, grid: grids.Grid)->Iterable:
        self._assert_compatibility(grid)
        x = self.coords
        a, b = self.lims
        nod = grid.node(*x(a))
        nods = [nod]
        u_arr = [(a,)]
        neighbors = grid.neighboring_nodes(nod)
        while grid.node(*x(self.lims[1])) != nod or len(u_arr) == 1:
            u = (a+b)/2
            nod = grid.node(*x(u))
            if nod in neighbors:
                if nod in nods:
                    break
                else:
                    neighbors = grid.neighboring_nodes(nod)
                    b = self.lims[1]
                    u_arr.append((u,))
                    nods.append(nod)
            elif nod == nods[-1]:
                a = u
            else:
                b = u
        return u_arr
    
    def nodes(self, grid: grids.Grid):
        nods = []
        for u in self.param_values(grid):
            nods.append(grid.node(*self.coords(*u)))
        return nods

    def normal_vector(self, u):
        #The line must lie on a 2d plane so that is has a uniquely defined normal Vector2D
        #Otherwise, normal_vector will raise an error
        tangent_vec = self.tangent(u)
        return normal_vector(tangent_vec)
        
    def reverse(self):
        return self.__class__(self._x, (self.lims[1], self.lims[0]))



class Line2D(Line):

    def __init__(self, x: sym.Expr, y: sym.Expr, lims: tuple[float, float]):
        super().__init__([x, y], lims)

    def x(self, u):
        return self._x_funcs[0](u)
    
    def y(self, u):
        return self._x_funcs[1](u)
    
    def xdot(self, u):
        return self._xdot_funcs[0](u)
    
    def ydot(self, u):
        return self._xdot_funcs[1](u)



class Surface(GeomObject):
    #TODO
    dof = 2

    pass


def normal_vector(*vecs):
    m = np.array(vecs).transpose()
    nd = len(vecs) + 1 #==len(vecs[i]) for all i
    if m.shape != (nd, nd-1):
        raise ValueError('You need n-1 vectors to generate a Vector2D perpendicular to them in an n-dimensional space')
    
    v = np.zeros(nd)
    m = np.array(vecs).transpose()
    for i in range(nd):
        v[i] = (-1)**i*np.linalg.det(np.vstack((m[:i], m[i+1:])))
    return v

class Parallelogram(Line2D):

    def __init__(self, x1, x2, y1, y2):
        self.x1, self.x2, self.y1, self.y2 = x1, x2, y1, y2
        self.a, self.b = 0, 4
        self.Lx = x2-x1
        self.Ly = y2-y1
        self.center = [(x1+x2)/2, (y1+y2)/2]

        x0, y0 = x1, y1
        a, b = self.Lx, self.Ly
        u = sym.Variable('u')
        x = sym.Piecewise((x0+u*a, u<1), (x0+a, u<2), (x0+a*(3-u), u<3), (x0, True))

        y = sym.Piecewise((y0, u<1), (y0+(u-1)*b, u<2), (y0+b, u<3), (y0+(4-u)*b, True))
        super().__init__(x, y, lims=(0, 4))

    def split(self)->list[Parallelogram]:
        a, b, c, d = self.x1, self.x2, self.y1, self.y2
        shapes = []
        xaxis = [(a, (a+b)/2), ((a+b)/2, b)]
        yaxis = [(c, (c+d)/2), ((c+d)/2, d)]
        for x in xaxis:
            for y in yaxis:
                shapes.append(Parallelogram(*x, *y))
        return shapes

def Circle(r: float, center: tuple[float])->Line:

    t = sym.Variable('t')

    x = center[0] + r*sym.cos(t)
    y = center[1] + r*sym.sin(t)

    return Line2D(x, y, (0, 2*math.pi))


def Square(a: float, start: tuple[float]):
    return Parallelogram(start[0], start[0]+a, start[1], start[1]+a)

'''

TODO:

Everything returns a vector-like object:

Vector1D inherits from vector and float
Vector2D ready
Vector3D do it
VectorNd for higher dimensions


.coords returns Vector
.normal_vector returns Vector
.tangent returns Vector


'''