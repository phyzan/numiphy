from __future__ import annotations
import itertools
import numpy as np
import scipy.sparse as sp
from ..toolkit import tools
from . import finitedifferences as fds
from typing import Dict


class Grid:
    shape: tuple[int] = ()
    shape_full: tuple[int] = ()
    limits: tuple[list[float]] = ()
    periodic: tuple[bool] = ()
    nd: int = 0
    x_full: tuple[np.ndarray] = ()
    x: tuple[np.ndarray] = ()
    n: int = 1
    n_full: int = 1
    n_bounds: int
    n_goodbounds: int
    n_corners: int
    grids: tuple[Grid1D]

    def __eq__(self, other: Grid):
        return self.grids == other.grids

    def __getitem__(self, i: int):
        return self.grids[i]
    
    def __mul__(self, other: Grid)->NdGrid:
        return NdGrid(*self.grids, *other.grids)
    
    def __pow__(self, n: int)->NdGrid:
        return NdGrid(*(n*list(self.grids)))

    def node(self, *coords: float)->tuple[int]:
        pass
    
    def coords(self, index: tuple[int]):
        '''
        Arguments
        ------------
        index (tuple): The index of the grid in the format (i, j, k)
        '''
        return tuple(float(self.x[axis][index[axis]]) for axis in range(self.nd))

    def x_mesh(self, *axis):
        '''
        Create meshgrid of grid variables to passed into a funtion.
        e.g

        In 3D, if f = f(x, y, z), then f(*self.x_mesh()).flatten() can be used
        in a Function operator, or for plotting (when reshaped)

        Parameters
        -------------
        axis: axis to not be included in the meshgrid.

        '''

        if axis:
            x = ()
            for i in range(self.nd):
                if i not in axis:
                    x += (self.x[i],)
        else:
            x = self.x
        return np.meshgrid(*x, indexing='ij')

    def flatten_index(self, index: tuple[int]):
        return tools.flatten_index(index, self.shape, order='F')
    
    def tonode(self, i: int):
        '''
        Argument: Index of flattened grid
        Returns: Reshaped index
        '''
        return tools.reshape_toindex(i, self.shape, order='F')

    def nodes(self, edges: bool = True):
        '''
        Generator object that iterates through all (i, j, k) indices of the grid

        Arguments
        -------------
        edges (bool): Whether or not to include the exterior indices in the iteration

        Notes
        -------
        Total number will be self.n
        '''
        r = 1-edges
        ranges = [(r, self.shape[i]-1-r) for i in range(self.nd)]
        return tools.Iterator('F', *ranges)

    def coords_iterator(self):
        for node in self.nodes():
            yield self.coords(node)

    def edge(self, axis: int, bound: int, corners: bool):
        if corners is True:
            r = [(0, s-1) for s in self.shape]
        else:
            r = [(1, s-2) for s in self.shape]
        return tools.Iterator('F', *r[:axis], 2*[bound*(self.shape[axis]-1)], *r[axis+1:])

    def edges(self, corners: bool):
        '''
        Generator object that iterates through all boundary (i, j, k) indices of the grid


        Notes
        -------
        if corners is True:
            Total number is self.n_bounds
        if corners is False:
            Total number is self.n_goodbounds
        
        '''
        generators = []
        _all = [(0, s-1) for s in self.shape]
        _interior = [(1, s-2) for s in self.shape]
        _bound = [[2*[k*(s-1)] for k in range(2)] for s in self.shape]
        for d in range(self.nd-1, -1, -1):
            for k in range(2):
                if corners is True:
                    generators.append(tools.Iterator('F', *_interior[:d], _bound[d][k], *_all[d+1:]))
                else:
                    generators.append(tools.Iterator('F', *_interior[:d], _bound[d][k], *_interior[d+1:]))
        return itertools.chain(*generators)

    def corners(self):
        '''
        Notes
        ----------
        Total number is self.n_corners
        '''
        if self.nd == 1:
            return itertools.product([])
        else:
            _all = [(0, s-1) for s in self.shape]
            _interior = [(1, s-2) for s in self.shape]
            _bound = [[2*[k*(s-1)] for k in range(2)] for s in self.shape]
            generators = []
            for i in range(self.nd-1, 0, -1):
                for j in range(i-1, -1, -1):
                    for k1 in range(2):
                        for k2 in range(2):
                            generators.append(tools.Iterator('F', *_interior[:j],_bound[j][k1], *_interior[j+1:i],_bound[i][k2],*_all[i+1:]))
        return itertools.chain(*generators)
    
    def neighboring_nodes(self, node: tuple):
        '''
        Given a node, returns all neighboting nodes. e.g for a 2D grid, those will be 8 in total usually. In the edges, this will be 5, and in the corners 3.
        '''
        k = []
        for i in range(self.nd):
            if self.periodic[i] or 0 < node[i] < self.shape[i]-1:
                low = (self.shape[i]+node[i]-1) % self.shape[i]
                high = (node[i]+1) % self.shape[i]
                k.append([low, node[i], high])
            elif node[i] == 0:
                k.append([0, 1])
            else:
                k.append([self.shape[i]-2, self.shape[i]-1])
        nods = []
        for i in itertools.product(*k):
            if not all([i[j] == node[j] for j in range(self.nd)]):
                nods.append(tuple(i))
        return nods
    
    def empty_matrix(self):
        '''
        Creates an empty sparse matrix whose dimensions are those of a discretized linear operator that can operate in this grid.
        '''
        return sp.csr_matrix((self.n, self.n))

    def identity_element(self, node: tuple[int]):
        '''
        Identity matrix (sparse) with only a single entry in the row that corresponds to the given node (n-dimensional).

        Parameters
        --------------
        node: node in this n-dimensional space. The node must be a list of integer numbers, the indices in each dimension. e.g. for 3D node = (3, 8, 0)

        Returns
        --------------
        sparse matrix with a single element
        '''
        row = [self.flatten_index(node)]
        return sp.csr_matrix(([1], (row, row)), shape=(self.n, self.n))

    def is_uniform(self):
        return all([isinstance(g, Uniform1D) for g in self.grids]) 
    
    def as_not_periodic(self, *axis)->Grid:...

    def as_periodic(self, *axis)->Grid:...

    def with_periodicity(self, *Bool: bool)->Grid:...

    def partial_diff_element(self, node: tuple[int], axis: int, order: int, fd='central', acc=1):
        '''
        Similar to diff_element, but is generalized for multi-dimensional grids

        Parameters
        ------------

        node: node in this n-dimensional space, so that the differential operator-matrix will be evaluated only in the corresponding row
        axis: The *partial* differential operator must be created with respect to an axis
        order: order of differentiation
        acc: accuracy of finite differences used
        fd (0, 1, or -1): Type of finite differences used (see diff_element)

        Returns
        ------------
        sparse matrix with entries only in one row
        '''
        findiff = self.grids[axis].findiff
        offsets_1d = findiff.offsets(node[axis], order, acc, fd)
        nods_1d = (node[axis] + offsets_1d + self.shape[axis]) % self.shape[axis]

        nf = len(offsets_1d)
        rows = self.flatten_index(node) + np.zeros(nf, dtype=int)
        node = tuple(node)
        cols = []
        for i in range(nf):
            _node = node[:axis] + (nods_1d[i],) + node[axis+1:]
            cols.append(self.flatten_index(_node))
        
        vals = findiff.coefs_from_offsets(node[axis], order, offsets=offsets_1d)

        return sp.csr_matrix((vals, (rows, cols)), shape=(self.n, self.n))

    def directional_diff_element(self, node: tuple[int], direction: tuple[float], order: int=1, fd='central', acc: int=1):
        '''
        Similar to partial_diff_element. Specifically, this is a linear combination of partial_diff_elelement matrices of different axis, but all with entries in the same row, and with normalized coefficients along derived from a given direction.

        Parameters
        -------------
        node: node in this n-dimensional space, so that the differential operator-matrix will be evaluated only in the corresponding row
        direction: Direction along which the differential operator should operate (d/dn)
        order: order of differentiation
        acc: accuracy of finite differences
        fd (0, 1, or -1): Type of finite differences used (see diff_element)
        '''
        vec = direction/np.linalg.norm(direction)
        m = self.empty_matrix()
        kwargs = dict(node=node, order=order, acc=acc)
        for i in range(self.nd):
            m += vec[i] * self.partial_diff_element(**kwargs, axis=i, fd=fds.fd_map[fds.fd_map[fd]*tools.sign(vec[i])])
        return m
    
    def partialdiff_matrix(self, order: int, axis: int, acc=1, fd='central'):
        '''
        Function that creates sparse matrices representing discretized partial derivative operators

        Arguments
        -----------
        order (int): The order of the derivative operator
        shape (tuple): The shape of the corresponding grid that a discretized function lies at
        axis (int): The axis (variable) of the partial derivative
        dx (float): The grid spacing along the given axis
        acc (int): The accuracy of the differentiation (see fd_coefs and cfd_coefs)
        edges (bool): Whether or not to apply the procedure at the two edges of the axis

        Returns
        ---------------
        Sparse differentiation matrix
        '''
        diff = self.grids[axis].diff_matrix(order=order, acc=acc, fd=fd)
        return tools.generalize_operation(axis=axis, shape=self.shape, matrix=diff, edges=True)

    def without_axis(self, *axis: int)->Grid:
        axis = [(i+self.nd)%self.nd for i in axis]
        grids = []
        for i in range(self.nd):
            if i not in axis:
                grids.append(self.grids[i])
        return NdGrid(*grids)

    def reduce_to(self, axis: int):
        return self.grids[axis]

    def slice(self, axis, i):
        s = self.nd*[slice(None)]
        s[axis] = i
        return tuple(s)
    
    def replace(self, axis: int, grid: Grid1D):
        if axis == -1:
            axis = self.nd - 1
        g = self.grids[:axis] + (grid,) + self.grids[axis+1:]
        if isinstance(self, Grid1D):
            return grid
        else:
            return NdGrid(*g)
        
    def swapaxes(self, i: int, j: int):
        grids = list[self.grids]
        grids[i], grids[j] = grids[j], grids[i]
        return NdGrid(*grids)
    
    def up_to(self, nd: int):
        grids = [self[i] for i in range(nd)]
        return NdGrid(*grids)
    
    def higher_than(self, nd: int):
        grids = [self[i] for i in range(nd, self.nd)]
        return NdGrid(*grids)


class NdGrid(Grid):

    def __new__(cls, *grids: Grid1D):
        if len(grids) == 0:
            return Grid()
        elif not all([isinstance(g, Grid1D) for g in grids]):
            raise ValueError('NdGrid takes for input 1D grids, from subclasses of Grid1D')
        elif len(grids) == 1:
            return grids[0]
        else:
            return super().__new__(cls)

    def __init__(self, *grids: Grid1D):
        self.grids = grids
        self.shape = sum([g.shape for g in grids], ())
        self.shape_full = sum([g.shape_full for g in grids], ())
        self.limits = sum([g.limits for g in grids], ())
        self.periodic = sum([g.periodic for g in grids], ())
        self.x_full = sum([g.x_full for g in grids], ())
        self.x = sum([g.x for g in grids], ())
        self.nd = len(grids)

        self.n = tools.prod(self.shape)
        self.n_full = tools.prod(self.shape_full)
        self.n_bounds = self.n - tools.prod([i-2 for i in self.shape])
        self.n_goodbounds = 2*sum([tools.prod([j-2 for j in self.shape[:i]])*tools.prod([j-2 for j in self.shape[i+1:]]) for i in range(self.nd)])
        self.n_corners = self.n_bounds - self.n_goodbounds # "corners" will be all nodes that are shared between at least 2 boundaries. For nd = 2, corners = 4. For nd >= 3, this is more complicated

    def node(self, *coords: float):
        if len(coords) != self.nd:
            raise ValueError(f'This is a {self.nd}D grid, so node takes {self.nd} coordinates as input')
        return sum([self.grids[i].node(coords[i]) for i in range(self.nd)], ())

    def as_not_periodic(self, *axis):
        if axis != ():
            return NdGrid(*[self.grids[i].as_not_periodic() if i in axis else self.grids[i] for i in range(self.nd)])
        else:
            return NdGrid(*[g.as_not_periodic() for g in self.grids])

    def as_periodic(self, *axis):
        if axis != ():
            return NdGrid(*[self.grids[i].as_periodic() if i in axis else self.grids[i] for i in range(self.nd)])
        else:
            return NdGrid(*[g.as_periodic() for g in self.grids])
        
    def with_periodicity(self, *Bool: bool):
        if len(Bool) != self.nd:
            raise ValueError('with_periodicity() takes boolean arguments, as many as the dimensions of the grid')
        
        return NdGrid(*[self.grids[i].as_periodic() if Bool[i] is True else self.grids[i].as_not_periodic() for i in range(self.nd)])



class Grid1D(Grid):
    nd = 1
    n_bounds = 2
    n_goodbounds = 2
    n_corners = 0
    findiff: fds.Fin_Derivative


    def node(self, coord: float):
        '''
        e.g. coords = (x, y, z) for 3D
        returns nearest node
        '''
        if not (self.limits[0][0] <= coord <= self.limits[0][1]):
            raise ValueError('Coordinates are not within the grid limits')

        nod = np.abs(self.x[0]-coord).argmin()
        return (nod,)

    def diff_matrix(self, order, acc = 1, fd = 'central'):
        return self.findiff.operator(order, acc, fd)

    def as_periodic(self, *axis)->Grid1D:...

    def as_not_periodic(self, *axis)->Grid1D:...

    def with_periodicity(self, Bool: bool)->Grid1D:...

    @property
    def grids(self):
        return self,


class Uniform1D(Grid1D):

    findiff: fds.UniformFinDiff

    def __init__(self, a: float, b: float, n: int, periodic: bool = False):
        self.shape, self.shape_full = (n-periodic,), (n,)
        self.dx = (b-a)/(n-1)
        self.x_full = (np.linspace(a, b, n),)
        if periodic:
            self.x = (self.x_full[0][:-1],)
        else:
            self.x = (self.x_full[0], )
        self.limits = ([a, b],)
        self.periodic = (periodic,)
        self.n = n-periodic
        self.n_full = n
        self.findiff = fds.UniformFinDiff(n, self.dx, periodic)
        

    def __eq__(self, other):
        if isinstance(other, Uniform1D):
            return self.limits == other.limits and self.periodic == other.periodic and self.n == other.n
        else:
            return False

    def node(self, coord: float):
        '''
        e.g. coords = (x, y, z) for 3D
        returns nearest node
        '''
        if not (self.limits[0][0] <= coord <= self.limits[0][1]):
            raise ValueError('Coordinates are not within the grid limits')

        _n = (coord - self.limits[0][0])
        _d = (self.limits[0][1] - self.limits[0][0])
        nod = round(_n/_d*(self.shape[0] - 1))

        return (nod,)

    def diff_matrix(self, order: int, acc=1, fd='central'):
        return self.findiff.operator(order, acc, fd)
    
    def as_not_periodic(self, *axis):
        if axis:
            assert axis[0] == 0
        return Uniform1D(*self.limits[0], self.n_full, periodic=False)

    def as_periodic(self, *axis):
        if axis:
            assert axis[0] == 0
        return Uniform1D(*self.limits[0], self.n_full, periodic=True)
    
    def with_periodicity(self, Bool: bool):
        return Uniform1D(*self.limits[0], self.n_full, periodic=Bool)


class Logarithmic1D(Grid1D):

    def __init__(self, a: float, b: float, n: int, periodic: bool=False):
        self.shape, self.shape_full = (n-periodic,), (n,)
        self.x_full = (np.geomspace(a, b, n),)
        if periodic:
            self.x = (self.x_full[0][:-1],)
        else:
            self.x = (self.x_full[0],)
        self.limits = ([a, b],)
        self.periodic = (periodic,)
        self.n = n-periodic
        self.n_full = n
        self.findiff = fds.FinDiff(self.x_full[0], periodic)

    def __eq__(self, other):
        if isinstance(other, Logarithmic1D):
            return self.limits == other.limits and self.periodic == other.periodic and self.n == other.n
        else:
            return False

    def as_not_periodic(self, *axis):
        if axis:
            assert axis[0] == 0
        return Logarithmic1D(*self.limits[0], self.n_full, periodic=False)
    
    def as_periodic(self, *axis):
        if axis:
            assert axis[0] == 0
        return Logarithmic1D(*self.limits[0], self.n_full, periodic=True)

    def with_periodicity(self, Bool: bool):
        return Logarithmic1D(*self.limits[0], self.n_full, periodic=Bool)


class Unstructured1D(Grid1D):


    def __init__(self, x: np.ndarray, periodic: bool = False):
        self.x_full = (x,)
        self.shape, self.shape_full = (len(x)-periodic,), (len(x),)
        if periodic:
            self.x = (x[:-1],)
        else:
            self.x = (x,)
        self.limits = ([x[0], x[-1]],)
        self.periodic = (periodic,)
        self.n = len(x)-periodic
        self.n_full = len(x)
        self.findiff = fds.FinDiff(x, periodic)

    def __eq__(self, other):
        if isinstance(other, Unstructured1D):
            return self.x == other.x and self.periodic == other.periodic
        else:
            return False

    def as_not_periodic(self, *axis):
        if axis:
            assert axis[0] == 0
        return Unstructured1D(*self.x_full, periodic=False)

    def as_periodic(self, *axis):
        if axis:
            assert axis[0] == 0
        return Unstructured1D(*self.x_full, periodic=True)
    
    def with_periodicity(self, Bool: bool):
        return Unstructured1D(*self.x_full, periodic=Bool)


class InterpedArray:

    def __init__(self, arr: np.ndarray, grid: Grid):
        if arr.shape != grid.shape:
            raise ValueError(f'Grid shape is {grid.shape} while field shape is {arr.shape}')
        self._ndarray = arr.copy()
        self.grid = grid
        self.ndim = grid.nd

    def is_Number(self):
        return self.grid.nd == 0

    def is_complex(self):
        return np.iscomplexobj(self._ndarray)

    def ndarray_full(self):
        ndarr = self.ndarray(self.grid)
        full_ndarr = np.zeros(self.grid.shape_full, dtype=ndarr.dtype)
        slices = tuple([slice(None, -1) if self.grid.periodic[i] else slice(None) for i in range(self.nd)])
        full_ndarr[slices] = ndarr
        '''
        now apply ghost nodes
        '''
        g = self.grid.as_not_periodic()
        for axis in range(self.ndim):
            if self.grid.periodic[axis]:
                for upper_node in g.edge(axis=axis, bound=1, corners=True):
                    lower_node = list(upper_node)
                    lower_node[axis] = 0
                    lower_node = tuple(lower_node)
                    if full_ndarr[upper_node] == 0:
                        full_ndarr[upper_node] = full_ndarr[lower_node]
        return full_ndarr

    def ndarray(self, grid: Grid=None)->np.ndarray:
        #grid must have same dimensions as self._ndarray
        if grid is None or grid == self.grid:
            return self._ndarray
        else:
            if grid.nd < self.ndim:
                raise ValueError(f'This ScalarField object cannot be interpolated in this grid because the grid is {grid.nd}-dimensional instead of {self.ndim}')
            if grid.periodic[:self.ndim] != self.grid.periodic:
                raise ValueError('This ScalarField object cannot be interpolated in this grid because the grid periodicity is different')
            
            if grid.nd == self.ndim:
                return self.get_ndarray(*grid.x)
            else:
                main = grid.up_to(self.ndim)
                rest = grid.higher_than(self.ndim)
                return self.interpolate(main).repeat_along_new_axes(rest)._ndarray
    
    def get_ndarray(self, *x: np.ndarray):
        return tools.interpolate(self.grid.x_full, self.ndarray_full(), x)

    def repeat_along_new_axes(self, grid: Grid):
        #grid: grid of new dimensions, so that the new grid is self.grid * grid

        #see np.broadcast_to for a more memory efficient algorithm
        newarr = tools.repeat_along_extra_dims(self._ndarray, grid.shape)
        return InterpedArray(newarr, self.grid*grid)
    
    def reduced(self, vals: Dict[int, float]):
        slices = self.ndim*[slice(None)]
        raxes = []
        for axis in range(self.ndim):
            if axis in vals:
                slices[axis] = self.grid.reduce_to(axis).node(vals[axis])[0]
                raxes.append(axis)
        slices = tuple(slices)
        return InterpedArray(self._ndarray[slices], self.grid.without_axis(*raxes))
    
    def subs(self, vals: Dict[int, float]):
        arr = self.reduced(vals)._ndarray
        s2 = self.ndim*[slice(None)]
        tiled_axis = []
        for axis in range(self.ndim):
            if axis in vals:
                s2[axis] = np.newaxis
                tiled_axis.append(self.grid.shape[axis])
            else:
                tiled_axis.append(1)
        return InterpedArray(np.tile(arr[tuple(s2)], tuple(tiled_axis)), self.grid)

    def reorder(self, *axis):
        '''
        e.g .reorder(0, 2, 1)
        '''
        assert len(axis) == self.ndim
        for i in range(self.ndim):
            assert i in axis
        arr = self._ndarray.copy()
        grid = self.grid
        myaxes = list(range(self.ndim))
        i = 0
        while i < self.ndim:
            if myaxes[i] == axis[i]:
                i += 1
            else:
                j = axis.index(myaxes[i])
                myaxes[i], myaxes[j] = myaxes[j], myaxes[i]
                arr = arr.swapaxes(i, j)
                grid = grid.swapaxes(i, j)
        
        return InterpedArray(arr, grid)

    def interpolate(self, grid: Grid):
        if grid.limits != grid.limits or grid.periodic != self.grid.periodic:
            raise ValueError('Grids not compatible')
        
        return InterpedArray(self.ndarray(grid=grid), grid)
    
    def diff(self, axis=0, order=1, acc=1, fd='central'):
        arr = self._ndarray
        diffarr = self.grid[axis].findiff.apply(arr, order, acc, fd, axis)
        return InterpedArray(diffarr, self.grid)

    def integrate(self, axis=0):
        arr = self._ndarray
        return InterpedArray(tools.cumulative_simpson(arr, *self.grid.x, initial=0, axis=axis), self.grid)

    def plot(self, grid: Grid=None, ax=None, **kwargs):
        from ..toolkit.plotting import plot
        if grid is None:
            grid = self.grid
        return plot(self.ndarray(grid), grid, ax, **kwargs)
    

def UniformGrid(shape: tuple[int], limits: tuple[list[float]], periodic: tuple[bool]=None):
    if periodic is None:
        periodic = len(x)*[False]
    grids = [Uniform1D(*limits[i], shape[i], periodic[i]) for i in range(len(shape))]
    return NdGrid(*grids)

def LogarithmicGrid(shape: tuple[int], limits: tuple[list[float]], periodic: tuple[bool]=None):
    if periodic is None:
        periodic = len(x)*[False]
    grids = [Logarithmic1D(*limits[i], shape[i], periodic[i]) for i in range(len(shape))]
    return NdGrid(*grids)

def UnstructuredGrid(*x, periodic: tuple[bool]=None):
    if periodic is None:
        periodic = len(x)*[False]
    grids = [Unstructured1D(x[i], periodic[i]) for i in range(len(x))]
    return NdGrid(*grids)

def Random1D(a: float, b: float, n: int, periodic = False):
    x = tools.randomly_spaced_array(a, b, n)
    return Unstructured1D(x=x, periodic=periodic)


def Gaussian1D(a: float, b: float, n: int, center=0., sigma=1., periodic = False):

    arr = tools.inv_gaussian_dist(np.linspace(0, 1, n), a, b, center, sigma)
    
    return Unstructured1D(arr, periodic=periodic)

def UniformFinDiffInterval(center: float, stepsize: float, order=1, acc=1, fd='central'):
    offsets = fds.Fin_Derivative._offsets[fd](order=order, acc=acc)
    x = center + offsets*stepsize
    return Uniform1D(x[0], x[-1], len(x))
