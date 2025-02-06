import numpy as np
import scipy.sparse as sp
import math
from ..toolkit import tools
from abc import ABC, abstractmethod


def _nco(order, acc):
    return int((order+1)/2)-1+acc

def _nfbo(order, acc):
    return order+acc-1

def central_offsets(order, acc):
    nco = _nco(order, acc)
    return np.arange(-nco, nco+1, dtype=int)

def forward_offsets(order, acc):
    return np.arange(0, order+acc, dtype=int)

def backward_offsets(order, acc):
    return np.arange(-(order+acc)+1, 1, dtype=int)

def uniform_findiff_weights(order: int, offsets: list):
    '''
    Creates the finite difference coefficients to approximate any n-th order derivative
    of a discretized function

    Arguments
    -------------
    order (int): order of differentiation
    oddsets (tuple of ints): Indices around a given point to be used in the
        differentiation approximantion

    https://en.wikipedia.org/wiki/Finite_difference_coefficient
    '''
    n = len(offsets)
    offsets = np.array(offsets)
    A = np.zeros((n, n), dtype=int)
    for i in range(n):
        A[i] = offsets**i
    b = np.zeros(n)
    b[order] = math.factorial(order)
    return np.linalg.solve(A, b)

def diff(y: np.ndarray, x: np.ndarray, order=1, periodic: bool = False, acc=1)->np.ndarray:
    if x.shape != y.shape:
        raise ValueError('x and y arrays have different shapes')
    if len(x.shape) != 1:
        raise ValueError('This function only applies for 1D arrays')

    fd = FinDiff(x, periodic)
    oper = fd.operator(order, acc, fd)
    return oper.dot(y)


fd_map = {'central': 0, 'forward': 1, 'backward': -1, 0: 'central', 1: 'forward', -1: 'backward'}


class Fin_Derivative(ABC):

    _offsets = {'central': central_offsets, 'forward': forward_offsets, 'backward': backward_offsets}

    def __init__(self, n: int, periodic: bool = False):
        self.n = n-periodic
        self.is_periodic = periodic

    def fits_central_fd(self, node: int, order, acc):
        if self.is_periodic:
            return True
        else:
            return node >= _nco(order, acc) and node+_nco(order, acc) <= self.n-1
    
    def can_fit(self, node, order, acc, fd: str):
        if self.is_periodic:
            return True
        else:
            if fd == 'central':
                return node >= _nco(order, acc) and node+_nco(order, acc) <= self.n-1
            elif fd == 'forward':
                return node+_nfbo(order, acc) < self.n
            elif fd == 'backward':
                return node-_nfbo(order, acc) >= 0

    def offsets(self, node: int, order, acc, fd = 'central'):
        if self.can_fit(node, order, acc, fd):
            return self._offsets[fd](order, acc)
        else:
            p = _nco(order, acc)
            l = ((node-p) < 0) * (p-node)
            r = 2*p+1 - (node+p>self.n-1)*(node+p-self.n+1)
            if r-l < order+1:
                if l != 0:
                    return np.arange(0, order+1, dtype=int) - node
                else:
                    return np.arange(self.n-node-order-1, self.n-node, dtype=int)
            else:
                return central_offsets(order, acc)[l:r]


    def stencil_points_from_offsets(self, node: int, offsets):
        if self.is_periodic:
            return (node+offsets+self.n)%self.n
        else:
            return offsets + node

    def stencil_points(self, node: int, order, acc, fd='central'):
        offsets = self.offsets(node, order, acc, fd)
        return self.stencil_points_from_offsets(node, offsets)
    
    @abstractmethod
    def coefs_from_offsets(self, node: int, order, offsets: list[int]):
        pass

    def coefs(self, node: int, order, acc, fd='central'):
        return self.coefs_from_offsets(node, order, self.offsets(node, order, acc, fd))

    def operator_element(self, node: int, order, acc, fd='central'):
        '''
        Creates a sparse matrix that represents a differential operator, with entries only in one row.

        Parameters
        -----------
        n: number of grid points
        node: Row of the differential operator that we want to calculate (equivalently the corresponding grid node)
        dx: grid spacing
        order: order of the differential operator
        acc: accuracy of the finite differences used
        periodic: whether the grid is periodic or not
        fd (0, 1, or -1): Type of finite differences used
            0: central finite differences (if possible)
            1: forward finite differences
            -1: backward finite differences

            If fd=1 or -1, then if the given node is close enough to the grid edges and the grid is not periodic, an error will be raised.

        Returns
        -----------
        n x n sparse matrix with entries only in one row


        Examples
        ------------
        
        >>> m = diff_element(n=5, node=2, dx=1, order=2, acc=1, periodic=False, fd=0)
        >>> print(m.toarray())
        [[ 0.  0.  0.  0.  0.]
        [ 0.  0.  0.  0.  0.]
        [ 0.  1. -2.  1.  0.]
        [ 0.  0.  0.  0.  0.]
        [ 0.  0.  0.  0.  0.]]
        >>>
        >>>
        >>>
        >>> m = diff_element(n=5, node=4, dx=1, order=2, acc=1, periodic=False, fd=0) #backward finite differences will be used because node=4 is at the edge
        >>> print(m.toarray())
        [[ 0.  0.  0.  0.  0.]
        [ 0.  0.  0.  0.  0.]
        [ 0.  0.  0.  0.  0.]
        [ 0.  0.  0.  0.  0.]
        [ 0.  0.  1. -2.  1.]]
        >>>
        >>>
        >>>
        >>> m = diff_element(n=5, node=4, dx=1, order=2, acc=1, periodic=True, fd=0)
        >>> print(m.toarray()) #periodic is True now, see what happens
        [[ 0.  0.  0.  0.  0.]
        [ 0.  0.  0.  0.  0.]
        [ 0.  0.  0.  0.  0.]
        [ 0.  0.  0.  0.  0.]
        [ 1.  0.  0.  1. -2.]]
        '''
        
        offsets = self.offsets(node, order, acc, fd)
        nf = len(offsets)
        rows = node + np.zeros(nf, dtype=int)
        cols = self.stencil_points_from_offsets(node, offsets)
        vals = self.coefs(node, order, acc, fd)
        return sp.csr_matrix((vals, (rows, cols)), shape=(self.n, self.n))

    def operator(self, order, acc=1, fd='central')->sp.csr_matrix:
        m = sp.csr_matrix((self.n, self.n))
        for i in range(self.n):
            m += self.operator_element(i, order, acc, fd=fd)
        return m
    
    def apply(self, arr: np.ndarray, order=1, acc=1, fd='central', axis=0):
        m = self.operator(order, acc, fd)
        return tools.tensordot(m, arr, axis=axis)


class FinDiff(Fin_Derivative):
    _ghost_point: float
    def __init__(self, x: np.ndarray, periodic: bool = False):
        if periodic:
            self._ghost_point = x[-1]
            self.x = x[:-1]
        else:
            self.x = x
        super().__init__(len(x), periodic)

    def coefs_from_offsets(self, node: int, order, offsets: list):
        '''https://www.ams.org/journals/mcom/1988-51-184/S0025-5718-1988-0935077-0/'''
        offsets = np.asarray(offsets)
        x = self.x
        n = len(offsets)
        lenx = len(x)
        
        if self.is_periodic and not np.all(np.logical_and(node+offsets>=0, node+offsets<lenx)):
            extra = self._ghost_point-self.x[-1] + (x[-1]-x[0])
            xx = np.hstack((x, x+extra, x+2*extra))
            a = xx[lenx+node+offsets]
            x0 = xx[node+lenx]
        else:
            a = x[node+offsets]
            x0 = x[node]
        
        coefs = np.zeros(shape=(order+1, n, n))
        coefs[0,0,0]=1
        c1 = 1
        for i in range(1, n):
            c2 = 1
            for j in range(i):
                c3 = a[i] - a[j]
                c2 *= c3
                for k in range(min(i, order)+1):
                    coefs[k,i,j] = ((a[i]-x0)*coefs[k,i-1,j]-k*coefs[k-1,i-1,j])/c3
            for k in range(min(i, order)+1):
                coefs[k,i,i] = c1/c2*(k*coefs[k-1,i-1,i-1]-(a[i-1]-x0)*coefs[k,i-1,i-1])
            c1=c2
        return coefs[-1, -1]


class UniformFinDiff(Fin_Derivative):

    def __init__(self, n: int, dx = 1.0, periodic = False):
        super().__init__(n, periodic)
        self.dx = dx

    def coefficients(self, order, acc=1, fd='central'):
        return uniform_findiff_weights(order, self._offsets[fd](order, acc))/self.dx**order


    def coefs_from_offsets(self, node: int, order, offsets: list[int]):
        '''
        Creates the finite difference coefficients to approximate any n-th order derivative
        of a discretized function

        Arguments
        -------------
        order (int): order of differentiation
        oddsets (tuple of ints): Indices around a given point to be used in the
            differentiation approximantion

        https://en.wikipedia.org/wiki/Finite_difference_coefficient
        '''
        return uniform_findiff_weights(order, offsets)/self.dx**order

    def coefs(self, node: int, order, acc, fd='central'):
        if self.can_fit(node, order, acc, fd):
            return self.coefficients(order, acc, fd)
        else:
            return super().coefs(node, order, acc, fd)
