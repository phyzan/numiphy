from .. symlib import symcore as ops
from ..findiffs import grids
from ..toolkit import tools
from . import bounds
import numpy as np
import scipy.sparse as sp
from abc import ABC, abstractmethod


class CachedOperator(ABC):

    @abstractmethod
    def matrix(self, grid: grids.Grid, *args)->sp.csr_matrix:
        pass

    @abstractmethod
    def array(self, grid: grids.Grid, *args)->np.ndarray:
        pass


class MatrixOperator(CachedOperator):

    def __init__(self, m: sp.csr_matrix):
        self.m = m

    def matrix(self, grid: grids.Grid, *args):
        return self.m
    
    def array(self, grid: grids.Grid, *args):
        raise NotImplementedError('MatrixOperator objects are in general non-diagonal matrices, so they do not have a well-defined array')


class DiagonalMatrixOperator(MatrixOperator):

    def __init__(self, arr: np.ndarray):
        self.arr = arr.copy()
        self.m = tools.as_sparse_diag(arr)

    def array(self, grid: grids.Grid, *args):
        return self.arr


class CachedAdd(CachedOperator):

    def __init__(self, *seq: CachedOperator):
        self.add_seq = seq

    @staticmethod
    def simplify(*seq):
        m_seq: list[sp.csr_matrix] = []
        dm_seq: list[np.ndarray] = []
        other = []
        for op in seq:
            if isinstance(op, DiagonalMatrixOperator):
                dm_seq.append(op.arr)
            elif isinstance(op, MatrixOperator):
                m_seq.append(op.m)
            else:
                other.append(op)
        new_seq: list[MatrixOperator] = other
        if dm_seq and not m_seq:
            new_seq.append(DiagonalMatrixOperator(sum(dm_seq)))
        elif dm_seq and m_seq:
            new_seq.append(MatrixOperator(sum(m_seq)+tools.as_sparse_diag(sum(dm_seq))))
        elif not dm_seq and m_seq:
            new_seq.append(MatrixOperator(sum(m_seq)))

        if len(new_seq) == 1:
            return new_seq[0]
        else:
            return CachedAdd(*new_seq)

    def matrix(self, grid: grids.Grid, *args):
        seq = self.add_seq
        return sum([m.matrix(grid, *args) for m in seq[1:]], start = seq[0].matrix(grid, *args))

    def array(self, grid: grids.Grid, *args):
        seq = self.add_seq
        return np.sum([m.array(grid, *args) for m in seq], axis=0)


class CachedMul(CachedOperator):
    
    def __init__(self, *seq: CachedOperator):
        self.mul_seq = seq

    @staticmethod
    def simplify(*seq):
        new_seq = []
        i = 0
        while i < len(seq):
            if not isinstance(seq[i], MatrixOperator):
                new_seq.append(seq[i])
                i += 1
                continue
            else:
                m_seq = []
                while isinstance(seq[i], MatrixOperator):
                    m_seq.append(seq[i].m)
                    i += 1
                    if i == len(seq):
                        break

                if all([isinstance(mi, DiagonalMatrixOperator) for mi in m_seq]):
                    m_seq: list[DiagonalMatrixOperator]
                    arr = np.sum([mi.arr for mi in m_seq])
                    new_seq.append(DiagonalMatrixOperator(arr))
                else:
                    m = tools.multi_dot_product(*m_seq)
                    new_seq.append(MatrixOperator(m))
        if len(new_seq) == 1:
            return new_seq[0]
        else:
            return CachedMul(*new_seq)

    def matrix(self, grid: grids.Grid, *args):
        return tools.multi_dot_product(*[m.matrix(grid, *args) for m in self.mul_seq])

    def array(self, grid: grids.Grid, *args):
        return np.prod([arg.array(grid, *args) for arg in self.mul_seq], axis=0)
    


class CachedPow(CachedOperator):
    
    def __init__(self, base: CachedOperator, exp: CachedOperator):
        self.base, self.exp = base, exp

    @staticmethod
    def simplify(base: CachedOperator, exp: CachedOperator):
        if isinstance(base, DiagonalMatrixOperator) and isinstance(exp, DiagonalMatrixOperator):
            arr = base.arr ** exp.arr
            return DiagonalMatrixOperator(arr)
        else:
            return CachedPow(base, exp)
    
    def array(self, grid: grids.Grid, *args):
        return self.base.array(grid, *args) ** self.exp.array(grid, *args)
    
    def matrix(self, grid: grids.Grid, *args):
        arr = self.array(grid, *args)
        return tools.as_sparse_diag(arr)


class CachedVar(CachedOperator):

    def __init__(self, axis: int, deduct=0):
        self.axis = axis
        self.n_ded = deduct

    def array(self, grid: grids.Grid, *args):
        val = args[self.axis-grid.nd]
        return val * np.ones(grid.n-self.n_ded)

    def matrix(self, grid: grids.Grid, *args):
        val = args[self.axis-grid.nd]
        return val * sp.identity(grid.n-self.n_ded, format='csr')
    

def cache_operator(op: ops.Expr, grid: grids.Grid, bcs: bounds.GroupedBcs=None, acc=1, fd='central')->CachedOperator:
    if bcs is not None:
        if not bcs.is_homogeneous:
            raise ValueError('Boundary conditions that are given as input in order to cache an operator need to be homogeneous, so that matrices can be reduced accordingly')
    if isinstance(op, ops.AddOp):
        return CachedAdd.simplify(*[cache_operator(arg, grid, bcs, acc, fd) for arg in op.args])
    elif isinstance(op, ops.MulOp):
        return CachedMul.simplify(*[cache_operator(arg, grid, bcs, acc, fd) for arg in op.args])
    elif isinstance(op, ops.PowOp):
        base = cache_operator(op.base, grid, bcs, acc, fd)
        exp = cache_operator(op.power, grid, bcs, acc, fd)
        return CachedPow.simplify(base, exp)
    elif isinstance(op, ops.Diff):
        m = op.matrix(grid, acc, fd)
        if bcs is None:
            return MatrixOperator(m)
        else:
            return MatrixOperator(bcs.reduced_matrix(m))
    elif isinstance(op, ops.VariableOp):
        if op.axis < grid.nd:
            arr = op.array(grid)
            if bcs is None:
                return DiagonalMatrixOperator(arr)
            else:
                return DiagonalMatrixOperator(bcs.reduced_array(arr))
        else:
            if bcs is None:
                return CachedVar(op.axis)
            else:
                return CachedVar(op.axis, deduct=len(bcs.reserved_nods()))
    else:
        arr = op.array(grid)
        if bcs is None:
            return DiagonalMatrixOperator(arr)
        else:
            return DiagonalMatrixOperator(bcs.reduced_array(arr))
    

    '''
    In Const and Var, include a feature that some nods are skipped (to accomodate for reduced matrices in homogeneous bcs)
    '''