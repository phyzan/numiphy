from __future__ import annotations
from typing import Callable
from ..findiffs import grids
from . import bounds
from ..symlib import symcore as ops
import scipy.sparse.linalg as spl
import numpy as np



class LinearBVP:

    def __init__(self, op: ops.Expr, bcs: bounds.GroupedBcs, grid: grids.Grid):
        bcs.apply_grid(grid)
        self.grid = bcs.grid
        self.bcs = bcs
        self.op = op

    def solve(self, acc=1, fd='central'):
        res = spl.spsolve(self.bcs.Lhs(self.op, acc=acc, fd=fd), self.bcs.rhs()).reshape(self.grid.shape, order='F')
        return *self.grid.x, res

    def get_ScalarField(self, name: str, acc=1, fd='central'):
        res = self.solve(acc, fd)[-1]
        return ops.ScalarField(res, self.grid, name, self.op.oper_symbols)
    

class InhomLinearBVP(LinearBVP):

    def __init__(self, op: ops.Expr, bcs: bounds.GroupedBcs, grid: grids.Grid, source: Callable[..., np.ndarray]):

        super().__init__(op, bcs, grid)
        self._source = source
    
    def source(self, *args)->np.ndarray:
        return self._source(*args)
    
    def solve(self, acc=1, fd='central'):
        src = self.bcs.discretize(self.source)
        res = spl.spsolve(self.bcs.Lhs(self.op, acc=acc, fd=fd), src).reshape(self.grid.shape, order='F')
        return *self.grid.x, res