from __future__ import annotations
from . import bounds
from ..findiffs import grids
from ..symlib import operators as sym
from ..toolkit import tools
import scipy.linalg as sl
import numpy as np

'''

CREATE NULL GRID


CHECK HERMICITY CONDITIONS. MAYBE AN SL OPERATOR IS HERMITIAN WITH PROPER BCS

DETERMINTE ADJOINT AND HERMICITY WITH WEIGHT FUNCTION. MAYBE STURM LIOUVILLE ONLY IN 1D
'''

def has_adjoint(op: sym.Operator, bcs: bounds.GroupedBcs):
    for arg in op.deepsearch():
        if isinstance(arg, sym.VariableOp):
            if arg.axis >=  bcs.nd:
                raise ValueError(f'The "{arg.name}" variable has axis={arg.axis}, while the boundary conditions correspond to a lower, {bcs.nd}-dimensional, space')
        elif isinstance(arg, sym.Diff):
            assert has_adjoint(arg.symbol, bcs)
            for bc in bcs.bcs:
                if not bc.is_homogeneous:
                    return False
                
            for bc in bcs.int_bcs:
                if not bc.is_symmetric():
                    return False
                
            bc = bcs.ext_bcs[arg.symbol.axis]
            if arg.order % 2 == 1:
                if isinstance(bc, bounds.StandardAxisBcs):
                    if not (bc.low.is_dirichlet and bc.up.is_dirichlet):
                        return False
                    
                for bc_in in bcs.int_bcs:
                    if not bc_in.is_dirichlet:
                        return False
        elif arg.is_AbstractFunction:
            if not all([has_adjoint(v, bcs) for v in arg.variables]):
                return False
        elif arg.is_MathFunction:
            if not has_adjoint(arg.arg, bcs):
                return False
    
    return True

def is_hermitian(op: sym.Operator, bcs: bounds.GroupedBcs):
    return op.adjoint() == op and has_adjoint(op, bcs)


class EigProblem:
    '''
    MAKE THIS COMPATIBLE WITH STURM LIOUVILLEEEEEEE
    '''

    eigp1D: tuple[EigProblem]

    def __init__(self, operator: sym.Operator, bcs: bounds.GroupedBcs, grid: grids.Grid):
        if not bcs.is_homogeneous:
            raise NotImplementedError('Diagonalization has not been implemented for problems with non homogeneous boundary conditions')
        if not (operator == operator.expand()):
            raise ValueError('Differential operator must be expanded')
        self.op = operator
        self.bcs = bcs
        self.bcs.apply_grid(grid)
        self.grid = self.bcs.grid
        
        q = self.op.separate()
        if len(list(q)) > 1:
            self.is_separable = True
            eigp1D = []
            for i in range(len(q)):
                axis = q[i].variables[0].axis
                eigp1D.append(EigProblem(q[i].toaxis(0), bounds.GroupedBcs(bcs.ext_bcs[axis].newcopy(0)), self.grid.grids[axis]))
            self.eigp1D = tuple(eigp1D)
        else:
            self.is_separable = False
            self.eigp1D = ()
        
        self.is_solved = False

    def _can_be_reduced(self):
        '''
        As long as the diagonal is not zero in the bcs entries, the
        matrix can be reduced and diagonalized using standard algorithms
        '''
        m = self.bcs.lhs()
        for i in self.bcs.nods_generator():
            if m[i, i] == 0:
                return False
        return True

    def solve(self):
        '''
        manage hermicity here
        '''
        if self.is_solved:
            return
        
        if self.is_separable and len(self.bcs.int_bcs) == 0:
            for op in self.eigp1D:
                op.solve()

            e = tools.kronsum(*[op.e for op in self.eigp1D]) # (I_z kron I_y kron e_x) + (I_z kron e_y kron I_x) + (...)
            argsort = e.argsort()
            e = e[argsort]
            f = tools.KronProd(*[op.f for op in self.eigp1D])
            f.sort(argsort, axis=1)
            self.e, self.f = e, f
        else:
            if not self._can_be_reduced():
                raise ValueError('At least one interior boundary condition has no dependence on the grid point it is reffering to.')
            
            m = self.bcs.Lhs(self.op, reduced=True)
            self.isherm = False
            if is_hermitian(self.op, self.bcs):
                if self.grid.is_uniform():
                    self.isherm = True
                    if not tools.is_sparse_herm(m):
                        raise ValueError('Matrix is not hermitian but should be. Check numerical representation')
                    elif tools.is_tridiag(m):
                        e, f = sl.eigh_tridiagonal(m.diagonal(0), m.diagonal(1))
                    else:
                        e, f = sl.eigh(m.toarray()) #check if the operator is the kronecker sum of two or more operators. If not, go banded
                    
                    s = e.argsort()
                    e, f = e[s], f[:, s]
                else:
                    e, f = sl.eig(m.toarray())
            else:
                e, f = sl.eig(m)

            self.e, self.f = e, f
        self.is_solved = True

    def eigenvec(self, i, normalized=True):
        '''
        ex + ey + ez
        n1 n2 n3
        '''
        f = self.bcs.insert(self.f[:, i])
        if normalized:
            integral = tools.full_multidim_simpson((f*f.conj()).reshape(self.grid.shape, order='F'), *self.grid.x)
            f = f / np.sqrt(integral)
        return f.reshape(self.grid.shape, order='F')


class SturmLiouville:

    grid: grids.Uniform1D
    e: np.ndarray
    f: np.ndarray

    def __init__(self, p: sym.Operator, q: sym.Operator, r: sym.Operator, bcs: bounds.AxisBcs, grid: grids.Uniform1D):
        p, q, r = sym.asexpr(p), sym.asexpr(q), sym.asexpr(r)
        for f in (p, q, r):
            if len(f.variables) > 1:
                raise ValueError('Variables must be at most 1')
            elif len(f.variables) == 1:
                assert f.variables[0].axis == 0
        
        x = p.variables+q.variables+r.variables
        for i in range(len(x)-1):
            assert x[i]==x[i+1]

        if not bcs.is_homogeneous:
            raise NotImplementedError('The Sturm-Liouvilly problem has not been implemented for inhomogeneous boundary conditions')

        self.grouped_bcs = bounds.GroupedBcs(bcs)
        self.bcs = bcs
        bcs.apply_grid(grid)
        self.grouped_bcs.apply_grid(grid)
        self.grid = bcs.grid
        if len(x) == 0:
            self.var = sym.Variable('x')
        else:
            self.var = x[0]

        if p == 1 and q == 0:
            self.w = sym.I
        else:
            self.w = 1/p * sym.exp(sym.asintegral(q/p, self.var))

        Dx = sym.Diff(self.var)

        self.operator = p*Dx**2 + q*Dx + r

        self._op = Dx*self.w*p*Dx + r*self.w #is obviously self-adjoint

    def solve(self):
        op = self._op.expand()

        if (op - op.adjoint()).expand() != 0: #sanity check
            raise ValueError('Operator should be self-adjoint but is not. Check Sturm-Liouville implementation and expand operation')
        
        m = self.bcs.Lhs(op, reduced=True)
        if not tools.is_sparse_herm(m):
            m = (m + m.transpose())/2
        assert tools.is_sparse_herm(m) and tools.is_tridiag(m)


        w_arr = self.bcs.reduced_array(self.w.array(self.grid))
        if isinstance(self.w, sym.Const):
            m = m/self.w.a
            e, f = sl.eigh_tridiagonal(m.diagonal(0), m.diagonal(1))
        else:
            e, f = sl.eigh(m.toarray(), np.diag(w_arr))

        self.e, self.f = e, f
        self.w_arr = self.w.array(self.grid)

    def eigenvec(self, i, normalized=True):
        '''
        ex + ey + ez
        n1 n2 n3
        '''
        f = self.grouped_bcs.insert(self.f[:, i])
        if normalized:
            integral = tools.full_multidim_simpson((self.w_arr*f*f.conj()).reshape(self.grid.shape, order='F'), *self.grid.x)
            f = f / np.sqrt(integral)
        return f.reshape(self.grid.shape, order='F')
