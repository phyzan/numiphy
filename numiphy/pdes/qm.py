from __future__ import annotations
from . import ivps
from . import bounds
from ..symlib import symcore as sym
from ..findiffs import grids
from ..symlib import geom
from . import linalg
import numpy as np


class Schroedinger1D(ivps.HomLinearIBVP):

    def __init__(self, V: sym.Expr, grid: grids.Grid1D, hbar=1., m=1.):
        x = sym.Symbol('x', 0)
        Dx = sym.Diff(x)
        self.H = -hbar**2/(2*m) * Dx**2 + V
        bcx = bounds.StandardAxisBcs(bounds.Dirichlet(), bounds.Dirichlet())
        bcs = bounds.StaticBcs(bcx)
        self.hbar, self.m = hbar, m
        super().__init__(bcs, grid, self.H, -1j/hbar)

    def get_eigenstates(self):
        self.eigproblem.solve()

    def add_delta_potential(self, intensity, loc):
        bc = bounds.Robin([-2*self.m*intensity/self.hbar**2, 1, -1]).apply(geom.Point(loc))
        self.bcs = bounds.StaticBcs(*self.bcs.newcopy().bcs, bc)
        self.eigproblem = linalg.EigProblem(self.H, self.bcs, self.grid)
    
    def get_ScalarField(self, t, dt, method='RK45', rtol=0., max_frames=-1, display=True):
        return super().get_ScalarField('Ψ', t, dt, method, rtol, max_frames, display)

    def eigfunc(self, i, normalized=True):
        return self.grid.x[0], self.eigproblem.eigenvec(i, normalized)
    
    @property
    def eigvals(self):
        return self.eigproblem.e


class WavePacket1D:

    def __init__(self, x0, p0, sx, t0=0, m=1, hbar=1):
        self.t0 = t0
        self.x0 = x0
        self.p0 = p0
        self.sx = sx
        self.m = m
        self.hbar = hbar

    def __call__(self, x, t):
        t0, x0, p0, sx, m, hbar = self.t0, self.x0, self.p0, self.sx, self.m, self.hbar
        t = t-t0
        sx_t = (sx**2+t**2*hbar**2/(4*m**2*sx**2))**0.5
        k0, y, a = p0/hbar, x-(x0+p0/m*t), np.angle((sx/sx_t-1j*hbar*t/(2*m*sx*sx_t))**0.5)
        theta = k0*(x-x0)-hbar*k0**2/(2*m)*t+a+hbar*t*y**2/(8*m*sx**2*sx_t**2)
        return 1/(2*np.pi*sx_t**2)**(1/4)*np.exp(-y**2/(4*sx_t**2))*np.exp(1j*theta)
    
    def ic(self, x):
        return self(x, self.t0)
