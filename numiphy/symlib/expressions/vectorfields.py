from __future__ import annotations
from numiphy.findiffs import grids
from numiphy.odesolvers import ode_solvers as ods
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sciopt
import scipy.integrate as scint
from numiphy.symlib import expressions as sym
from numiphy.symlib.geom import Line2D, Circle
        

class VectorField2D:


    def __init__(self, Fx: sym.Expr, Fy: sym.Expr, x:sym.Variable, y:sym.Variable, *args: sym.Variable):
        symbols = []
        for xi in (x, y)+args:
            if xi in symbols:
                raise ValueError('Repeated symbols')
            symbols.append(xi)
            
        self.xvar, self.yvar = x, y
        self.x = sym.ScalarLambdaExpr(Fx, *symbols)
        self.y = sym.ScalarLambdaExpr(Fy, *symbols)

        self.Jac = sym.VectorLambdaExpr([[Fx.diff(x), Fx.diff(y)], [Fy.diff(x), Fy.diff(y)]], *symbols)


    def __call__(self, *args, **kwargs):
        return np.array([self.x(*args, **kwargs), self.y(*args, **kwargs)])

    def __mul__(self, other: float):
        return VectorField2D(other*self.x.expr, other*self.y.expr, self.xvar, self.yvar)
    
    def __rmul__(self, other: float):
        return self*other
    
    def unitvec(self, *args, **kwargs):
        vec = self(*args, **kwargs)
        return vec/np.sqrt(vec[0]**2+vec[1]**2)

    def call(self, q, *args):
        return self(*q, *args)
    
    def calljac(self, q, *args):
        return self.Jac(*q, *args)
    
    def fixed_point(self, x0, y0, *args):
        return sciopt.root(self.call, [x0, y0], jac = self.calljac, args=args).x

    def flowdot(self, line: Line2D):
        return lambda u, *args: self.x(line.x(u), line.y(u), *args)*line.xdot(u) + self.y(line.x(u), line.y(u), *args)*line.ydot(u)

    def flow(self, line: Line2D, *args)->float|complex:
        cdot = self.flowdot(line)
        return scint.quad(cdot, *line.lims, args=args, epsabs=1e-10)[0]
    
    def streamline(self, x0, y0, s, ds=1e-3, err=1e-8, *args):
        '''
        Let F be a vector field.
        A field line R(s) passing through a point (x0, y0) satisfies the equation

        dR/ds = F(R), with initial conditions R(s) = (x0, y0)

        The "s" parameter is the more useful curve-length parameter, if we instead choose the ode:
        dR/ds = F(R) / |F(R)|
        which means dR/ds is the unit vector of the vector field at each point
        '''
        ics = (0, np.array([x0, y0]))
        res = ods.PythonicODE(lambda s, q: self.unitvec(*q, *args)).solve(ics, s, ds, err=err).func
        return res.transpose()
        
    def loop(self, q, r, *args):
        c = Circle(r, q)
        return self.flow(c, *args)
    
    def plot(self, grid: grids.Grid, *args, **kwargs):

        def update(event):
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            lines = ax.get_lines()
            ax.clear()
            x = np.linspace(*xlim, grid.shape[0])
            y = np.linspace(*ylim, grid.shape[1])
            xmesh = np.meshgrid(x, y, indexing='ij')
            X, Y = self(*xmesh, *args)
            mag = np.sqrt(X**2+Y**2)
            ax.quiver(*xmesh, X/mag, Y/mag, mag, **kwargs)
            for line in lines:
                ax.add_line(line)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlabel(str(self.xvar), fontsize=20)
        ax.set_ylabel(str(self.yvar), fontsize=20)

        xmesh = grid.x_mesh()
        X, Y = self(*xmesh, *args)
        mag = np.sqrt(X**2+Y**2)

        ax.quiver(*xmesh, X/mag, Y/mag, mag, **kwargs)
        fig.canvas.mpl_connect('button_release_event', update)
        return fig, ax
    
    def plot_line(self, line: Line2D, grid: grids.Grid, n=400, *args, **kwargs):
        fig, ax = VectorField2D.plot(self, grid, *args, **kwargs)
        u = np.linspace(*line.lims, n)
        ax.plot(line.x(u), line.y(u))
        return fig, ax
    
    def plot_circle(self, q, r, grid: grids.Grid, n=400, *args, **kwargs):
        c = Circle(r, q)
        return VectorField2D.plot_line(self, c, grid, n, *args, **kwargs)