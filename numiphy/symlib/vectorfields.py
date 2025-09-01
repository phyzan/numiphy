from __future__ import annotations
from numiphy.findiffs import grids
from numiphy.odesolvers import LowLevelODE, OdeSystem, Symbol, Rational
from numiphy.toolkit.plotting import *
import numpy as np
import scipy.optimize as sciopt
import scipy.integrate as scint
from numiphy.symlib import symcore as sym
from numiphy.symlib.pylambda import ScalarLambdaExpr, VectorLambdaExpr
from numiphy.symlib.geom import Line2D, Circle
from numiphy.toolkit.tools import call_with_consumed, call_builtin_with_consumed
import warnings

class VectorField2D:


    def __init__(self, Fx: sym.Expr, Fy: sym.Expr, x:sym.Symbol, y:sym.Symbol, *args: sym.Symbol):
        symbols = []
        for xi in (x, y)+args:
            if xi in symbols:
                raise ValueError('Repeated symbols')
            symbols.append(xi)
            
        self.xvar, self.yvar = x, y
        self.x = ScalarLambdaExpr(Fx, *symbols)
        self.y = ScalarLambdaExpr(Fy, *symbols)

        self.Jac = VectorLambdaExpr([[Fx.diff(x), Fx.diff(y)], [Fy.diff(x), Fy.diff(y)]], *symbols)
        self.args = args


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
    
    def streamline(self, x0, y0, s, curve_length=True, rich=False, **kwargs):
        '''
        Let F be a vector field.
        A field line R(s) passing through a point (x0, y0) satisfies the equation

        dR/ds = F(R), with initial conditions R(s) = (x0, y0)

        The "s" parameter is the more useful curve-length parameter, if we instead choose the ode:
        dR/ds = F(R) / |F(R)|
        which means dR/ds is the unit vector of the vector field at each point
        '''
        t = Symbol("t")
        fx, fy = self.x.expr, self.y.expr
        if curve_length:
            A = (fx**2 + fy**2)**Rational(1, 2)
        else:
            A = 1
        odesys = OdeSystem([fx/A, fy/A], t, [self.xvar, self.yvar], self.args, events=kwargs.pop('events', ()))

        ode, kwargs = call_with_consumed(odesys.get, t0=0, q0=[x0, y0], **kwargs)
        if rich:
            res, kwargs = call_builtin_with_consumed(ode.rich_integrate, dict(event_options=[], max_prints=0), s, **kwargs)
        else:
            res, kwargs = call_builtin_with_consumed(ode.integrate, dict(max_frames=-1, event_options=[], max_prints=0, include_first=False), s, **kwargs)
        if kwargs:
            warnings.warn(f"The keyword arguments {kwargs} had no effect")
        return res
        
    def loop(self, q, r, *args):
        c = Circle(r, q)
        return self.flow(c, *args)
    
    def plot(self, grid: grids.Grid, args=(), scaled=True, **kwargs):

        def update(event):
            xlims = ax.get_xlim() if figure.xlims[0] is None else figure.xlims
            ylims = ax.get_ylim() if figure.ylims[0] is None else figure.ylims
            ax.clear()

            qp: QuiverPlot = figure.artists[0]
            qp.args = get_quiver_data(xlims, ylims)
            figure.draw(ax)

            ax.set_xlim(*xlims)
            ax.set_ylim(*ylims)

        def get_quiver_data(xlims, ylims):
            x = np.linspace(*xlims, grid.shape[0])
            y = np.linspace(*ylims, grid.shape[1])
            xmesh = np.meshgrid(x, y, indexing='xy')
            X, Y = self(*xmesh, *args)
            mag = np.sqrt(X**2 + Y**2)
            with np.errstate(invalid='ignore'):
                X = np.where(X!=0, X/mag, 0)
                Y = np.where(Y!=0, Y/mag, 0)
            res = x, y, X, Y
            if scaled:
                res += (mag,)
            return res

        figure = SquareFigure("VectorField", figsize=(5, 5))
        figure.ftype = 'pdf'
        figure.xlabel = f'${self.xvar}$'
        figure.ylabel = f'${self.yvar}$'
        figure.yrot = 0
        figure.fontsize['label'] = 15
        figure.grid = dict(visible=False)

        figure.aspect = 'equal'
        figure.add(QuiverPlot(*get_quiver_data(*grid.limits), **kwargs))

        fig, ax = figure.plot()
        if figure.xlims[0] is not None and figure.xlims[1] is not None:
            ax.set_xlim(*figure.xlims)
        if figure.ylims[0] is not None and figure.ylims[1] is not None:
            ax.set_ylim(*figure.ylims)    
        fig.canvas.mpl_connect('button_release_event', update)
        return figure, fig, ax, lambda : update(None)