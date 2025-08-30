from __future__ import annotations
from numiphy.findiffs import grids
from numiphy.odesolvers import LowLevelODE, OdeSystem, Symbol, Rational
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as mcollections
from matplotlib.patches import FancyArrowPatch
import matplotlib.quiver as mquiver
import scipy.optimize as sciopt
import scipy.integrate as scint
from numiphy.symlib import symcore as sym
from numiphy.symlib.pylambda import ScalarLambdaExpr, VectorLambdaExpr
from numiphy.symlib.geom import Line2D, Circle


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
    
    def streamline(self, x0, y0, s, curve_length=True, *args, **odekw):
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
        odesys = OdeSystem([fx/A, fy/A], t, [self.xvar, self.yvar], self.args)
        ode = odesys.get(0, [x0, y0], args=args, **odekw)
        return ode.integrate(s, include_first=True)
        
    def loop(self, q, r, *args):
        c = Circle(r, q)
        return self.flow(c, *args)
    
    def plot(self, grid: grids.Grid, *args, scaled=True, **kwargs):

        def update(event):
            # -------------------------
            # 0. Store current axes limits
            # -------------------------
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

            # -------------------------
            # 1. Store lines data
            # -------------------------
            lines_data = []
            for line in ax.get_lines():
                lines_data.append({
                    'x': line.get_xdata(),
                    'y': line.get_ydata(),
                    'color': line.get_color(),
                    'linestyle': line.get_linestyle(),
                    'linewidth': line.get_linewidth(),
                    'marker': line.get_marker(),
                    'markersize': line.get_markersize(),
                    'alpha': line.get_alpha(),
                    'zorder': line.get_zorder()
                })

            # -------------------------
            # 2. Store scatters data
            # -------------------------
            scatters_data = []
            for sc in ax.collections:
                if isinstance(sc, mcollections.PathCollection) and not isinstance(sc, mquiver.Quiver):
                    scatters_data.append({
                        'offsets': sc.get_offsets(),
                        'sizes': sc.get_sizes(),
                        'facecolors': sc.get_facecolors(),
                        'edgecolors': sc.get_edgecolors(),
                        'alpha': sc.get_alpha(),
                        'zorder': sc.get_zorder()
                    })

            # -------------------------
            # 3. Store FancyArrowPatch data
            # -------------------------
            arrows_data = []
            for patch in ax.patches:
                if isinstance(patch, FancyArrowPatch):
                    arrows_data.append({
                        'posA': patch.get_path().vertices[0],   # start
                        'posB': patch.get_path().vertices[-1],  # end
                        'arrowstyle': patch.get_arrowstyle(),
                        'color': patch.get_edgecolor(),
                        'linewidth': patch.get_linewidth(),
                        'alpha': patch.get_alpha(),
                        'zorder': patch.get_zorder(),
                        'mutation_scale': patch.get_mutation_scale()
                    })

            # -------------------------
            # 4. Clear axes
            # -------------------------
            ax.clear()

            # -------------------------
            # 5. Recreate quiver
            # -------------------------
            x = np.linspace(*xlim, grid.shape[0])
            y = np.linspace(*ylim, grid.shape[1])
            xmesh = np.meshgrid(x, y, indexing='ij')
            X, Y = self(*xmesh, *args)
            mag = np.sqrt(X**2 + Y**2)
            if scaled:
                ax.quiver(*xmesh, X/mag, Y/mag, mag, **kwargs)
            else:
                ax.quiver(*xmesh, X/mag, Y/mag, **kwargs)

            # -------------------------
            # 6. Re-add old lines
            # -------------------------
            for ld in lines_data:
                ax.plot(ld['x'], ld['y'],
                        color=ld['color'],
                        linestyle=ld['linestyle'],
                        linewidth=ld['linewidth'],
                        marker=ld['marker'],
                        markersize=ld['markersize'],
                        alpha=ld['alpha'],
                        zorder=ld['zorder'])

            # -------------------------
            # 7. Re-add old scatters
            # -------------------------
            for sd in scatters_data:
                ax.scatter(sd['offsets'][:, 0], sd['offsets'][:, 1],
                        s=sd['sizes'],
                        facecolors=sd['facecolors'],
                        edgecolors=sd['edgecolors'],
                        alpha=sd['alpha'],
                        zorder=sd['zorder'])

            # -------------------------
            # 8. Re-add old FancyArrowPatches
            # -------------------------
            for ad in arrows_data:
                arrow = FancyArrowPatch(ad['posA'], ad['posB'],
                                        arrowstyle=ad['arrowstyle'],
                                        color=ad['color'],
                                        linewidth=ad['linewidth'],
                                        alpha=ad['alpha'],
                                        zorder=ad['zorder'],
                                        mutation_scale=ad['mutation_scale'])
                ax.add_patch(arrow)

            # -------------------------
            # 9. Restore limits
            # -------------------------
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlabel(str(self.xvar), fontsize=15)
        ax.set_ylabel(str(self.yvar), fontsize=15, rotation=0)

        xmesh = grid.x_mesh()
        X, Y = self(*xmesh, *args)
        mag = np.sqrt(X**2+Y**2)

        if (scaled):
            ax.quiver(*xmesh, X/mag, Y/mag, mag, **kwargs)
        else:
            ax.quiver(*xmesh, X/mag, Y/mag, **kwargs)
        fig.canvas.mpl_connect('button_release_event', update)
        return fig, ax, lambda : update(None)
    
    def plot_line(self, line: Line2D, grid: grids.Grid, n=400, *args, **kwargs):
        fig, ax, upd = VectorField2D.plot(self, grid, *args, **kwargs)
        u = np.linspace(*line.lims, n)
        ax.plot(line.x(u), line.y(u))
        return fig, ax, upd
    
    def plot_circle(self, q, r, grid: grids.Grid, n=400, *args, **kwargs):
        c = Circle(r, q)
        return VectorField2D.plot_line(self, c, grid, n, *args, **kwargs)