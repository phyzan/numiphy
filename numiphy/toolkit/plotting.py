from __future__ import annotations
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.animation as mpl_anim
from matplotlib.patches import FancyArrowPatch
from matplotlib.collections import LineCollection
from ..findiffs import grids
import sys, os
from ..toolkit import tools
import numpy as np
from typing import Dict
from matplotlib.figure import Figure as Fig
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot(arr: np.ndarray, grid: grids.Grid, ax=None, **kwargs):
    """
    Returns (fig, ax) for 1D or (fig, ax, cbar) for 2D.
    Uses pcolormesh for 2D unless use_imshow=True is passed.
    """
    figsize = kwargs.pop('figsize', None)
    aspect  = kwargs.pop('aspect', None)
    use_imshow = kwargs.pop('use_imshow', False)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        if aspect is not None:
            ax.set_aspect(aspect)
    else:
        fig = None

    x = grid.x_mesh()

    if arr.ndim == 1:
        ax.plot(*x, arr, **kwargs)
        return (fig, ax) if fig is not None else (ax,)

    elif arr.ndim == 2:
        # Plotting
        if use_imshow:
            X, Y = x
            extent = (X.min(), X.max(), Y.min(), Y.max())
            im = ax.imshow(arr, extent=extent, origin='lower',
                           aspect=aspect if aspect is not None else 'auto', **kwargs)
            mesh = im
        else:
            mesh = ax.pcolormesh(*x, arr, **kwargs)
            # pcolormesh will use data coordinates

        # Create a colorbar axis that matches the axes height
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.06)  # tweak size/pad as desired
        cbar = fig.colorbar(mesh, cax=cax)

        return (fig, ax, cbar) if fig is not None else (ax, cbar)

    else:
        raise NotImplementedError("3D plotting not implemented")



def animate(varname: str, f: np.ndarray, duration: float, save: str, grid: grids.Grid, axis=-1, display = True, **kwargs):
    '''
    TO INCLUDE:
        - axes = 2d or 3d
        - correct the line  ax.pcolormesh(*x, f[self.grid.slice(axis, i)].reshape((""""""y.shape[0], x.shape[0]"""""""")), norm=norm, shading='auto', cmap=cmap) if needed (is it y.shape, x.shape or x.shape, y.shape)
    '''
    def default_base():
        ax.clear()
        ax.set_xscale(kwargs['xscale'])
        ax.set_yscale(kwargs['yscale'])
        ax.set_title(kwargs['title'])
        ax.set_xlabel(kwargs['xlabel'])
        ax.set_ylabel(kwargs['ylabel'])

    def label(i):
        return f"{varname} = {tools.format_number(grid.x[axis][i])} {kwargs['unit']}"
    default_args = dict(title = None, figsize = None, axes='2d', xscale='linear', yscale='linear', unit = '', xlabel=None, ylabel=None, cmap=None, dpi=200, plot=[], labels=[])
    for defarg in default_args:
        if defarg not in kwargs:
            kwargs[defarg] = default_args[defarg]

    fps = f.shape[axis]/duration

    if 'lims' not in kwargs:
        if f.ndim == 2:
            df = f.max() - f.min()
            lims = f.min()-df/8, f.max()+df/8
        else:
            lims = f.min(), f.max()
    else:
        lims = kwargs['lims']
    fig, ax = plt.subplots(figsize = kwargs['figsize'])
    if f.ndim == 2:
        def update(i):
            if display:
                mes = 'Animating: {} %'.format('%.3g' % (i/f.shape[axis]*100))
                tools.fprint(mes)
            default_base()
            ax.plot(*x, f[grid.slice(axis, i)], label=label(i), linewidth=0.5)
            for j in range(len(kwargs['plot'])):
                ax.plot(*x, kwargs['plot'][j], label=kwargs['label'][j])
            if kwargs['plot']:
                ax.legend()
            ax.set_ylim(*lims)
            
            return [ax]
    elif f.ndim == 3:
        norm = plt.Normalize(*lims)
        sm = cm.ScalarMappable(cmap=kwargs['cmap'], norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        def update(i):
            if display:
                mes = 'Animating: {} %'.format('%.3g' % (i/f.shape[axis]*100))
                tools.fprint(mes)
            default_base()
            ax.pcolormesh(*x, f[grid.slice(axis, i)].reshape((x[0].shape[0], x[1].shape[0]), order='F').T, label=label(i), norm=norm, shading='auto', cmap=kwargs['cmap'])
            return ax, cbar
    else:
        raise NotImplementedError('Cannot animate in higher than 2 dimensions')

    x = grid.without_axis(axis).x
    return _process_animation(frames=f.shape[axis], update=update, fig=fig, fps=fps, dpi = kwargs['dpi'], save=save)


def _process_animation(frames, update, fig, fps, dpi, save: str):
    interval = 1000/fps
    ani = mpl_anim.FuncAnimation(fig, update, frames=frames, interval=interval, save_count=0)
    ani.save(sys.path[0]+f'/{save}', fps=fps, dpi=dpi, writer='ffmpeg')
    return ani



class Project:

    fontsize: Dict[str, int]
    ftype: str
    dpi: int
    figsize: tuple[float, float]
    grid: dict
    aspect: str
    folder: str

    artist_types = 'scatters', 'lines', 'fills'
    general_fig_params = 'figsize', 'fontsize', 'aspect', 'grid', 'ftype', 'dpi'
    
    __slots__ = ('folder',) + general_fig_params + artist_types

    def __init__(self, folder=None, **kwargs):
        if folder is None:
            folder = os.getcwd()
        else:
            folder = folder
        tools.try_create(folder)
        self.folder = folder
        defdir = os.path.join(folder, 'defaults')
        dflts = tools.try_read(defdir)
        if dflts is None:
            dflts = kwargs
        
        dflts.update(kwargs)
        for attr in dflts:
            setattr(self, attr, dflts[attr])

        tools.write_binary_data(defdir, {item: getattr(self, item) for item in self.__slots__[1:]})
        
        for f in [self.data_folder, self.fig_folder]:
            tools.try_create(f)

    @property
    def figparams(self):
        return {attr: getattr(self, attr) for attr in self.general_fig_params}

    @property
    def data_folder(self):
        return os.path.join(self.folder, 'figdata')
    
    @property
    def fig_folder(self):
        return os.path.join(self.folder, 'figures')
    
    def savefig(self, fig: Figure, override_fig_params = True):
        fig = fig.copy()
        if override_fig_params:
            fig.set(**self.figparams)
        fig.save(self.fig_folder)
        fig.savedata(self.data_folder)

    def figure(self, name, title='', xlabel='', ylabel='', yrot=90, xlims=(None, None), ylims=(None, None), square=False):
        kwargs = dict(title=title, xlabel=xlabel, ylabel=ylabel, yrot=yrot, xlims=xlims, ylims=ylims, **self.figparams)
        if square:
            return SquareFigure(name, **kwargs)
        else:
            return Figure(name, **kwargs)

    def redo_all(self):
        figs = self.load_saved_figs()
        for fig in figs:
            self.savefig(fig)

    def load_saved_figs(self):
        figs = os.listdir(self.data_folder)
        figlist: list[Figure] = []
        for name in figs:
            fig = Figure.load(self.data_folder, name)
            figlist.append(fig)
        return figlist
    
    def scatter(self, x, y, **kwargs):
        args = self.scatters.copy()
        args.update(kwargs)
        return ScatterPlot(x=x, y=y, **args)

    def line(self, x, y, **kwargs):
        args = self.lines.copy()
        args.update(kwargs)
        return LinePlot(x=x, y=y, **args)


class Figure:

    fontsize: Dict[str, int]
    ftype: str
    dpi: int
    figsize: tuple[float, float]
    grid: dict
    aspect: str

    xlims: tuple[float, float]
    ylims: tuple[float, float]
    yrot: float
    legend: bool

    _data = ('title', 'xlabel', 'ylabel', 'yrot', 'xlims', 'ylims', 'legend') + Project.general_fig_params

    __slots__ = ('name', '_artists') + _data

    def __init__(self, name: str, **kwargs):
        self.name = name
        self._artists: list[Artist] = []
        self.set_default_params()
        for kw in kwargs:
            setattr(self, kw, kwargs[kw])

    @property
    def artists(self):
        return self._artists

    def set_default_params(self):
        self.fontsize = dict(title=20, label=15, ticks=10)
        self.ftype = 'png'
        self.dpi = 300
        self.figsize = (6, 6)
        self.grid = dict(visible=True, zorder=0)
        self.aspect = 'auto'
        self.xlims = None, None
        self.ylims = None, None
        self.xlabel = ''
        self.ylabel = ''
        self.yrot = 90
        self.legend = True
        self.title = ''

    @property
    def parameters(self):
        res = {}
        for arg in self._data:
            res[arg] = getattr(self, arg)
        return res

    def set(self, **kwargs):
        for kw in kwargs:
            setattr(self, kw, kwargs[kw])

    def add(self, art: Artist):
        self._artists.append(art)

    def save(self, folder):
        fig, ax = self.plot()
        fig_path = os.path.join(folder, f"{self.name}.{self.ftype}")
        fig.savefig(fig_path, bbox_inches='tight', dpi=self.dpi)
        plt.close()
        
    def savedata(self, folder):
        file = os.path.join(folder, self.name)
        artdata = [(artist.__class__.__name__, artist.kwargs) for artist in self._artists]
        tools.write_binary_data(file, [self.__class__.__name__, self.parameters, artdata])

    @staticmethod
    def load(folder, name):
        path = os.path.join(folder, name)
        cls, params, artdata = tools.try_read(path, error=True)
        artists: list[Artist] = []
        for cl, data in artdata:
            artists.append(eval(cl)(**data))
        
        fig: Figure = eval(cls)(name, **params)
        fig._artists = artists
        return fig
    
    def draw(self, ax: Axes):
        ax.clear()
        ax.set_aspect(self.aspect, adjustable='box')
        ax.grid(**self.grid)
        ax.set_title(self.title, fontsize = self.fontsize['title'])
        ax.set_xlabel(self.xlabel, fontsize = self.fontsize['label'])
        ax.set_ylabel(self.ylabel, fontsize = self.fontsize['label'], rotation=self.yrot)
        ax.tick_params(labelsize=self.fontsize['ticks'])
        for artist in self.artists:
            artist.apply(ax)
        ax.set_xlim(*self.xlims)
        ax.set_ylim(*self.ylims)
        if self.legend and ax.get_legend_handles_labels()[0]:
            ax.legend(loc='best')

    def plot(self):
        fig, ax = plt.subplots(figsize=self.figsize)
        self.draw(ax)
        fig.tight_layout()
        return fig, ax

    def copy(self):
        fig = self.__class__(self.name, **self.parameters)
        fig._artists = [artist.copy() for artist in self._artists]
        fig.set(**self.parameters)
        return fig


class SquareFigure(Figure):

    def __init__(self, name: str, **kwargs):
        self.name = name
        self._artists: list[Artist] = []
        self.set_default_params()
        for kw in kwargs:
            if kw == 'figsize':
                assert kwargs[kw][0] == kwargs[kw][1]
            setattr(self, kw, kwargs[kw])

    def draw(self, ax: Axes):
        super().draw(ax)
        if self.aspect == 'equal':
            xlims = ax.get_xlim()
            ylims = ax.get_ylim()
            xm = (xlims[0]+xlims[1])/2
            ym = (ylims[0]+ylims[1])/2
            dx = xlims[1]-xlims[0]
            dy = ylims[1]-ylims[0]
            d = max(dx, dy)
            ax.set_xlim(xm-d/2, xm+d/2)
            ax.set_ylim(ym-d/2, ym+d/2)


class Artist:

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def apply(self, ax: Axes):...
    
    @property
    def params(self):
        return self.kwargs.copy()
    
    def copy(self)->Artist:
        return self.__class__(**self.kwargs)


class PointCollection(Artist):

    def __init__(self, **kwargs):
        Artist.__init__(self, **kwargs)
        self.kwargs['x'] = [float(i) for i in kwargs['x']]
        self.kwargs['y'] = [float(i) for i in kwargs['y']]

    @property
    def xcoords(self)->list[float]:
        return self.kwargs['x']

    @property
    def ycoords(self)->list[float]:
        return self.kwargs['y']
    
    @property
    def params(self):
        d = Artist.params.__get__(self)
        d.pop('x')
        d.pop('y')
        return d
    
    def add(self, x, y):
        self.xcoords.append(x)
        self.ycoords.append(y)

    def clear(self):
        self.xcoords.clear()
        self.ycoords.clear()

    def isempty(self):
        return not self.xcoords
    
    def all_coords(self):
        return [(x, y) for x, y in zip(self.xcoords, self.ycoords)]
    
    def copy(self):
        return self.__class__(x=self.xcoords.copy(), y=self.ycoords.copy(), **self.params)
    

class LinePlot(PointCollection):

    def apply(self, ax):
        return ax.plot(self.xcoords, self.ycoords, **self.params)


class ScatterPlot(PointCollection):

    def apply(self, ax):
        return ax.scatter(self.xcoords, self.ycoords, **self.params)


class ErrorPlot(PointCollection):

    def __init__(self, **kwargs):

        if 'yerr' in kwargs:
            if kwargs['yerr'] is not None:
                kwargs['yerr'] = [float(i) for i in kwargs['yerr']]
        if 'xerr' in kwargs:
            if kwargs['xerr'] is not None:
                kwargs['xerr'] = [float(i) for i in kwargs['xerr']]
        super().__init__(**kwargs)

    def apply(self, ax):
        return ax.errorbar(self.xcoords, self.ycoords, **self.params)


class LineCollectionPlot(Artist):

    def apply(self, ax):
        item = LineCollection(**self.kwargs)
        ax.add_collection(item)


class QuiverPlot(Artist):

    def __init__(self, x: np.ndarray, y: np.ndarray, X: np.ndarray, Y:np.ndarray, mag=None, **kwargs):
        super().__init__(**kwargs)
        self.args = (x, y, X, Y)
        if mag is not None:
            self.args += (mag,)

    def apply(self, ax):
        return ax.quiver(*self.args, **self.kwargs)
    
    def copy(self):
        return QuiverPlot(*[arg.copy() if hasattr(arg, '__iter__') else arg for arg in self.args], **self.kwargs)


class Arrow(Artist):

    def apply(self, ax):
        arrow = FancyArrowPatch(**self.kwargs)
        return ax.add_patch(arrow)


def gradient_line(x_array, y_array, cmap='coolwarm', linewidth=None, by_arclength=False):
    """
    Create a LineCollection for a line that gradually changes color.

    Parameters
    ----------
    x_array, y_array : 1D arrays
        Coordinates of the line.
    cmap : str or Colormap, default 'coolwarm'
        Colormap for the gradient (0=start, 1=end).
    linewidth : float, default 2
        Width of the line.
    by_arclength : bool, default False
        If True, color is parameterized by arc length instead of index.

    Returns
    -------
    LineCollection
        A matplotlib LineCollection that can be added to an axes.
    """
    x_array = np.asarray(x_array)
    y_array = np.asarray(y_array)

    points = np.array([x_array, y_array]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    if by_arclength:
        dx = np.diff(x_array)
        dy = np.diff(y_array)
        s = np.concatenate([[0], np.cumsum(np.sqrt(dx**2 + dy**2))])
        values = (s - s.min()) / (s.max() - s.min())
    else:
        values = np.linspace(0, 1, len(segments))

    lc = LineCollectionPlot(
        segments=segments,
        cmap=cmap,
        norm=plt.Normalize(0, 1),
        array=values,
        linewidths=linewidth
    )
    return lc


'''
e.g.

fontsize = dict(label=10, title=10, ticks=10)
lines = dict(linewidth = 2, zorder=2)
scatters = dict(s=1, linewidth=0, zorder=3)
fills = dict(zorder=0)

proj = Project(folder='MyProject', figsize=(5, 5), fontsize=fontsize, aspect='equal', grid=dict(visible=True, zorder=1), ftype='png', dpi=300, scatters=scatters, lines=lines, fills=fills)

'''