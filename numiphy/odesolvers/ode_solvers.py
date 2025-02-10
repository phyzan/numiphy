from __future__ import annotations

from ..symlib import expressions as sym
import numpy as np
import time
from typing import Callable, Literal, Tuple, Union
from ..toolkit import tools
import importlib.util
import subprocess, glob, os, sys
from functools import cached_property
import tempfile
import importlib.machinery


def _bisect(f, a, b, tol = 1e-12):
    #TODO
    #in the future, maybe include the option with tol = 0,
    #where the loop stops only when a and b become the same due to
    #machine precision.
    err = 2*tol
    a, b = float(a), float(b)
    if f(a) * f(b) > 0:
        raise ValueError('Bisection failed: There might be no root in the given interval')
    while err > tol:
        c = (a+b)/2
        if c == a or c == b:
            break
        fm = f(c)
        if f(a)*fm > 0:
            a = c
        else:
            b = c
        err = abs(b-a)
    return [a, c, b]

def bisect(f, a, b, tol = 1e-12):
    return _bisect(f, a, b, tol)[1]

def bisectleft(f, a, b, tol = 1e-12):
    return _bisect(f, a, b, tol)[0]

def bisectright(f, a, b, tol = 1e-12):
    return _bisect(f, a, b, tol)[2]

Bisect = _bisect

def _suffix():
    ext_sfxs = importlib.machinery.EXTENSION_SUFFIXES
    so_lib_sfxs = [suffix for suffix in ext_sfxs if suffix.endswith('.so')]
    return so_lib_sfxs[0] if so_lib_sfxs else None

class BisectionRegion:

    def __init__(self, xmin, xmax, checker=None):
        self.xmin = xmin
        self.xmax = xmax
        if checker is None:
            checker = lambda root: True
        self.isroot = checker

    def subdivide(self):
        return BisectionRegion(self.xmin, (self.xmin+self.xmax)/2, self.isroot), BisectionRegion((self.xmin+self.xmax)/2, self.xmax, self.isroot)
    
    def get_root(self, f, nmax=5, err=1e-8):
        if nmax == 0:
            return None
        elif f(self.xmin) * f(self.xmax) < 0:
            print(f'Possible root in the interval {self.xmin, self.xmax}')
            root = bisect(f, self.xmin, self.xmax, err)
            if self.isroot(root):
                return root
            else:
                print('Root discarded')
        
        region1, region2 = self.subdivide()
        r1 = region1.get_root(f, nmax-1, err)
        r2 = region2.get_root(f, nmax-1, err)
        if r1 is not None:
            return r1
        elif r2 is not None:
            return r2
        else:
            return None


def euler_step(df, t, f, dt, *args):
    '''
    Advanve a function by a single step using the explicit euler formula
    '''
    return f + dt*df(t, f, *args)

def RK2_step(df, t, f, dt, *args):
    '''
    Advanve a function by a single step using the 2nd order Runge-Kutta formula
    '''
    k1 = df(t, f, *args)
    k2 = df(t+dt/2, f+dt*k1/2, *args)
    return f + dt*k2


def RK4_step(df, t, f, dt, *args):
    '''
    Advanve a function by a single step using the 4th order Runge-Kutta formula
    '''
    k1 = df(t, f, *args)
    k2 = df(t+dt/2, f+dt*k1/2, *args)
    k3 = df(t+dt/2, f+dt*k2/2, *args)
    k4 = df(t+dt, f+dt*k3, *args)
    return f + dt/6*(k1 + 2*k2 + 2*k3 + k4)

def RK7_step(df, t, f, dt, *args):
    k1 = dt*df(t, f, *args)
    k2 = dt*df(t+dt/12, f+k1/12, *args)
    k3 = dt*df(t+dt/12, f+(11*k2-10*k1)/12, *args)
    k4 = dt*df(t+dt/6, f+k3/6, *args)
    k5 = dt*df(t+dt/3, f+(157*k1-318*k2+4*k3+160*k4)/9, *args)
    k6 = dt*df(t+dt/2, f+(-322*k1+199*k2+108*k3-131*k5)/30, *args)
    k7 = dt*df(t+8*dt/12, f+3158*k1/45-638*k2/6-23*k3/2+157*k4/3+157*k6/45, *args)
    k8 = dt*df(t+10*dt/12, f-53*k1/14+38*k2/7-3*k3/14-65*k5/72+29*k7/90, *args)
    k9 = dt*df(t+dt, f+56*k1/25 + 283*k2/14-119*k3/6-26*k4/7-13*k5/15+149*k6/32-25*k7/9+27*k8/25, *args)
    return f + (41*k1 + 216*k4 + 27*k5 + 272*k6 + 27*k7 + 216*k8 + 41*k9)/840

def cromer_step(df, t, f, dt, *args):
    '''
    Advanve a Hamiltonian system of equations by a single step using the Euler-Cromer formula
    '''
    nd = f.shape[0]//2
    v = f[nd:] + dt*df(t, f, *args)[nd:]
    x = f[:nd] + dt*v
    return np.array([*x, *v])


def verlet_step(df, t, f, dt, *args):
    '''
    Advanve a Hamiltonian system of equations by a single step using the velocity-Verlet formula
    '''
    nd = f.shape[0]//2
    a_now = df(t, f, *args)[nd:]
    x = f[:nd] + f[nd:]*dt + 0.5*a_now*dt**2
    a_next = df(t+dt, np.array([*x, *f[nd:]]), *args)[nd:]
    v = f[nd:] + (a_next+a_now)*dt/2
    return np.array([*x, *v])

# def verlet_step(xdot, vdot, t, x, v, dt, *args):
#     '''
#     Advanve a Hamiltonian system of equations by a single step using the velocity-Verlet formula
#     '''
#     a = vdot(t, x, *args)
#     x = x + xdot(t, v, *args)*dt + 0.5*a*dt**2
#     v = v + (a+vdot(t+dt, x, *args))*dt/2
#     return np.array([x, v])

class Updater:
    def __init__(self, update: Callable, mask: Callable):
        self.update = update
        self.mask = mask

    def advance(self, df, t, f, dt, *args):
        fnew = self.update(df, t, f, dt, *args)
        tnew = t + dt
        return self.mask(tnew, fnew)
    


class ODE:

    def solve(self, ics: tuple[float, np.ndarray], t, dt, **kwargs)->OdeResult:...

    def solve_all(self, ics: list[tuple[float, np.ndarray]], t, dt, **kwargs)->list[OdeResult]:...

    def copy(self)->ODE:...
    

class PythonicODE(ODE):

    '''
    Basic class for defining an Ordinary Differential Equation, or system of ODE's of the form:

    df/dt = F(t, f)
    
    where f can represent a vector for a system of ODE's
    '''
    methods = ('euler', 'RK2', 'RK4', 'RK7')
    literal = Literal['euler', 'RK2', 'RK4', 'RK7']

    def __init__(self, df: Callable):
        '''
        Parameters
        -------------

        df: Callable. It must take 2 parameters, df(t, f)
        ics: tuple of initial conditions (t0, f0)

        f and f0 must be vectors of the same shape as the shape returned by df(t, f)
        '''
        self.df = df

    def copy(self):
        return PythonicODE(self.df)

    def custom_solver(self, ics, t, dt, update, lte, args=(), getcond=None, breakcond = None, err = 0., max_frames = -1, display = False, mask = None, thres = 1e-30, checknan=True):
        '''
        This is the core algorithm that solves a system of ode's using any available method.
        It is an internal function and should not be called by the user yet. It will be easy to
        implement a feature where the user defines their own method of advancing by a single step,
        along with the local truncation error, and this function will be available for the user.
        '''
        if mask is not None:
            update = Updater(update, mask).advance
        if breakcond is None:
            breakcond = lambda *args: False
        cond_sat = False
        _cond_sat = False
        t0 = ics[0]

        if dt*t0 > t*dt:
            raise ValueError(f'The ode integration direction from {t0} does not lead to {t}')
        ti, f = ics
        x_arr, f_arr = [ti], [f]
        pow = 1/lte
        k = 1
        _dir = np.sign(dt)
        diverges = False

        t1 = time.time()
        with np.errstate(all='ignore'):
            while _dir*ti < _dir*t and not cond_sat:
                #determine step size first
                rel_err = 2*err
                while rel_err > err:
                    f_single = update(self.df, ti, f, dt, *args)

                    f_half = update(self.df, ti, f, dt/2, *args)
                    f_double = update(self.df, ti+dt/2, f_half, dt/2, *args)
                    
                    if thres > 0.0 and isinstance(f_double, np.ndarray):
                        msk = np.abs(f_double) > thres
                        errs = np.abs((f_single - f_double)[msk]/np.abs(f_double)[msk])
                        if len(errs) > 0:
                            rel_err = np.max(errs)
                            if rel_err > 0:
                                dt = 0.9* dt * (err/rel_err)**pow
                            else:
                                break
                        else:
                            break
                    else:
                        rel_err = np.max(np.abs((f_single - f_double)/np.abs(f_double)))
                        if rel_err != 0:
                            dt = 0.9* dt * (err/rel_err)**pow
                if (ti+dt)*_dir > t*_dir:
                    dt = t - ti
                #step size determined
                f_new = update(self.df, ti, f, dt, *args)
                if checknan:
                    if np.any(np.logical_or(np.isnan(f_new), np.isinf(f_new))):
                        diverges = True
                        break
                if breakcond(ti, ti+dt, f, f_new):
                    cond_sat = True
                    dt, f_new = self._get_step(update, breakcond, f, ti, dt, args)
                elif getcond is not None:
                    if getcond(ti, ti+dt, f, f_new):
                        _cond_sat = True
                        dt, f_new = self._get_step(update, getcond, f, ti, dt, args)


                f = f_new
                ti = ti + dt
                if getcond is not None:
                    if _cond_sat:
                        f_arr.append(f)
                        x_arr.append(ti)
                        _cond_sat = False
                        k += 1
                elif max_frames == -1 or abs(ti-t0) * (max_frames-1) >= k*abs(t-t0):
                    f_arr.append(f)
                    x_arr.append(ti)
                    k += 1
                

                if display:
                    mes = '{} %'.format('%.3g' % ((ti-t0)/(t-t0)*100))
                    tools.fprint(mes)
                if k == max_frames:
                    break
        t2 = time.time()
        x_arr, f_arr = np.array(x_arr), np.array(f_arr)

        return OdeResult(x_arr, f_arr, t2-t1, diverges)

    def _get_step(self, update, cond, f1, ti, dt, args):
        def h(t):
            _f = update(self.df, ti, f1, t-ti, *args)
            return cond(ti, ti+dt, f1, _f)-0.5
        dt = bisectright(h, ti, ti+dt) - ti
        return dt, update(self.df, ti, f1, dt, *args)

    def solve(self, ics: tuple[float, np.ndarray], t: float, dt: float, method: literal='RK4', **kwargs)->OdeResult:
        '''
        Solve the ode

        Parameters
        ---------------
        t: Advance the ode until its parameter reaches this value
        dt: Step size
        method: Explicit numerical method to solve the ode (see self.methods)
        kwargs:
            args: tuple to be passes into df(t, f, *args)
            breakcond(t, f): Callable function that takes 2 parameters, the ode parameter and function.
                It represents the break condition of the ode integration.
                If given, the ode integration stops when breakcond(t, f) == 0. Default is None
            err: The highest allowed relative error in each step. If it is non-zero,
                the given method is used, but with an adaptive stepsize that minimizes the error
                to the given value. The default is zero (not adaptive).
            max_frames: The maximum number of instances of the solution as it is integrated to be returned. Default is -1 (all of them).
            display: Default is True. On every timestep, the completed percentage of the integration is printed.
                It is calculated assuming that the integrator stops when the integration variable reaches "t". This
                percentage may be completely misleading if a break condition "breakcond" has been given.
            mask: If given, the value of the function in each time step is filtered through this function
            thres: Applies only for a system of equations. All function values above this value will be used in the calculation of the relative error (if given). This is used to avoid overflow errors. Default is zero.
            checknan (bool): After performing every step, the code checks if the result is nan. If it is, the code exits.

        Returns
        ---------------
        Array of the parameter values in each step
        Array of the function values in each step. This can be an array of arrays, depending on how the user defined the initial conditions.
        '''
        return getattr(self, method)(ics=ics, t=t, dt=dt, **kwargs)
    
    def solve_all(self, ics, t, dt, **kwargs)->list[OdeResult]:
        res = []
        for ics_i in ics:
            res.append(self.solve(ics_i, t, dt, **kwargs))
        return res

    def euler(self, ics, t, dt, **kwargs):
        '''
        Solve the system of ode's using the explicit euler method.

        see "solve"

        '''
        return self.custom_solver(ics, t, dt, euler_step, lte=2, **kwargs)

    def RK2(self, ics, t, dt, **kwargs):
        '''
        Solve the system of ode's using the 2nd order Runge-Kutta method

        see "solve"
        '''
        return self.custom_solver(ics, t, dt, RK2_step, lte=3, **kwargs)
    
    def RK4(self, ics, t, dt, **kwargs):
        '''
        Solve the system of ode's using the 4th order Runge-Kutta method

        see "solve"
        '''
        return self.custom_solver(ics, t, dt, RK4_step, lte=5, **kwargs)

    def RK7(self, ics, t, dt, **kwargs):
        '''
        Solve the system of ode's using the 7th order Runge-Kutta method

        see "solve"
        '''
        return self.custom_solver(ics, t, dt, RK7_step, lte=8, **kwargs)


# class HamiltonianSystem(PythonicODE): #needs different name, this class already exists
#     '''
#     System of the form

#     dx/dt = xdot(t, v)
#     dv/dt = vdot(t, x)

#     '''

#     methods = PythonicODE.methods + ('cromer', 'verlet')
#     literal = Union[PythonicODE.literal, Literal['cromer', 'verlet']]


#     def __init__(self, vdot):
#         '''
#         Parameters
#         ------------

#         xdot(t, v): Callable
#         vdot(t, x): Callable
#         ics: (t0, x0, v0), tuple. The initial conditions

#         The parameters are translated to initialize an ODE object
#         '''
#         self.vdot = vdot

#         super().__init__(self._df)

#     def _df(self, t, f, *args):
#         return np.array([*f[self.nd:], *self.vdot(t, f[:self.nd], *args)])

#     def cromer(self, t: float, dt: float, **kwargs):
#         '''
#         Solve the system of ode's using the Euler-Cromer method

#         Parameters
#         -------------
#         t: Advance the system of ode's until their parameter reaches this value
#         dt: Stepsize

#         Returns
#         ---------------
#         Array of the parameter values in each step
#         Array of x, v values in each step. The return shape is (nt, 2, ...), where nt is the number of steps
#         '''

#         return self.custom_solver(t, dt, cromer_step, lte=2, **kwargs)

#     def verlet(self, t: float, dt: float, **kwargs):
#         '''
#         Solve the system of ode's using the velocity-verlet method

#         see "cromer" for parameters and return values
#         '''
        
#         return self.custom_solver(t, dt, verlet_step, lte=3, **kwargs)

def import_lowlevel_module(directory: str, module_name):
    so_file = os.path.join(directory, module_name)
    so_full_path = so_file + _suffix()
    spec = importlib.util.spec_from_file_location(module_name, so_full_path)
    temp_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(temp_module)
    return temp_module

def get_virtualenv_path():
    """Used to work out path to install compiled binaries to."""
    if hasattr(sys, 'real_prefix'):
        return sys.prefix

    if hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix:
        return sys.prefix

    if 'conda' in sys.prefix:
        return sys.prefix

    return None

class LowLevelODE(ODE):

    def copy(self)->LowLevelODE:...


class SymbolicOde:

    _counter = 0

    def __init__(self, *odesys: sym.Expr, symbols: list[sym.Variable], args: tuple[sym.Variable,...]=()):
        given = symbols + list(args)
        tvar = symbols[0]
        assert tools.all_different(given)
        odesymbols = []
        for ode in odesys:
            for arg in ode.variables:
                if arg not in odesymbols:
                    odesymbols.append(arg)

        if len(odesys) != len(symbols)-1:
            raise ValueError('')
        if tvar in odesymbols:
            assert len(odesymbols) == len(given)
        else:
            assert len(odesymbols) <= len(given) - 1
        self.odesys = odesys
        self.symbols = symbols
        self.args = args
        self.codegen = sym.CodeGenerator(*odesys, symbols = symbols, args=args)

    def to_python(self):
        df = self.codegen.python_callable(ode_style=True)
        return PythonicODE(df)
    
    def to_lowlevel(self, stack=True)->LowLevelODE:
        return self._lowlevel_stack.copy() if stack else self._lowlevel_heap.copy()
    
    def generate_cpp_file(self, directory, module_name, stack: bool):
        if not os.path.exists(directory):
            raise RuntimeError(f'Directory "{directory} does not exist"')
        code = self.codegen.get_cpp(ode_style=True, stack=stack)
        src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "odepack")
        cpp_code = f'#include "{os.path.join(src_path, 'pyode.hpp')}"\n\n{code}\n\n'
        cpp_code += f"PYBIND11_MODULE({module_name}, m){{\ndefine_ode_module(m, MyFunc);\n}}"
        cpp_file = os.path.join(directory, f"{module_name}.cpp")

        with open(cpp_file, "w") as f:
            f.write(cpp_code)

        return os.path.join(directory, f'{module_name}.cpp')

    def compile(self, directory: str, module_name, stack=True):

        if not os.path.exists(directory):
            raise RuntimeError(f"Cannot compile ode at {directory}: Path does not exist")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            cpp_file = self.generate_cpp_file(temp_dir, module_name, stack)

            compile_comm = f"g++ -O3 -Wall -shared -std=c++20 -fopenmp -I/usr/include/python3.12 -I/usr/include/pybind11 -fPIC $(python3 -m pybind11 --includes) {cpp_file} -o {os.path.join(directory, module_name)}$(python3-config --extension-suffix)"
            print('Compiling ODE...')
            subprocess.check_call(compile_comm, shell=True)
            print('Done')
        
    def _to_lowlevel(self, stack=True)->LowLevelODE:
        c = self.__class__._counter
        modname = f"ode_module{c}"

        with tempfile.TemporaryDirectory() as so_dir:
            self.compile(so_dir, modname, stack=stack)
            temp_module = import_lowlevel_module(so_dir, modname)

        self.__class__._counter += 1
        return temp_module.ode()
    
    @cached_property
    def _lowlevel_stack(self):
        return self._to_lowlevel(True)
    
    @cached_property
    def _lowlevel_heap(self):
        return self._to_lowlevel(False)



class OdeResult:

    def __init__(self, var_arr, f_arr, diverges, runtime):
        self.__args = (np.asarray(var_arr), np.asarray(f_arr), diverges, runtime)

    @property
    def var(self)->np.ndarray:
        return self.__args[0]
    
    @property
    def func(self)->np.ndarray:
        return self.__args[1]

    @property
    def runtime(self)->float:
        return self.__args[2]

    @property
    def diverges(self)->bool:
        return self.__args[3]


class Orbit:

    __args: tuple[ODE, np.ndarray, bool]

    def __init__(self, ode: ODE, dof):
        self.__args = (ode, np.empty((0, dof+1), dtype=np.float64), False)

    @property
    def ode(self)->ODE:
        return self.__args[0]
    
    @property
    def _data(self)->np.ndarray:
        return self.__args[1]
    
    @property
    def dof(self)->int:
        return self.__args[1].shape[1]-1
    
    @property
    def diverges(self)->bool:
        return self.__args[2]
    
    @property
    def t(self)->np.ndarray:
        return self._data[:, 0].copy()

    @property
    def f(self)->np.ndarray:
        return self._data[:, 1:].copy()

    def _set_divergence(self, arg: bool):
        self.__args = (self.ode, self._data, arg)

    def _remake(self, t0, f0):
        data = np.array([[t0, *f0]], dtype=np.float64)
        assert data.shape == (1, self.dof+1)
        self.__args = (self.ode.copy(), data, False)

    def copy(self):
        return Orbit(self.ode.copy(), self.dof)

    def clear(self):
        self.__args = (self.ode.copy(), np.empty((0, self.dof+1), dtype=np.float64), False)

    def reset(self):
        if self._data.shape[0] > 0:
            self._remake(self._data[0, 0], self._data[0, 1:])
    
    def set_ics(self, t0: float, f0: np.ndarray):
        self._remake(t0, f0)

    def integrate(self, Delta_t, dt, func = "solve", **kwargs):
        if self.diverges:
            print('Cannot integrate, orbit diverges')
        elif Delta_t<0 or dt<0:
            raise ValueError('Invalid Delta_t or dt inserted')
        elif Delta_t < dt:
            raise ValueError('Delta_t must be greater than dt')
        
        ics = self._parse_ics((self._data[-1, 0], self._data[-1, 1:]))
        res: OdeResult = getattr(self.ode, func)(ics, self._data[-1, 0]+Delta_t, dt, **kwargs)
        tarr, farr = res.var, res.func
        
        newdata = np.column_stack((tarr, farr))
        data = np.concatenate((self._data, newdata[1:]))
        self.__args = (self.ode, data, self.diverges)
        if res.diverges:
            self._set_divergence(True)
        return res
    
    def _copy_data_from(self, other: Orbit):

        if type(other) is not type(self):
            raise ValueError(f'Incompatible orbits: Cannot copy data from orbit object of type {other.__class__} to orbit object of type {self.__class__}')
        self.__args = (other.ode.copy(), other._data.copy(), other.diverges)

    def _parse_ics(self, ics):
        return (float(ics[0]), list(ics[1]))


class VariationalOrbit(Orbit):

    _logksi: list[float]

    def __init__(self, ode, dof):
        Orbit.__init__(self, ode, 2*dof)
        self._logksi = []

    @property
    def q(self):
        return self._data[:, 1:1+self.dof//2]

    @property
    def delq(self):
        return self._data[:, 1+self.dof//2:].copy()
    
    @property
    def ksi(self):
        return np.linalg.norm(self.delq, axis=1)
    
    @property
    def lyapunov(self):
        res = np.array(self._logksi[1:])/(self.t[1:]-self.t[0])
        res = np.concatenate((np.array([0.]), res))
        return self.t, res
    
    def reset(self):
        Orbit.reset(self)
        if self._logksi:
            self._logksi = [self._logksi[0]]

    def clear(self):
        Orbit.clear(self)
        self._logksi = []

    def set_ics(self, t0, f0):
        self._logksi = [0.]
        q0, delq0 = f0[:self.dof//2], f0[self.dof//2:]
        delq0 = np.array(delq0)/np.linalg.norm(delq0)
        f0 = [*q0, *delq0]
        Orbit.set_ics(self, t0, f0)

    def integrate(self, Delta_t, dt, err=1e-8, max_frames=-1):
        
        res = Orbit.integrate(self, Delta_t, dt, err=err, max_frames=max_frames)

        ksi = np.linalg.norm(res.func[:, self.dof//2:], axis=1)

        logksi = np.log(ksi) + self._logksi[-1]

        self._logksi += list(logksi[1:])

        return res

    def copy(self):
        return VariationalOrbit(self.ode.copy(), self.dof//2)
    
    def get(self, Delta_t, dt, err=1e-8, max_frames=-1, split=100):

        for _ in range(split):
            self.integrate(Delta_t/split, dt, err=err, max_frames=max_frames)
    
    def _parse_ics(self, ics):
        t0, f0 = ics
        q0 = f0[:self.dof//2]
        delq0 = f0[self.dof//2:]
        ksi = np.linalg.norm(delq0)
        delq0 = delq0/ksi
        qnew = np.concatenate((q0, delq0))
        return (t0, qnew)


class HamiltonianOrbit(Orbit):

    def __init__(self, ode, nd):
        Orbit.__init__(self, ode, 2*nd)

    @property
    def nd(self):
        return self.dof//2

    @property
    def x(self):
        return self._data[:, 1:1+self.nd].transpose()

    @property
    def p(self):
        return self._data[:, 1+self.nd:1+2*self.nd].transpose()

    def copy(self):
        return HamiltonianOrbit(self.ode.copy(), self.nd)


class VariationalHamiltonianOrbit(VariationalOrbit, HamiltonianOrbit):

    def __init__(self, ode, nd):
        VariationalOrbit.__init__(self, ode, 2*nd)

    @property
    def nd(self):
        return self.dof//4
    
    @property
    def delx(self):
        return self._data[:, 1+2*self.nd:1+3*self.nd]

    @property
    def delp(self):
        return self._data[:, 1+3*self.nd:]

    def copy(self):
        return VariationalHamiltonianOrbit(self.ode.copy(), self.nd)


class HamiltonianSystem:

    __args: tuple[sym.Expr, tuple[sym.Variable, ...], tuple[sym.Variable, ...]]

    def __init__(self, potential: sym.Expr, *variables: sym.Variable, args: tuple[sym.Variable, ...]=()):
        self.__args = (potential, variables, tuple(args))
        for v in variables:
            if len(v.name) != 1:
                raise ValueError("All variables in the dynamical system need to have exactly one letter")
            
    @property
    def nd(self):
        return len(self.variables)

    @property
    def variables(self)->tuple[sym.Variable, ...]:
        return self.__args[1]
    
    @property
    def extras(self):
        return self.__args[2]
    
    @property
    def V(self)->sym.Expr:
        return self.__args[0]
    
    @cached_property
    def H(self):
        p = self.ode_vars[self.nd:]
        T = sum([pi**2 for pi in p])/2
        return T + self.V
    
    @property
    def x(self):
        return self.variables
    
    @property
    def p(self):
        return self.ode_vars[self.nd:]
    
    @cached_property
    def rhs(self):
        xdot = [self.H.diff(pi) for pi in self.p]
        pdot = [-self.H.diff(xi) for xi in self.x]
        return xdot+pdot
    
    @cached_property
    def ode_vars(self)->tuple[sym.Variable, ...]:
        return tuple([*self.variables] + [sym.Variable('p'+xi.name) for xi in self.variables])

    @cached_property
    def varode_vars(self):
        v = list(self.ode_vars)
        for i in range(len(v)):
            v.append(sym.Variable('delta_'+v[i].name))
        return tuple(v)

    def odesys(self, variational=False):
        qdot = self.rhs.copy()

        if variational:
            qall = self.varode_vars
            for i in range(2*self.nd):
                qdot.append(sum([qdot[i].diff(qall[j])*qall[j+2*self.nd] for j in range(2*self.nd)]))

        return qdot
    
    @cached_property
    def ode(self):
        return SymbolicOde(*self.odesys(variational=False), symbols=[sym.Variable('t'), *self.ode_vars], args=self.extras)
    
    @cached_property
    def variatinal_ode(self):
        return SymbolicOde(*self.odesys(variational=True), symbols=[sym.Variable('t'), *self.varode_vars], args=self.extras)
    
    def code_generator(self, variational=False):
        q = self.ode_vars if not variational else self.varode_vars
        return sym.CodeGenerator(*self.odesys(variational), symbols=[sym.Variable('t')]+q, args=self.extras)

    def new_orbit(self, q0, lowlevel=True):
        orb = HamiltonianOrbit(self.ode.to_lowlevel(), self.nd) if lowlevel else HamiltonianOrbit(self.ode.to_python(), self.nd)
        orb.set_ics(0., q0)
        return orb

    def new_varorbit(self, q0, delq0=None, lowlevel=True):
        orb = VariationalHamiltonianOrbit(self.variatinal_ode.to_lowlevel(), self.nd) if lowlevel else VariationalHamiltonianOrbit(self.variatinal_ode.to_python(), self.nd)
        if delq0 is None:
            delq0 = [1., *((2*self.nd-1)*[0.])]
        orb.set_ics(0., [*q0, *delq0])
        return orb
