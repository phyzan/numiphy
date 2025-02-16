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
import copy


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

    def solve_all(self, params: list[dict], threads=-1)->list[OdeResult]:...

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

    def custom_solver(self, ics, t, dt, update, lte, args=(), getcond=None, breakcond = None, err = 0., cutoff_step=0., max_frames = -1, display = False, mask = None, thres = 1e-30, checknan=True):
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
        is_stiff = False


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
                                if abs(dt)< cutoff_step:
                                    is_stiff=True
                                    break
                            else:
                                break
                        else:
                            break
                    else:
                        rel_err = np.max(np.abs((f_single - f_double)/np.abs(f_double)))
                        if rel_err != 0:
                            dt = 0.9* dt * (err/rel_err)**pow
                            if abs(dt)< cutoff_step:
                                is_stiff=True
                                break
                if is_stiff:
                    break
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

        return OdeResult(var_arr=x_arr, f_arr=f_arr, diverges=diverges, is_stiff=is_stiff, runtime=t2-t1)

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
    
    def solve_all(self, params: list[dict], threads=-1)->list[OdeResult]:
        #TODO threads
        res = []
        for p in params:
            res.append(self.solve(**p))
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

def compile(cpp_path, so_dir, module_name):

    if not os.path.exists(cpp_path):
        raise RuntimeError(f"CPP file path does not exist")
    
    if not os.path.exists(so_dir):
        raise RuntimeError(f"Cannot compile ode at {so_dir}: Path does not exist")

    compile_comm = f"g++ -O3 -Wall -shared -std=c++20 -fopenmp -I/usr/include/python3.12 -I/usr/include/pybind11 -fPIC $(python3 -m pybind11 --includes) {cpp_path} -o {os.path.join(so_dir, module_name)}$(python3-config --extension-suffix)"
    print('Compiling ODE...')
    subprocess.check_call(compile_comm, shell=True)
    print('Done')


class LowLevelODE(ODE):

    def copy(self)->LowLevelODE:...

    @classmethod
    def dsolve_all(cls, data: list[tuple[LowLevelODE, dict]], threads=-1)->list[OdeResult]:...




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
            assert len(odesymbols) <= len(given)
        else:
            assert len(odesymbols) <= len(given) - 1

        self._odesys = odesys
        self._symbols = tuple(symbols)
        self.args = args

    @property
    def Nsys(self):
        return len(self._odesys)

    def ode_sys(self, variational=False)->tuple[tuple[sym.Expr, ...], tuple[sym.Variable,...]]:
        if not variational:
            return self._odesys, self._symbols
        else:
            assert not any([x.name.startswith('delta_') for x in self._symbols])
            q = self._symbols[1:]
            delq = [sym.Variable('delta_'+qi.name) for qi in q]
            n = len(self._odesys)
            var_odesys = []
            for i in range(n):
                var_odesys.append(sum([self._odesys[i].diff(q[j])*delq[j] for j in range(n)]))
            
            odesys = self._odesys + tuple(var_odesys)
            symbols = self._symbols + tuple(delq)
            return odesys, symbols
    
    def codegen(self, variational=False):
        odesys, symbols = self.ode_sys(variational=variational)
        return sym.CodeGenerator(*odesys, symbols=symbols, args=self.args)
    
    def ode(self, lowlevel=True, stack=True, variational=False):
        if lowlevel:
            return self.to_lowlevel(stack=stack, variational=variational)
        else:
            return self.to_python(variational=variational)

    def to_python(self, variational=False):
        df = self.codegen(variational).python_callable(ode_style=True)
        return PythonicODE(df)
    
    def to_lowlevel(self, stack=True, variational=False)->LowLevelODE:
        if variational:
            return self._lowlevel_stack_var.copy() if stack else self._lowlevel_heap_var.copy()
        else:
            return self._lowlevel_stack.copy() if stack else self._lowlevel_heap.copy()
    
    def generate_cpp_file(self, directory, module_name, stack: bool, variational=False):
        if not os.path.exists(directory):
            raise RuntimeError(f'Directory "{directory} does not exist"')
        code = self.codegen(variational).get_cpp(ode_style=True, stack=stack)
        src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "odepack")
        cpp_code = f'#include "{os.path.join(src_path, 'pyode.hpp')}"\n\n{code}\n\n'
        cpp_code += f"PYBIND11_MODULE({module_name}, m){{\ndefine_ode_module(m, MyFunc);\n}}"
        cpp_file = os.path.join(directory, f"{module_name}.cpp")

        with open(cpp_file, "w") as f:
            f.write(cpp_code)

        return os.path.join(directory, f'{module_name}.cpp')
    
    def compile(self, directory: str, module_name, stack=True, variational=False):
        if not os.path.exists(directory):
            raise RuntimeError(f"Cannot compile ode at {directory}: Path does not exist")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            cpp_file = self.generate_cpp_file(temp_dir, module_name, stack, variational)
            compile(cpp_file, directory, module_name)
        
    def _to_lowlevel(self, stack=True, variational=False)->LowLevelODE:
        c = self.__class__._counter
        modname = f"ode_module{c}"

        with tempfile.TemporaryDirectory() as so_dir:
            self.compile(so_dir, modname, stack=stack, variational=variational)
            temp_module = import_lowlevel_module(so_dir, modname)

        self.__class__._counter += 1
        return temp_module.ode()
    
    @cached_property
    def _lowlevel_stack(self):
        return self._to_lowlevel(stack=True, variational=False)
    
    @cached_property
    def _lowlevel_heap(self):
        return self._to_lowlevel(stack=True, variational=False)

    @cached_property
    def _lowlevel_stack_var(self):
        return self._to_lowlevel(stack=True, variational=True)
    
    @cached_property
    def _lowlevel_heap_var(self):
        return self._to_lowlevel(stack=True, variational=True)





class Base:

    __slots__ = '__dict__',

    def __init__(self, **kwargs):
        self.__dict__['_args'] = {key: kwargs[key] for key in kwargs}
    
    @property
    def _args(self)->dict:
        return self.__dict__['_args']
    
    def __getattr__(self, attr):
        return self._args[attr]
    
    def __setattr__(self, attr, value):
        raise ValueError("Class does not allow attribute setting")
    
    def _set(self, **kwargs):
        for attr in kwargs:
            if attr not in self._args:
                raise ValueError(f'{self.__class__} object has no attribute named "{attr}"')
            else:
                self._args[attr] = kwargs[attr]
    
    def _copy_data_from(self, other: Base):
        if ( type(other) is not type(self) ) or ( list(other._args.keys()) != list(self._args.keys()) ):
            raise ValueError(f'Cannot copy data from object of type "{other.__class__}" to object of type "{self.__class__}"')
        else:
            for key in self._args:
                value = other._args[key]
                if isinstance(value, (list, np.ndarray)):
                    self._args[key] = value.copy()
                else:
                    self._args[key] = value

    def clone(self):
        obj = object.__new__(type(self))
        for key in self.__dict__:
            obj.__dict__[key] = copy.deepcopy(self.__dict__[key])
        return obj


class OdeResult(Base):

    var: np.ndarray
    func: np.ndarray
    diverges: bool
    is_stiff: bool
    runtime: float

    def __init__(self, var_arr, f_arr, diverges, is_stiff, runtime):
        Base.__init__(self, var=np.asarray(var_arr), func=np.asarray(f_arr), diverges=diverges, is_stiff=is_stiff, runtime=runtime)
    

class Orbit(Base):

    ode: ODE
    data: np.ndarray
    diverges: bool
    is_stiff: bool
    
    def __init__(self, ode: SymbolicOde, lowlevel=True, stack=True):
        Orbit._init(self, ode, lowlevel=lowlevel, stack=stack)

    def _init(self, ode: SymbolicOde, lowlevel: bool, stack: bool, **kwargs):
        _ode = ode.ode(lowlevel=lowlevel, stack=stack, variational=self.is_variational)
        
        nsys = 2*ode.Nsys if self.is_variational else ode.Nsys
        Base.__init__(self, ode=_ode, data=np.empty((0, nsys+1), dtype=np.float64), diverges=False, is_stiff=False, **kwargs)
    
    @property
    def dof(self)->int:
        return self.data.shape[1]-1
    
    @property
    def t(self)->np.ndarray:
        return self.data[:, 0].copy()

    @property
    def f(self)->np.ndarray:
        return self.data[:, 1:].copy()
    
    @property
    def is_variational(self):
        return isinstance(self, VariationalOrbit)

    def newcopy(self):
        obj = self.clone()
        obj.clear()
        return obj
    
    def clear(self):
        self._set(data=self._empty(), diverges=False, is_stiff=False)

    def reset(self):
        if self.data.shape[0] > 0:
            self._remake(self.data[0, 0], self.data[0, 1:])
    
    def set_ics(self, t0: float, f0: np.ndarray):
        f0 = np.array(f0)
        if f0.shape != (self.dof,):
            raise ValueError(f"Initial conditions need to be a 1D array of size {self.dof}")
        self._remake(t0, f0)

    def current_ics(self):
        return self._parse_ics((self.data[-1, 0], self.data[-1, 1:]))

    def integrate(self, Delta_t, dt, func = "solve", **kwargs):
        if self.diverges or self.is_stiff:
            return OdeResult(self.t[-1:], self.f[-1:, :], diverges=self.diverges, is_stiff=self.is_stiff)
        elif Delta_t<0 or dt<0:
            raise ValueError('Invalid Delta_t or dt inserted')
        elif Delta_t < dt:
            raise ValueError('Delta_t must be greater than dt')
        
        if self.data.shape[0] == 0:
            raise RuntimeError('No initial conditions have been set')
        
        ics = self._parse_ics((self.data[-1, 0], self.data[-1, 1:]))
        res: OdeResult = getattr(self.ode, func)(ics, self.data[-1, 0]+Delta_t, dt, **kwargs)
        Orbit._absorb_oderes(self, res)
        return res

    def _parse_ics(self, ics):
        return (float(ics[0]), list(ics[1]))
    
    def _absorb_oderes(self, res: OdeResult):
        tarr, farr = res.var, res.func
        
        newdata = np.column_stack((tarr, farr))
        data = np.concatenate((self.data, newdata[1:]))
        self._set(data=data, diverges=res.diverges, is_stiff=res.is_stiff)
        return res

    def _empty(self):
        return np.empty((0, self.dof+1), dtype=np.float64)

    def _remake(self, t0, f0):
        data = np.array([[t0, *f0]], dtype=np.float64)
        if data.shape != (1, self.dof+1):
            raise ValueError(f"The provided initial conditions have data shape {data.shape} instead of {(1, self.dof+1)}")
        self._set(data=data, diverges=False, is_stiff=False)


class VariationalOrbit(Orbit):

    _logksi: list[float]

    def __init__(self, ode: SymbolicOde, lowlevel=True, stack=True):
        VariationalOrbit._init(self, ode, lowlevel=lowlevel, stack=stack, _logksi=[])

    @property
    def q(self):
        return self.data[:, 1:1+self.dof//2]

    @property
    def delq(self):
        return self.data[:, 1+self.dof//2:].copy()
    
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
            self._set(_logksi = [self._logksi[0]])

    def clear(self):
        Orbit.clear(self)
        self._set(_logksi=[])

    def set_ics(self, t0, f0):
        self._set(_logksi=[0.])
        q0, delq0 = f0[:self.dof//2], f0[self.dof//2:]
        delq0 = np.array(delq0)/np.linalg.norm(delq0)
        f0 = [*q0, *delq0]
        Orbit.set_ics(self, t0, f0)

    def integrate(self,  Delta_t, dt, func = "solve", **kwargs):
        res = Orbit.integrate(self, Delta_t, dt, func, **kwargs)
        self._absorb_ksi(res)
        return res
    
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
    
    def _absorb_oderes(self, res):
        Orbit._absorb_oderes(self, res)
        self._absorb_ksi(res)

    def _absorb_ksi(self, res: OdeResult):
        ksi = np.linalg.norm(res.func[:, self.dof//2:], axis=1)
        logksi = np.log(ksi) + self._logksi[-1]
        self._set(_logksi=self._logksi+list(logksi[1:]))


class HamiltonianOrbit(Orbit):

    def __init__(self, potential: sym.Expr, variables: tuple[sym.Variable, ...], args: tuple[sym.Variable, ...] = (), lowlevel=True, stack=True):
        HamiltonianOrbit._init(self, potential=potential, variables=variables, args=args, lowlevel=lowlevel, stack=stack)
    
    def _init(self, potential: sym.Expr, variables: tuple[sym.Variable, ...], args: tuple[sym.Variable, ...], lowlevel: bool, stack: bool, **kwargs):
        hs = HamiltonianSystem(potential, *variables, args=args)
        return Orbit._init(self, hs.symbolic_ode, lowlevel, stack, **kwargs)

    @property
    def nd(self):
        if self.is_variational:
            return self.dof//4
        else:
            return self.dof//2

    @property
    def x(self):
        return self.data[:, 1:1+self.nd].transpose()

    @property
    def p(self):
        return self.data[:, 1+self.nd:1+2*self.nd].transpose()


class VariationalHamiltonianOrbit(VariationalOrbit, HamiltonianOrbit):

    def __init__(self, potential: sym.Expr, variables: tuple[sym.Variable, ...], args: tuple[sym.Variable, ...] = (), lowlevel=True, stack=True):
        HamiltonianOrbit._init(self, potential=potential, variables=variables, args=args, lowlevel=lowlevel, stack=stack, _logksi=[])
    
    @property
    def delx(self):
        return self.data[:, 1+2*self.nd:1+3*self.nd]

    @property
    def delp(self):
        return self.data[:, 1+3*self.nd:]


class FlowOrbit(Orbit):

    def __init__(self, ode: SymbolicOde, lowlevel=True, stack=True):
        Orbit.__init__(self, ode, lowlevel, stack)

    @property
    def nd(self):
        if self.is_variational:
            return self.dof//2
        else:
            return self.dof

    @property
    def x(self):
        return self.data[:, 1:].transpose()


class VariationalFlowOrbit(VariationalOrbit, FlowOrbit):

    def __init__(self, ode: SymbolicOde, lowlevel=True, stack=True):
        VariationalOrbit.__init__(self, ode, lowlevel, stack)

    @property
    def x(self):
        return self.data[:, 1:1+self.nd].transpose()
    
    @property
    def delx(self):
        return self.data[:, 1+self.nd:]


class HamiltonianSystem:

    _instances: list[tuple[tuple[sym.Expr, ...], HamiltonianSystem]] = []

    __args: tuple[sym.Expr, tuple[sym.Variable, ...], tuple[sym.Variable, ...]]

    def __new__(cls, potential: sym.Expr, *variables: sym.Variable, args: tuple[sym.Variable, ...]=()):
        args = (potential, variables, args)
        for i in range(len(cls._instances)):
            if cls._instances[i][0] == args:
                return cls._instances[i][1]
        for v in variables:
            if len(v.name) != 1 or v.name == 't':
                raise ValueError("All variables in the dynamical system need to have exactly one letter, and different from 't'") 
        obj = super().__new__(cls)
        cls._instances.append((args, obj))
        return obj

    def __init__(self, potential: sym.Expr, *variables: sym.Variable, args: tuple[sym.Variable, ...]=()):
        self.__args = (potential, variables, tuple(args))
            
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
    def symbolic_ode(self):
        return SymbolicOde(*self.rhs, symbols=[sym.Variable('t'), *self.ode_vars], args=self.extras)
    
    def ode(self, lowlevel=True, stack=True, variational=False):
        return self.symbolic_ode.ode(lowlevel=lowlevel, stack=stack, variational=variational)

    def new_orbit(self, q0, lowlevel=True, stack=True):
        orb = HamiltonianOrbit(self.V, self.variables, self.extras, lowlevel=lowlevel, stack=stack)
        orb.set_ics(0., q0)
        return orb

    def new_varorbit(self, q0, delq0=None, lowlevel=True, stack=True):
        orb = VariationalHamiltonianOrbit(self.V, self.variables, self.extras, lowlevel=lowlevel, stack=stack)
        if delq0 is None:
            delq0 = [1., *((2*self.nd-1)*[0.])]
        orb.set_ics(0., [*q0, *delq0])
        return orb

def integrate_all(orbits: list[Orbit], Delta_t, dt, err=1e-8, method='RK4', max_frames=-1, args=(), threads=-1)->list[OdeResult]:

    cls = [orb.ode.__class__ for orb in orbits]
    if not hasattr(cls[0], 'dsolve_all'):
        raise ValueError("All orbits passed in the parallel integrator must have a 'LowLevelODE' ode ")
    ode_data = []
    for orb in orbits:
        ics = orb.current_ics()
        ode_data.append((orb.ode, dict(ics = ics, t=ics[0]+Delta_t, dt=dt, err=err, method=method, max_frames=max_frames, args=args)))

    cls: LowLevelODE = cls[0]

    res = cls.dsolve_all(ode_data, threads)

    for i in range(len(orbits)):
        orbits[i]._absorb_oderes(res[i])
    
    return res

