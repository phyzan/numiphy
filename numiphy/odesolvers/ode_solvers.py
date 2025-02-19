from __future__ import annotations

from ..symlib import expressions as sym
import numpy as np
import time
from typing import Callable, Literal
from ..toolkit import tools
from ..toolkit import Template, suffix
import importlib.util
import subprocess, os
from functools import cached_property
import tempfile


def bisect(f, a, b, tol = 1e-12):
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

    def custom_solver(self, ics, t, dt, update, lte, args=(), getcond=None, breakcond = None, err = 0., cutoff_step=0., max_frames = -1, display = False, mask = None):
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
                    
                    if isinstance(f_double, np.ndarray):
                        msk = np.abs(f_double) > 0
                        errs = np.abs((f_single - f_double)[msk]/f_double[msk])
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
                        rel_err = np.max(np.abs((f_single - f_double)/f_double))
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
        dt = bisect(h, ti, ti+dt)[2] - ti
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
    so_full_path = so_file + suffix()
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
        return self._to_lowlevel(stack=False, variational=False)

    @cached_property
    def _lowlevel_stack_var(self):
        return self._to_lowlevel(stack=True, variational=True)
    
    @cached_property
    def _lowlevel_heap_var(self):
        return self._to_lowlevel(stack=False, variational=True)


class OdeResult(Template):

    var: np.ndarray
    func: np.ndarray
    diverges: bool
    is_stiff: bool
    runtime: float

    def __init__(self, var_arr, f_arr, diverges, is_stiff, runtime):
        Template.__init__(self, var=np.asarray(var_arr), func=np.asarray(f_arr), diverges=diverges, is_stiff=is_stiff, runtime=runtime)
