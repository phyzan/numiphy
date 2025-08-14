from __future__ import annotations
from ..symlib.symcore import *
from ..symlib.pylambda import *
from typing import Iterable
from ..symlib.boolean import Boolean
from ..toolkit import tools
import os
import tempfile
from .odepack import * # type: ignore
from ..symlib.pylambda import _CallableFunction, _BooleanCallable, _ScalarCallable, _TensorCallable


class _LowLevelCallable(_CallableFunction):

    def core_impl(self)->str:...

    def scalar_id(self, scalar_name):
        if (self._is_array[scalar_name]):
            return f'const double* {scalar_name}'
        else:
            return f'const double& {scalar_name}'
    
    def _code(self, name, return_type, arg_list, code_impl):
        return f'{return_type} {name}({arg_list})'+'{\n'+code_impl+'\n}'
    
    def code(self, name: str):
        return self._code(name, self.return_id(), self.argument_list(), self.core_impl())
    
    def lambda_code(self):
        return f'[]({self.argument_list()}) -> {self.return_id()} ' + '{' +f'{self.core_impl()}'+'}'


class BooleanLowLevelCallable(_BooleanCallable, _LowLevelCallable):

    def core_impl(self):
        res = self.expr.varsub(self._map).lowlevel_repr("double")
        return f"return {res};"

class ScalarLowLevelCallable(_ScalarCallable, _LowLevelCallable):

    def core_impl(self):
        res = self.expr.varsub(self._map).lowlevel_repr("double")
        return f"return {res};"
    
    def return_id(self):
        return 'double'


class TensorLowLevelCallable(_TensorCallable, _LowLevelCallable):

    @property
    def new_array(self)->list[Expr]:
        return [arg.varsub(self._map) for arg in self.array]

    def return_id(self):
        return 'void'
    
    def core_impl(self):
        res = self.new_array
        return '\n'.join([f'result[{i}] = {res[i].lowlevel_repr("double")};' for i in range(len(res))])

    def lambda_code(self):
        return f'[](double* result, {self.argument_list()})' + '{' +f'{self.core_impl()}'+'}'

    def argument_list(self):
        arglist = 'double* result, '+_CallableFunction.argument_list(self)
        return arglist


class SymbolicEvent:

    _cls = 'PreciseEvent'
    name: str
    mask: Iterable[Expr]
    hide_mask: bool

    def __init__(self, name: str, event: Expr, check_if: Boolean=None, mask: Iterable[Expr]=None, hide_mask=False, event_tol=1e-12):
        self.name = name
        self.event = event
        self.check_if = check_if
        self.mask = mask
        self.hide_mask = hide_mask
        self.event_tol = event_tol

    def __eq__(self, other):
        if type(other) is type(self):
            if other is self:
                return True
            else:
                return (self.name, self.event, self.check_if, self.mask, self.hide_mask, self.event_tol) == (other.name, other.event, other.check_if, other.mask, other.hide_mask, other.event_tol)
        return False
            
    def init_code(self, var_name, t, *q, args):
        lambda_code = ScalarLowLevelCallable(self.event, t, q=q, args=args).lambda_code()
        checkif = "nullptr"
        if self.check_if is not None:
            checkif = BooleanLowLevelCallable(self.check_if, t, q=q, args=args).lambda_code()
        mask = "nullptr"
        if self.mask is not None:
            mask = TensorLowLevelCallable(self.mask, t, q=q, args=args).lambda_code()
        return f'{self._cls}<double, -1> {var_name}("{self.name}", {lambda_code}, {checkif}, {mask}, {'true' if self.hide_mask else 'false'}, {self.event_tol});'


class SymbolicPeriodicEvent(SymbolicEvent):

    _cls = 'PeriodicEvent'

    def __init__(self, name: str, period: float, start = None, mask: Iterable[Expr]=None, hide_mask=False):
        SymbolicEvent.__init__(self, name, None, None, mask, hide_mask, 0)
        self.period = period
        self.start = start

    def __eq__(self, other):
        if isinstance(other, SymbolicPeriodicEvent):
            if other is self:
                return True
            elif (self.period, self.start) == (other.period, other.start):
                return SymbolicEvent.__eq__(self, other)
        return False
            

    def init_code(self, var_name, t, *q, args):
        args = tuple(args)
        mask = "nullptr"
        if self.mask is not None:
            mask = TensorLowLevelCallable(self.mask, t=t, args=args).lambda_code()
        if self.start is None:
            return f'{self._cls}<double, -1> {var_name}("{self.name}", {self.period}, {mask}, {'true' if self.hide_mask else 'false'});'
        else:
            return f'{self._cls}<double, -1> {var_name}("{self.name}", {self.period}, {self.start}, {mask}, {'true' if self.hide_mask else 'false'});'



class OdeSystem:

    _cls_instances: list[OdeSystem] = []

    _counter = 0

    ode_sys: tuple[Expr, ...]
    args: tuple[Symbol, ...]
    t: Symbol
    q: tuple[Symbol, ...]
    events: tuple[SymbolicEvent, ...]
    _ptrs: tuple

    def __new__(cls, ode_sys: Iterable[Expr], t: Symbol, *q: Symbol, args: Iterable[Symbol] = (), events: Iterable[SymbolicEvent]=()):
        obj = object.__new__(cls)
        return cls._process_args(obj, ode_sys, t, *q, args=args, events=events)
    
    @classmethod
    def _process_args(cls, obj: OdeSystem, ode_sys: Iterable[Expr], t: Symbol, *q: Symbol, args: Iterable[Symbol] = (), events: Iterable[SymbolicEvent]=()):
        ode_sys = tuple(ode_sys)
        args = tuple(args)
        events = tuple(events)

        given = (t,)+q+args
        assert tools.all_different(given)
        odesymbols = []
        for ode in ode_sys:
            for arg in ode.variables:
                if arg not in odesymbols:
                    odesymbols.append(arg)
        if len(ode_sys) != len(q):
            raise ValueError('')
        if t in odesymbols:
            assert len(odesymbols) <= len(given)
        else:
            assert len(odesymbols) <= len(given) - 1
        
        obj.ode_sys = ode_sys
        obj.args = args
        obj.t = t
        obj.q = q
        obj.events = events
        for i in range(len(cls._cls_instances)):
            if cls._cls_instances[i] == obj:
                return cls._cls_instances[i]
        cls._cls_instances.append(obj)
        obj._ptrs = None
        return obj

    def __eq__(self, other: OdeSystem):
        if other is self:
            return True
        else:
            return (self.ode_sys, self.args, self.t, self.q, self.events) == (other.ode_sys, other.args, other.t, other.q, other.events)

    @property
    def Nsys(self):
        return len(self.ode_sys)
    
    @property
    def module_name(self):
        return f'ODE_MODULE_{self.__class__._counter}'
    
    @cached_property
    def jacmat(self):
        array, symbols = self.ode_sys, self.q
        matrix = [[None for i in range(self.Nsys)] for j in range(self.Nsys)]
        for i in range(self.Nsys):
            for j in range(self.Nsys):
                matrix[i][j] = array[i].diff(symbols[j])
        return matrix
    
    def odefunc_code(self):
        f = TensorLowLevelCallable(self.ode_sys, self.t, q=self.q, args=self.args)
        return f.code("ODE_FUNC")
    
    def jacobian_code(self):
        f = TensorLowLevelCallable(self.jacmat, self.t, q=self.q, args=self.args)
        return f.code("JAC_FUNC")
    
    def event_block(self)->str:
        event_block = ''
        event_array = []
        for i in range(len(self.events)):
            ev_name = f'ev{i}'
            event_block += self.events[i].init_code(ev_name, self.t, *self.q, args=self.args)+'\n'
            event_array.append(f'&{ev_name}')
        event_array = f'std::vector<Event<double, -1>*>' +' events = {'+', '.join(event_array)+'};'
        return '\n\n'.join([event_block, event_array])

    def module_code(self, name = "ode_module"):
        header = "#include <odepack/pyode.hpp>"
        event_block = self.event_block()
        ode_func = self.odefunc_code()
        jac_func = self.jacobian_code()
        r = '\n\tm.def("func_ptr", [](){return reinterpret_cast<const void*>(ODE_FUNC);});'
        r += '\n\tm.def("jac_ptr", [](){return reinterpret_cast<const void*>(JAC_FUNC);});'
        r += '\n\tm.def("ev_ptr", [](){return reinterpret_cast<const void*>(&events);});'
        r += '\n\tm.def("mask_ptr", [](){void* ptr = nullptr; return ptr;});'
        pybind_cond = f"PYBIND11_MODULE({name}, m)"+'{'+r+'\n}'
        items = [header, event_block, ode_func, jac_func, pybind_cond]
        return "\n\n".join(items)

    def generate_cpp_file(self, directory, module_name):
        if not os.path.exists(directory):
            raise RuntimeError(f'Directory "{directory}" does not exist')
        code = self.module_code(name=module_name)
        cpp_file = os.path.join(directory, f"{module_name}.cpp")
        with open(cpp_file, "w") as f:
            f.write(code)
        
        return os.path.join(directory, f'{module_name}.cpp')

    def compile(self, directory: str, module_name: str):
        if not os.path.exists(directory):
            raise RuntimeError(f"Cannot compile ode at {directory}: Path does not exist")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            cpp_file = self.generate_cpp_file(temp_dir, module_name)
            tools.compile(cpp_file, directory, module_name)
    
    def get(self, t0: float, q0: np.ndarray, *, rtol=1e-6, atol=1e-12, min_step=0., max_step=np.inf, first_step=0., args=(), method="RK45")->LowLevelODE:
        if len(q0) != self.Nsys:
            raise ValueError(f"The size of the initial conditions provided is {len(q0)} instead of {self.Nsys}")
        return LowLevelODE(self.lowlevel_odefunc, self.lowlevel_jac, t0=t0, q0=q0, rtol=rtol, atol=atol, min_step=min_step, max_step=max_step, first_step=first_step, args=args, method=method, events=self.lowlevel_events)
    
    def get_variational(self, t0: float, q0: np.ndarray, period: float, *, rtol=1e-6, atol=1e-12, min_step=0., max_step=np.inf, first_step=0., args=(), method="RK45")->VariationalLowLevelODE:
        if len(q0) != self.Nsys:
            raise ValueError(f"The size of the initial conditions provided is {len(q0)} instead of {self.Nsys}")
        return VariationalLowLevelODE(self.lowlevel_odefunc, self.lowlevel_jac, t0=t0, q0=q0, period=period, rtol=rtol, atol=atol, min_step=min_step, max_step=max_step, first_step=first_step, args=args, method=method, events=self.lowlevel_events)
    
    def pointers(self):
        if self._ptrs is not None:
            return self._ptrs
        
        modname = self.module_name

        with tempfile.TemporaryDirectory() as so_dir:
            self.compile(so_dir, modname)
            temp_module = tools.import_lowlevel_module(so_dir, modname)
        
        self.__class__._counter += 1
        self._ptrs = temp_module.func_ptr(), temp_module.jac_ptr(), temp_module.ev_ptr()
        return self._ptrs
    
    @cached_property
    def lowlevel_odefunc(self):
        return LowLevelFunction(self.pointers()[0], len(self.ode_sys), len(self.args))
    
    @cached_property
    def lowlevel_jac(self):
        return LowLevelJacobian(self.pointers()[1], len(self.ode_sys), len(self.args))
    
    @cached_property
    def lowlevel_events(self):
        return LowLevelEventArray(self.pointers()[2], len(self.ode_sys), len(self.args))
    
    @cached_property
    def _odefunc(self):
        kwargs = {str(x): x for x in self.args}
        return lambdify(self.q, "numpy", self.t, q=self.q, **kwargs)
    
    @cached_property
    def _jac(self):
        kwargs = {str(x): x for x in self.args}
        return lambdify(self.jacmat, "numpy", self.t, q=self.q, **kwargs)
    
    def odefunc(self, t: float, q: np.ndarray, *args: float)->np.ndarray:
        return self._odefunc(t, q, *args)
    
    def jac(self, t: float, q: np.ndarray, *args: float)->np.ndarray:
        return self._jac(t, q, *args)




def VariationalOdeSystem(ode_sys: Iterable[Expr], t: Symbol, q: Iterable[Symbol], delq: Iterable[Symbol], args: Iterable[Symbol] = (), events: Iterable[SymbolicEvent]=()):
    n = len(ode_sys)
    ode_sys = tuple(ode_sys)
    var_odesys = []
    for i in range(n):
        var_odesys.append(sum([ode_sys[i].diff(q[j])*delq[j] for j in range(n)]))
    
    new_sys = ode_sys + tuple(var_odesys)
    return OdeSystem(new_sys, t, *q, *delq, args=args, events=events)


def load_ode_data(filedir: str)->tuple[np.ndarray[int], np.ndarray, np.ndarray]:
    data = np.loadtxt(filedir)
    events = data[:, 0].astype(int)
    t = data[:, 1].copy()
    q = data[:, 2:].copy()
    return events, t, q