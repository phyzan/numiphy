from ..symlib.expressions import *
from typing import Iterable
from ..symlib.conditional import Boolean
from ..toolkit import tools
import os
import tempfile
from .odepack import * # type: ignore
from functools import cached_property
from ..symlib.expressions.pylambda import _CallableFunction, _BooleanCallable, _ScalarCallable, _VectorCallable
import pybind11 #imported only to to raise error if it does not exist. It is needed for the compiler


_stack_vec_alias = "StackVec"
_vector = "std::vector"

def _vec(stack: bool):
    return _stack_vec_alias if stack else "vec"


class _LowLevelCallable(_CallableFunction):

    def core_impl(self, scalar_type: str)->str:...

    def scalar_id(self, scalar_type, scalar_name):
        return f'const {scalar_type}& {scalar_name}'
    
    def _code(self, name, return_type, arg_list, code_impl):
        return f'{return_type} {name}({arg_list})'+'{\n'+code_impl+'\n}'
    
    def code(self, name: str, scalar_type: str):
        return self._code(name, self.return_id(scalar_type), self.argument_list(scalar_type), self.core_impl(scalar_type=scalar_type))
    
    def lambda_code(self, scalar_type):
        return f'[]({self.argument_list(scalar_type)}) -> {self.return_id(scalar_type)} ' + '{' +f'{self.core_impl(scalar_type)}'+'}'
    

class BooleanLowLevelCallable(_BooleanCallable, _LowLevelCallable):

    def core_impl(self, scalar_type):
        res = self._mapped_boolen().lowlevel_repr(scalar_type=scalar_type)
        return f"return {res};"
    

class ScalarLowLevelCallable(_ScalarCallable, _LowLevelCallable):

    def core_impl(self, scalar_type: str):
        res = self._mapped_expr().lowlevel_repr(scalar_type=scalar_type)
        return f"return {res};"


class VectorLowLevelCallable(_VectorCallable, _LowLevelCallable):

    def return_id(self, scalar_type):
        return f'{self.array_type}<{scalar_type}>'
    
    def core_impl(self, scalar_type):
        res: list[Expr] = self._converted_array(self.array)
        r = ", ".join([qi.lowlevel_repr(scalar_type=scalar_type) for qi in res])
        return 'return {'+r+'};'



class AnySymbolicEvent:

    _cls: str
    name: str
    mask: Iterable[Expr]
    hide_mask: bool

    def __init__(self, name: str, mask: Iterable[Expr], hide_mask: bool):
        if type(self) is AnySymbolicEvent:
            raise ValueError("AnySymbolicEvent class cannot be directly instanciated")
        self.name = name
        self.mask = mask
        self.hide_mask = hide_mask

    def arg_list(self, *q: Variable, args: Iterable[Variable], stack: bool):
        return dict(q=ContainerLowLevel(_vec(stack), *q), args=ContainerLowLevel(_vector, *args))

    def init_code(self, var_name, scalar_type: str, t: Variable, *q: Variable, args: Iterable[Variable], stack=True)->str:...


class SymbolicEvent(AnySymbolicEvent):

    _cls = 'Event'

    def __init__(self, name: str, event: Expr, check_if: Boolean=None, mask: Iterable[Expr]=None, hide_mask=False):
        AnySymbolicEvent.__init__(self, name, mask, hide_mask)
        if not isinstance(event, Expr):
            raise ValueError("Expr argument must be a valid symbolic expression")
        self.event = event
        self.check_if = check_if

    def init_code(self, var_name, scalar_type, t, *q, args, stack=True):
        args = tuple(args)
        arg_list = self.arg_list(*q, args=args, stack=stack)
        lambda_code = ScalarLowLevelCallable(self.event, t, **arg_list).lambda_code(scalar_type)
        checkif = "nullptr"
        if self.check_if is not None:
            checkif = BooleanLowLevelCallable(self.check_if, t, **arg_list).lambda_code(scalar_type)
        mask = "nullptr"
        if self.mask is not None:
            mask = VectorLowLevelCallable(_vec(stack), self.mask, t, **arg_list).lambda_code(scalar_type)
        return f'{self._cls}<{scalar_type}, {_vec(stack)}<{scalar_type}>> {var_name}("{self.name}", {lambda_code}, {checkif}, {mask}, {'true' if self.hide_mask else 'false'});'


class SymbolicPeriodicEvent(AnySymbolicEvent):

    _cls = 'PeriodicEvent'

    def __init__(self, name: str, period: float, start = 0., mask: Iterable[Expr]=None, hide_mask=False):
        AnySymbolicEvent.__init__(self, name, mask, hide_mask)
        self.period = period
        self.start = start

    def init_code(self, var_name, scalar_type, t, *q, args, stack=True):
        args = tuple(args)
        arg_list = self.arg_list(*q, args=args, stack=stack)
        mask = "nullptr"
        if self.mask is not None:
            mask = VectorLowLevelCallable(_vec(stack), self.mask, t, **arg_list).lambda_code(scalar_type)
        return f'{self._cls}<{scalar_type}, {_vec(stack)}<{scalar_type}>> {var_name}("{self.name}", {self.period}, {self.start}, {mask}, {'true' if self.hide_mask else 'false'});'


class SymbolicStopEvent(SymbolicEvent):

    _cls = 'StopEvent'



class OdeSystem:

    _counter = 0
    _int_all_func = None
    _compiled_odes: dict[tuple, LowLevelODE] = dict()

    def __init__(self, ode_sys: Iterable[Expr], t: Variable, *q: Variable, args: Iterable[Variable] = (), events: Iterable[AnySymbolicEvent]=()):
        self.ode_sys = tuple(ode_sys)
        self.args = tuple(args)
        self.t = t
        self.q = q
        self.events = tuple(events)

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

    @property
    def Nsys(self):
        return len(self.ode_sys)
    
    @property
    def module_name(self):
        return f'ODE_MODULE_{self.__class__._counter}'
    
    def odefunc_code(self, scalar_type, stack=True):
        arr_type = _stack_vec_alias if stack else "vec"
        array, symbols = self.ode_sys, self.q
        f = VectorLowLevelCallable(arr_type, array, self.t, q=ContainerLowLevel(arr_type, *symbols), args=ContainerLowLevel(_vector, *self.args))
        return f.code("ODE_FUNC", scalar_type)
        
    def ode_generator_code(self, stack=True, scalar_type="double"):
        Tt = scalar_type
        Ty = _vec(stack) + f"<{scalar_type}>"
        line1 = f"PyODE<{Tt}, {Ty}> GetOde(const {Tt}& t0, py::array q0, const {Tt}& stepsize, const {Tt}& rtol, const {Tt}& atol, const {Tt}& min_step, py::tuple args, py::str method, const {Tt}& event_tol, py::str savedir, py::bool_ save_events_only)"+'{\n\t'
        event_block = ''
        event_array = []
        for i in range(len(self.events)):
            ev_name = f'ev{i}'
            event_block += self.events[i].init_code(ev_name, scalar_type, self.t, *self.q, args=self.args, stack=stack)+'\n'
            event_array.append(f'&{ev_name}')
        event_array = '{'+', '.join(event_array)+'}'
        line2 = f'\treturn PyODE<{Tt}, {Ty}>(ODE_FUNC, t0, toCPP_Array<{Tt}, {Ty}>(q0), stepsize, rtol, atol, min_step, toCPP_Array<{Tt}, {_vector}<{Tt}>>(args), method.cast<std::string>(), event_tol, {event_array}, savedir.cast<std::string>(), save_events_only);\n'+'}'
        return line1+event_block+line2

    def module_code(self, scalar_type = "double", stack=True):
        header = "#include <odepack/pyode.hpp>"
        template_alias = f'template<class T>\nusing {_stack_vec_alias} = vec<T, {self.Nsys}>;'
        ode_func = self.odefunc_code(scalar_type, stack=stack)
        ode_gen_code = self.ode_generator_code(stack=stack, scalar_type=scalar_type)
        Ty = f"vec<{scalar_type}, {self.Nsys}>" if stack else f"vec<{scalar_type}>"
        pybind_cond = f"PYBIND11_MODULE({self.module_name}, m)"+"{\n\tdefine_ode_module" + f'<{scalar_type}, {Ty}>(m);\n\tm.def("get_ode", GetOde);\n'+'}'
        return "\n\n".join([header, template_alias, ode_func, ode_gen_code, pybind_cond])

    def generate_cpp_file(self, directory, module_name, stack = True, scalar_type="double"):
        if not os.path.exists(directory):
            raise RuntimeError(f'Directory "{directory}" does not exist')
        code = self.module_code(scalar_type=scalar_type, stack=stack)
        cpp_file = os.path.join(directory, f"{module_name}.cpp")

        with open(cpp_file, "w") as f:
            f.write(code)

        return os.path.join(directory, f'{module_name}.cpp')

    def compile(self, directory: str, module_name, stack=True, no_math_errno=False, no_math_trap=False, fast_math=False, scalar_type="double"):
        if not os.path.exists(directory):
            raise RuntimeError(f"Cannot compile ode at {directory}: Path does not exist")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            cpp_file = self.generate_cpp_file(temp_dir, module_name, stack, scalar_type=scalar_type)
            tools.compile(cpp_file, directory, module_name, no_math_errno=no_math_errno, no_math_trap=no_math_trap, fast_math=fast_math)

    def get(self, t0: float, q0: np.ndarray, stepsize=1e-3, rtol=1e-6, atol=1e-12, min_step=0., args=(), method="RK45", event_tol=1e-12, stack=True, no_math_errno=False, fast_math=False, no_math_trap=False, scalar_type="double", savedir="", save_events_only=False)->LowLevelODE:
        if len(args) != len(self.args):
            raise ValueError(".get(...) requires args=() with a size equal to the size of the args iterable of symbols provided in the initialization of the OdeSystem")
        params = (t0, q0, stepsize, rtol, atol, min_step, args, method, event_tol, savedir, save_events_only)
        if (stack, no_math_errno, no_math_trap, fast_math, scalar_type) not in self._compiled_odes:
            self._ode_generator(stack=stack, no_math_errno=no_math_errno, no_math_trap=no_math_trap, fast_math=fast_math)
        return self._compiled_odes[(stack, no_math_errno, no_math_trap, fast_math, scalar_type)](*params)
    
    def integrate_all(self, odes: Iterable[LowLevelODE], interval, *, max_frames=-1, max_events=-1, terminate=True, threads=-1, max_prints=0)->list[LowLevelODE]:
        return self._int_all_func(odes, interval, max_frames=max_frames, max_events=max_events, terminate=terminate, threads=threads, max_prints=max_prints)
    
    def _ode_generator(self, stack=True, no_math_errno=False, no_math_trap=False, fast_math=False, scalar_type="double")->Callable[[float, np.ndarray, float, float, float, float, tuple, str, float], LowLevelODE]:
        modname = self.module_name

        with tempfile.TemporaryDirectory() as so_dir:
            self.compile(so_dir, modname, stack=stack, no_math_errno=no_math_errno, no_math_trap=no_math_trap, fast_math=fast_math, scalar_type=scalar_type)
            temp_module = tools.import_lowlevel_module(so_dir, modname)

        self.__class__._counter += 1
        if stack and self._int_all_func is None:
            self._int_all_func = temp_module.integrate_all
        self._compiled_odes[(stack, no_math_errno, no_math_trap, fast_math, scalar_type)] = temp_module.get_ode
        return temp_module.get_ode


def VariationalOdeSystem(ode_sys: Iterable[Expr], t: Variable, q: Iterable[Variable], delq: Iterable[Variable], args: Iterable[Variable] = (), events: Iterable[SymbolicEvent]=(), stop_events: Iterable[SymbolicStopEvent]=()):
        n = len(ode_sys)
        ode_sys = tuple(ode_sys)
        var_odesys = []
        for i in range(n):
            var_odesys.append(sum([ode_sys[i].diff(q[j])*delq[j] for j in range(n)]))
        
        new_sys = ode_sys + tuple(var_odesys)
        return OdeSystem(new_sys, t, *q, *delq, args=args, events=events, stop_events=stop_events)


def load_ode_data(filedir: str)->tuple[np.ndarray[int], np.ndarray, np.ndarray]:
    data = np.loadtxt(filedir)
    events = data[:, 0].astype(int)
    t = data[:, 1].copy()
    q = data[:, 2:].copy()
    return events, t, q