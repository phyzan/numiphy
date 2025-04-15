from __future__ import annotations
from ..symlib.symcore import *
from ..symlib.pylambda import *
from typing import Iterable
from ..symlib.boolean import Boolean
from ..toolkit import tools
import os
import tempfile
from typing import Callable
from .odepack import * # type: ignore
from functools import cached_property
from ..symlib.pylambda import _CallableFunction, _BooleanCallable, _ScalarCallable, _VectorCallable


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
        res = self._mapped_boolean().lowlevel_repr(scalar_type=scalar_type)
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
        if self.array_type == 'vec':
            return f'{self.return_id(scalar_type)} res({len(res)}); res << {r}; return res;'
        else:
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

    def __eq__(self, other: AnySymbolicEvent):
        return (self.name, self.mask, self.hide_mask) == (other.name, other.mask, other.hide_mask)

    def arg_list(self, *q: Symbol, args: Iterable[Symbol], stack: bool):
        return dict(q=ContainerLowLevel(_vec(stack), *q), args=ContainerLowLevel(_vector, *args))

    def init_code(self, var_name, scalar_type: str, t: Symbol, *q: Symbol, args: Iterable[Symbol], stack=True)->str:...


class SymbolicEvent(AnySymbolicEvent):

    _cls = 'Event'

    def __init__(self, name: str, event: Expr, check_if: Boolean=None, mask: Iterable[Expr]=None, hide_mask=False, event_tol=1e-12):
        AnySymbolicEvent.__init__(self, name, mask, hide_mask)
        if not isinstance(event, Expr):
            raise ValueError("Expr argument must be a valid symbolic expression")
        self.event = event
        self.check_if = check_if
        self.event_tol = event_tol

    def __eq__(self, other):
        if isinstance(other, SymbolicEvent):
            if other is self:
                return True
            elif (self.event, self.check_if, self.event_tol) == (other.event, other.check_if, other.event_tol):
                return AnySymbolicEvent.__eq__(self, other)
        return False
            
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
        return f'{self._cls}<{scalar_type}, {_vec(stack)}<{scalar_type}>> {var_name}("{self.name}", {lambda_code}, {checkif}, {mask}, {'true' if self.hide_mask else 'false'}, {self.event_tol});'


class SymbolicPeriodicEvent(AnySymbolicEvent):

    _cls = 'PeriodicEvent'

    def __init__(self, name: str, period: float, start = 0., mask: Iterable[Expr]=None, hide_mask=False):
        AnySymbolicEvent.__init__(self, name, mask, hide_mask)
        self.period = period
        self.start = start

    def __eq__(self, other):
        if isinstance(other, SymbolicPeriodicEvent):
            if other is self:
                return True
            elif (self.period, self.start) == (other.period, other.mask):
                return AnySymbolicEvent.__eq__(self, other)
        return False
            

    def init_code(self, var_name, scalar_type, t, *q, args, stack=True):
        args = tuple(args)
        arg_list = self.arg_list(*q, args=args, stack=stack)
        mask = "nullptr"
        if self.mask is not None:
            mask = VectorLowLevelCallable(_vec(stack), self.mask, t, **arg_list).lambda_code(scalar_type)
        return f'{self._cls}<{scalar_type}, {_vec(stack)}<{scalar_type}>> {var_name}("{self.name}", {self.period}, {self.start}, {mask}, {'true' if self.hide_mask else 'false'});'


class SymbolicStopEvent(AnySymbolicEvent):

    _cls = 'StopEvent'

    def __init__(self, name: str, event: Expr, check_if: Boolean=None, mask: Iterable[Expr]=None, hide_mask=False):
        AnySymbolicEvent.__init__(self, name, mask, hide_mask)
        if not isinstance(event, Expr):
            raise ValueError("Expr argument must be a valid symbolic expression")
        self.event = event
        self.check_if = check_if

    def __eq__(self, other):
        if isinstance(other, SymbolicStopEvent):
            if other is self:
                return True
            elif (self.event, self.check_if) == (other.event, other.check_if):
                return AnySymbolicEvent.__eq__(self, other)
        return False

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



class OdeSystem:

    _counter = 0
    _int_all_func: dict[int] = dict()
    _compiled_odes: dict[tuple, LowLevelODE] = dict()

    def __init__(self, ode_sys: Iterable[Expr], t: Symbol, *q: Symbol, args: Iterable[Symbol] = (), events: Iterable[AnySymbolicEvent]=()):
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
    
    def odefunc_code(self, scalar_type, stack=True):
        arr_type = _stack_vec_alias if stack else "vec"
        array, symbols = self.ode_sys, self.q
        f = VectorLowLevelCallable(arr_type, array, self.t, q=ContainerLowLevel(arr_type, *symbols), args=ContainerLowLevel(_vector, *self.args))
        return f.code("ODE_FUNC", scalar_type)
    
    def event_block(self, stack=True, scalar_type="double")->str:
        event_block = ''
        event_array = []
        for i in range(len(self.events)):
            ev_name = f'ev{i}'
            event_block += self.events[i].init_code(ev_name, scalar_type, self.t, *self.q, args=self.args, stack=stack)+'\n'
            event_array.append(f'&{ev_name}')
        event_array = f'std::vector<AnyEvent<{scalar_type}, {_vec(stack)}<{scalar_type}>>*>' +' events = {'+', '.join(event_array)+'};'
        return '\n\n'.join([event_block, event_array])
        
    def ode_generator_code(self, stack=True, scalar_type="double"):
        Tt = scalar_type
        Ty = _vec(stack) + f"<{scalar_type}>"
        line1 = f"PyODE<{Tt}, {Ty}> GetOde(const {Tt}& t0, py::array q0, const {Tt}& rtol, const {Tt}& atol, const {Tt}& min_step, const {Tt}& max_step, const {Tt}& first_step, py::tuple args, py::str method, py::str savedir, py::bool_ save_events_only)"+'{\n\t'
        line2 = f'return PyODE<{Tt}, {Ty}>(ODE_FUNC, t0, toCPP_Array<{Tt}, {Ty}>(q0), rtol, atol, min_step, max_step, first_step, toCPP_Array<{Tt}, {_vector}<{Tt}>>(args), method.cast<std::string>(), events, nullptr, savedir.cast<std::string>(), save_events_only);\n'+'}'
        return line1+line2

    def module_code(self, scalar_type = "double", stack=True):
        def_ode_mod = stack
        header = "#include <odepack/pyode.hpp>"
        template_alias = f'template<class T>\nusing {_stack_vec_alias} = vec<T, {self.Nsys}>;'
        event_block = self.event_block(stack, scalar_type)
        ode_func = self.odefunc_code(scalar_type, stack=stack)
        ode_gen_code = self.ode_generator_code(stack=stack, scalar_type=scalar_type)
        r = '\n\tm.def("func_ptr", [](){return reinterpret_cast<const void*>(ODE_FUNC);});'
        r += '\n\tm.def("ev_ptr", [](){return reinterpret_cast<const void*>(&events);});'
        r += '\n\tm.def("mask_ptr", [](){void* ptr = nullptr; return ptr;});'
        y = '\n\tm.def("get_ode", GetOde);\n'
        Ty = f"vec<{scalar_type}, {self.Nsys}>" if stack else f"vec<{scalar_type}>"
        p = "\n\tdefine_ode_module" + f'<{scalar_type}, {Ty}>(m);'+y if def_ode_mod else ''
        commands = p + r
        pybind_cond = f"PYBIND11_MODULE({self.module_name}, m)"+'{'+commands+'\n}'
        items = [header, template_alias, event_block, ode_func, ode_gen_code, pybind_cond]
        if not stack:
            items.pop(-2)
        return "\n\n".join(items)

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

    def get(self, t0: float, q0: np.ndarray, rtol=1e-6, atol=1e-12, min_step=0., max_step=np.inf, first_step=0., args=(), method="RK45", no_math_errno=False, fast_math=False, no_math_trap=False, scalar_type="double", savedir="", save_events_only=False)->LowLevelODE:
        if len(args) != len(self.args):
            raise ValueError(".get(...) requires args=() with a size equal to the size of the args iterable of symbols provided in the initialization of the OdeSystem")
        params = (t0, q0, rtol, atol, min_step, max_step, first_step, args, method, savedir, save_events_only)
        if (no_math_errno, no_math_trap, fast_math, scalar_type) not in self._compiled_odes:
            self._ode_generator(no_math_errno=no_math_errno, no_math_trap=no_math_trap, fast_math=fast_math)
        return self._compiled_odes[(no_math_errno, no_math_trap, fast_math, scalar_type)](*params)
    
    def integrate_all(self, odes: Iterable[LowLevelODE], interval, *, max_frames=-1, max_events=-1, terminate=True, threads=-1, max_prints=0)->None:
        grouped: dict[int, list[tuple[LowLevelODE, int]]] = {}
        for i, ode in enumerate(odes):
            if ode.dim not in grouped:
                grouped[ode.dim] = [(ode, i)]
            else:
                grouped[ode.dim].append((ode, i))
        for dim in grouped:
            ode_arr = [l[0] for l in grouped[dim]]
            self._int_all_func[dim](ode_arr, interval, max_frames=max_frames, max_events=max_events, terminate=terminate, threads=threads, max_prints=max_prints)
    
    def _ode_generator(self, no_math_errno=False, no_math_trap=False, fast_math=False, scalar_type="double")->Callable[[float, np.ndarray, float, float, float, float, tuple, str, float], LowLevelODE]:
        modname = self.module_name

        with tempfile.TemporaryDirectory() as so_dir:
            self.compile(so_dir, modname, stack=True, no_math_errno=no_math_errno, no_math_trap=no_math_trap, fast_math=fast_math, scalar_type=scalar_type)
            temp_module = tools.import_lowlevel_module(so_dir, modname)

        self.__class__._counter += 1
        if self.Nsys not in self._int_all_func:
            self._int_all_func[self.Nsys] = temp_module.integrate_all
        self._compiled_odes[(no_math_errno, no_math_trap, fast_math, scalar_type)] = temp_module.get_ode
        return temp_module.get_ode
    
    def pointers(self, scalar_type="double"):
        modname = self.module_name

        with tempfile.TemporaryDirectory() as so_dir:
            self.compile(so_dir, modname, stack=False, no_math_errno=True, no_math_trap=False, fast_math=False, scalar_type=scalar_type)
            temp_module = tools.import_lowlevel_module(so_dir, modname)

        self.__class__._counter += 1
        return temp_module.func_ptr, temp_module.ev_ptr, temp_module.mask_ptr


def VariationalOdeSystem(ode_sys: Iterable[Expr], t: Symbol, q: Iterable[Symbol], delq: Iterable[Symbol], args: Iterable[Symbol] = (), events: Iterable[AnySymbolicEvent]=()):
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