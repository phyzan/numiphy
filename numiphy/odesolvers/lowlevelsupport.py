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
from ..symlib.pylambda import _CallableFunction, _BooleanCallable, _ScalarCallable, _VectorCallable, _TensorCallable


_vector = "std::vector"

class ContainerLowLevel(Container):

    def as_argument(self, scalar_type: str, name: str):
        return f'const {self.array_type}<{scalar_type}>& {name}'
        

class OdeGenerator:

    def __init__(self, gen1, gen2):
        self._gen1 = gen1
        self._gen2 = gen2

    def get(self, t0: float, q0: np.ndarray, rtol=1e-6, atol=1e-12, min_step=0., max_step=np.inf, first_step=0., args=(), method="RK45")->LowLevelODE:
        return self._gen1(t0, q0, rtol, atol, min_step, max_step, first_step, args, method)

    def get_variational(self, t0: float, q0: np.ndarray, period: float, rtol=1e-6, atol=1e-12, min_step=0., max_step=np.inf, first_step=0., args=(), method="RK45")->VariationalLowLevelODE:
        return self._gen2(t0, q0, period, rtol, atol, min_step, max_step, first_step, args, method)

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
        final = 'return {' + r + '};'
        return f"#if (_N==-1)\n    {self.return_id(scalar_type)} res({len(res)}); res << {r}; return res;\n#else\n    {final}\n#endif"
    
class VectorLowLevelVoidCallable(VectorLowLevelCallable):

    def return_id(self, scalar_type):
        return 'void'
    
    def core_impl(self, scalar_type):
        res: list[Expr] = self._converted_array(self.array)
        return '\n'.join([f'result[{i}] = {res[i].lowlevel_repr(scalar_type=scalar_type)};' for i in range(len(res))])

    def lambda_code(self, scalar_type):
        return f'[]({VectorLowLevelCallable.return_id(self, scalar_type)+'& result'}, {self.argument_list(scalar_type)})' + '{' +f'{self.core_impl(scalar_type)}'+'}'

    def argument_list(self, scalar_type):
        result_reference = VectorLowLevelCallable.return_id(self, scalar_type)+'& result'
        arglist = [self.scalar_id(scalar_type, x) for x in self.args]
        container_list = [self.containers[name].as_argument(scalar_type, name) for name in self.containers]
        return ', '.join([result_reference]+arglist+container_list)


class JacobianLowLevel(_TensorCallable, _LowLevelCallable):

    def __init__(self, matrix: list[list[Expr]], *args: Symbol, **containers: Container):
        n = len(matrix)
        mat = []
        for i in range(n):
            if len(matrix[i]) != n:
                raise ValueError("matrix shape must be square")
            mat += matrix[i]
        self.matrix = [[m_ij for m_ij in m_i] for m_i in matrix] #just a full copy of the matrix
        _TensorCallable.__init__(self, "jacobian", mat, (n, n), *args, **containers)

    def return_id(self, scalar_type):
        return "void"
    
    def core_impl(self, scalar_type):
        res: list[Expr] = self._converted_array(self.array)
        to_join = []
        n = self.shape[0]
        for i in range(n):
            for j in range(n):
                to_join.append(f'result({i}, {j}) = {res[i*n+j].lowlevel_repr(scalar_type=scalar_type)};')
        return '\n'.join(to_join)
    
    def lambda_code(self, scalar_type):
        return f'[]({f'{self.array_type}<{scalar_type}>& result'}, {self.argument_list(scalar_type)})' + '{' +f'{self.core_impl(scalar_type)}'+'}'

    def argument_list(self, scalar_type):
        result_reference = f'{self.array_type}<{scalar_type}>& result'
        arglist = [self.scalar_id(scalar_type, x) for x in self.args]
        container_list = [self.containers[name].as_argument(scalar_type, name) for name in self.containers]
        return ', '.join([result_reference]+arglist+container_list)


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
    
    def arg_list(self, *q: Symbol, args: Iterable[Symbol]):
        return dict(q=ContainerLowLevel("array", *q), args=ContainerLowLevel(_vector, *args))
            
    def init_code(self, var_name, scalar_type, t, *q, args):
        args = tuple(args)
        arg_list = self.arg_list(*q, args=args)
        lambda_code = ScalarLowLevelCallable(self.event, t, **arg_list).lambda_code(scalar_type)
        checkif = "nullptr"
        if self.check_if is not None:
            checkif = BooleanLowLevelCallable(self.check_if, t, **arg_list).lambda_code(scalar_type)
        mask = "nullptr"
        if self.mask is not None:
            mask = VectorLowLevelCallable("array", self.mask, t, **arg_list).lambda_code(scalar_type)
        return f'{self._cls}<{scalar_type}, _N> {var_name}("{self.name}", {lambda_code}, {checkif}, {mask}, {'true' if self.hide_mask else 'false'}, {self.event_tol});'


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
            

    def init_code(self, var_name, scalar_type, t, *q, args):
        args = tuple(args)
        arg_list = self.arg_list(*q, args=args)
        mask = "nullptr"
        if self.mask is not None:
            mask = VectorLowLevelCallable("array", self.mask, t, **arg_list).lambda_code(scalar_type)
        if self.start is None:
            return f'{self._cls}<{scalar_type}, _N> {var_name}("{self.name}", {self.period}, {mask}, {'true' if self.hide_mask else 'false'});'
        else:
            return f'{self._cls}<{scalar_type}, _N> {var_name}("{self.name}", {self.period}, {self.start}, {mask}, {'true' if self.hide_mask else 'false'});'



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
    
    def odefunc_code(self, scalar_type):
        array, symbols = self.ode_sys, self.q
        f = VectorLowLevelVoidCallable("array", array, self.t, q=ContainerLowLevel("array", *symbols), args=ContainerLowLevel(_vector, *self.args))
        return f.code("ODE_FUNC", scalar_type)
    
    def jacobian_code(self, scalar_type):
        array, symbols = self.ode_sys, self.q
        matrix = self.Nsys*[self.Nsys*[None]]
        for i in range(self.Nsys):
            for j in range(self.Nsys):
                matrix[i][j] = array[i].diff(symbols[j])
        f = JacobianLowLevel(matrix, self.t, q=ContainerLowLevel("array", *symbols), args=ContainerLowLevel(_vector, *self.args))
        return f.code("JAC_FUNC", scalar_type)
    
    def event_block(self, scalar_type="double")->str:
        event_block = ''
        event_array = []
        for i in range(len(self.events)):
            ev_name = f'ev{i}'
            event_block += self.events[i].init_code(ev_name, scalar_type, self.t, *self.q, args=self.args)+'\n'
            event_array.append(f'&{ev_name}')
        event_array = f'std::vector<Event<{scalar_type}, _N>*>' +' events = {'+', '.join(event_array)+'};'
        return '\n\n'.join([event_block, event_array])
    
    def ode_generator_code(self, scalar_type="double"):
        Tt = scalar_type
        line1 = f"PyODE<{Tt}, _N> GetOde(const {Tt}& t0, py::iterable q0, const {Tt}& rtol, const {Tt}& atol, const {Tt}& min_step, const {Tt}& max_step, const {Tt}& first_step, py::tuple args, py::str method)"+'{\n\t'
        line2 = f'return PyODE<{Tt}, _N>(ODE_FUNC, JAC_FUNC, t0, toCPP_Array<{Tt}, array<{scalar_type}>>(q0), rtol, atol, min_step, max_step, first_step, toCPP_Array<{Tt}, {_vector}<{Tt}>>(args), events, method.cast<std::string>());\n'+'}\n\n'
        line3 = f"PyVarODE<{Tt}, _N> GetVarOde(const {Tt}& t0, py::object q0, const {Tt}& period, const {Tt}& rtol, const {Tt}& atol, const {Tt}& min_step, const {Tt}& max_step, const {Tt}& first_step, py::tuple args, py::str method)"+'{\n\t'
        line4 = f'return PyVarODE<{Tt}, _N>(ODE_FUNC, JAC_FUNC, t0, toCPP_Array<{Tt}, array<{scalar_type}>>(q0), period, rtol, atol, min_step, max_step, first_step, toCPP_Array<{Tt}, {_vector}<{Tt}>>(args), events, method.cast<std::string>());\n'+'}\n\n'
        return line1+line2+line3+line4

    def module_code(self, name = "ode_module", scalar_type = "double", stack=True):
        header = "#include <odepack/pyode.hpp>"
        definitions = f'# define _N {self.Nsys if stack else -1}\n\ntemplate<class T>\nusing array = vec<T, _N>;\n\ntemplate<class T>\nusing jacobian = JacMat<T, _N>;\n'
        event_block = self.event_block(scalar_type)
        ode_func = self.odefunc_code(scalar_type)
        jac_func = self.jacobian_code(scalar_type)
        ode_gen_code = self.ode_generator_code(scalar_type=scalar_type)
        r = '\n\tm.def("func_ptr", [](){return reinterpret_cast<const void*>(ODE_FUNC);});'
        r += '\n\tm.def("jac_ptr", [](){return reinterpret_cast<const void*>(JAC_FUNC);});'
        r += '\n\tm.def("ev_ptr", [](){return reinterpret_cast<const void*>(&events);});'
        r += '\n\tm.def("mask_ptr", [](){void* ptr = nullptr; return ptr;});'
        y1 = '\n\tm.def("get_ode", GetOde);\n'
        y2 = '\n\tm.def("get_var_ode", GetVarOde);\n'
        p = "\n\tdefine_ode_module" + f'<{scalar_type}, _N>(m);'+y1+y2 if stack else ''
        commands = p + r
        pybind_cond = f"PYBIND11_MODULE({name}, m)"+'{'+commands+'\n}'
        items = [header, definitions, event_block, ode_func, jac_func, ode_gen_code, pybind_cond]
        if not stack:
            items.pop(-2)
        return "\n\n".join(items)

    def generate_cpp_file(self, directory, module_name, stack = True, scalar_type="double"):
        if not os.path.exists(directory):
            raise RuntimeError(f'Directory "{directory}" does not exist')
        code = self.module_code(name=module_name, scalar_type=scalar_type, stack=stack)
        cpp_file = os.path.join(directory, f"{module_name}.cpp")

        with open(cpp_file, "w") as f:
            f.write(code)
        
        return os.path.join(directory, f'{module_name}.cpp')

    def compile(self, directory: str, module_name, stack=True, no_math_errno=True, no_math_trap=False, fast_math=False, scalar_type="double"):
        if not os.path.exists(directory):
            raise RuntimeError(f"Cannot compile ode at {directory}: Path does not exist")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            cpp_file = self.generate_cpp_file(temp_dir, module_name, stack, scalar_type=scalar_type)
            tools.compile(cpp_file, directory, module_name, no_math_errno=no_math_errno, no_math_trap=no_math_trap, fast_math=fast_math)

    def compile_and_import(self, stack=True, no_math_errno=True, no_math_trap=False, fast_math=False, scalar_type="double"):
        '''
        The returned object's class will behave the same as LowLevelODE, but will appear as different class because it will
        originate from a newly compiled python module.
        '''
        name = tools.random_module_name()
        with tempfile.TemporaryDirectory() as compile_temp_dir:
            self.compile(directory=compile_temp_dir, module_name=name, stack=stack, no_math_errno=no_math_errno, no_math_trap=no_math_trap, fast_math=fast_math, scalar_type=scalar_type)
            mod = tools.import_lowlevel_module(compile_temp_dir, name)
        return OdeGenerator(getattr(mod, "get_ode"), getattr(mod, "get_var_ode"))
    
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
            self.compile(so_dir, modname, stack=False, no_math_errno=True, no_math_trap=False, fast_math=False, scalar_type="double")
            temp_module = tools.import_lowlevel_module(so_dir, modname)
        
        self.__class__._counter += 1
        self._ptrs = temp_module.func_ptr(), temp_module.jac_ptr(), temp_module.ev_ptr()
        return self._ptrs
    
    @property
    def lowlevel_odefunc(self):
        return LowLevelFunction(self.pointers()[0], len(self.ode_sys), len(self.args))
    
    @property
    def lowlevel_jac(self):
        return LowLevelJacobian(self.pointers()[1], len(self.ode_sys), len(self.args))
    
    @property
    def lowlevel_events(self):
        return LowLevelEventArray(self.pointers()[2], len(self.ode_sys), len(self.args))
    
    @cached_property
    def odefunc(self)->Callable[[float, np.ndarray, tuple[float]], np.ndarray]:
        f = VectorPythonCallable("numpy.ndarray", self.ode_sys, self.t, q=PyContainer("numpy.ndarray", *self.q), args=PyContainer("tuple", self.args))

        glob_vars = {"numpy": np, "math": math, "cmath": cmath}
        exec(f.code("ode_rhs", "float", "math"), glob_vars)
        return glob_vars['ode_rhs']



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