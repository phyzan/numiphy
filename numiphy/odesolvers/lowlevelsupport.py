from ..symlib.expressions import *
from typing import Iterable
from ..symlib.conditional import Boolean
from ..toolkit import tools
import os
import tempfile
from .odepack import *
from functools import cached_property


_stack_vec_alias = "StackVec"
_vector = "std::vector"

def _vec(stack: bool):
    return _stack_vec_alias if stack else "vec"

class ContainerLowLevel:

    def __init__(self, array_type: str, *args: Variable):
        self.array_type = array_type
        self.args = args
    
    def as_argument(self, scalar_type: str, name: str):
        return f'const {self.array_type}<{scalar_type}>& {name}'
    
    def map(self, array_name: str)->dict[Variable, Variable]:
        return {self.args[i]: Variable(f'{array_name}[{i}]') for i in range(len(self.args))}
    
    def converted(self, expr: Expr, array_name: str):
        return expr.replace(self.map(array_name))


class FunctionLowLevel:

    def __init__(self, *args: Variable, **containers: ContainerLowLevel):
        self.args = args
        self.containers = containers

    def argument_list(self, scalar_type):
        arglist = [f'const {scalar_type}& {x}' for x in self.args]
        container_list = [self.containers[name].as_argument(scalar_type, name) for name in self.containers]
        return ', '.join(arglist+container_list)

    def code(self, name: str, scalar_type: str):
        return f'{self.return_type(scalar_type)} {name}({self.argument_list(scalar_type)})'+'{\n\treturn '+f'{self.return_expression(scalar_type)};'+'\n}'

    def lambda_code(self, scalar_type: str):
        return f'[]({self.argument_list(scalar_type)}) -> {self.return_type(scalar_type)} ' + '{return '+f'{self.return_expression(scalar_type)};'+'}'

    def return_expression(self, scalar_type: str)->str:...

    def return_type(self, scalar_type: str)->str:...


class BooleanLowLevel(FunctionLowLevel):

    def __init__(self, expr: Boolean, *args: Variable, **containers: ContainerLowLevel):
        self.expr = expr
        super().__init__(*args, **containers)

    def return_expression(self, scalar_type: str):
        f = self.expr
        for name in self.containers:
            array = self.containers[name]
            _map = array.map(name)
            f = f.do("replace", _map)
        return f.lowlevel_repr(scalar_type)
    
    def return_type(self, scalar_type: str):
        return "bool"



class ScalarLowlevel(FunctionLowLevel):

    def __init__(self, expr: Expr, *args: Variable, **containers: ContainerLowLevel):
        self.expr = expr
        super().__init__(*args, **containers)

    def return_expression(self, scalar_type: str):
        f = self.expr
        for name in self.containers:
            f = self.containers[name].converted(f, name)
        return f.lowlevel_repr(scalar_type)

    def return_type(self, scalar_type: str):
        return scalar_type


class VectorLowLevel(FunctionLowLevel):

    def __init__(self, array_type: str, array: Iterable[Expr], *args: Variable, **containers: ContainerLowLevel):
        self.array_type = array_type
        self.array = array

        super().__init__(*args, **containers)

    def return_expression(self, scalar_type: str):
        converted_array: list[Expr] = []
        for f in self.array:
            g = f
            for name in self.containers:
                arg_array = self.containers[name]
                g = arg_array.converted(g, name)
            converted_array.append(g)
        r = ", ".join([qi.lowlevel_repr(scalar_type) for qi in converted_array])
        return '{'+r+'}'
        
    def return_type(self, scalar_type: str):
        return f'{self.array_type}<{scalar_type}>'






class _SymbolicEvent:

    def __init__(self, name: str, event: Expr, check_if: Boolean=None, mask: Iterable[Expr]=None):
        self.name = name
        self.event = event
        self.check_if = check_if
        self.mask = mask

    def _code(self, scalar_type: str, t: Variable, *q: Variable, args: Iterable[Variable], stack=True):
        args = tuple(args)
        arg_list = dict(q=ContainerLowLevel(_stack_vec_alias, *q), args=ContainerLowLevel(_vector, *args))
        event = ScalarLowlevel(self.event, t, **arg_list)
        checkif = "nullptr"
        if self.check_if is not None:
            checkif = BooleanLowLevel(self.check_if, t, **arg_list).lambda_code(scalar_type)
        return f'Event<{scalar_type}, {_vec(stack)}<{scalar_type}>>("{self.name}", {event.lambda_code(scalar_type)}, {checkif}', arg_list


class SymbolicEvent(_SymbolicEvent):

    def __init__(self, name: str, event: Expr, check_if: Boolean=None, mask: Iterable[Expr]=None):
        self.name = name
        self.event = event
        self.check_if = check_if
        self.mask = mask

    def code(self, scalar_type, t, *q, args, stack=True):
        res , arg_list = super()._code(scalar_type, t, *q, args=args, stack=stack)
        mask = "nullptr"
        if self.mask is not None:
            mask = VectorLowLevel(_vec(stack), self.mask, t, **arg_list).lambda_code(scalar_type)
        return res + f', {mask})'

class SymbolicStopEvent(_SymbolicEvent):

    def __init__(self, name: str, event: Expr, check_if: Boolean=None):
        self.name = name
        self.event = event
        self.check_if = check_if

    def code(self, scalar_type, t, *q, args, stack=True):
        return super()._code(scalar_type, t, *q, args=args, stack=stack)[0] + ')'
        





class SymbolicOde:

    _counter = 0

    def __init__(self, ode_sys: Iterable[Expr], t: Variable, *q: Variable, args: Iterable[Variable] = (), events: Iterable[SymbolicEvent]=(), stop_events: Iterable[SymbolicStopEvent]=()):
        self.ode_sys = tuple(ode_sys)
        self.args = tuple(args)
        self.t = t
        self.q = q
        self.events = tuple(events)
        self.stop_events = stop_events

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

    def ode_system(self, variational=False)->tuple[tuple[Expr, ...], tuple[Variable,...]]:
        if not variational:
            return self.ode_sys, self.q
        else:
            assert not any([x.name.startswith('delta_') for x in self.q+(self.t,)])
            q = self.q
            delq = [Variable('delta_'+qi.name) for qi in q]
            n = len(self.ode_sys)
            var_odesys = []
            for i in range(n):
                var_odesys.append(sum([self.ode_sys[i].diff(q[j])*delq[j] for j in range(n)]))
            
            odesys = self.ode_sys + tuple(var_odesys)
            symbols = self.q + tuple(delq)
            return odesys, symbols
    
    def odefunc_code(self, scalar_type, stack=True, variational=False):
        arr_type = _stack_vec_alias if stack else "vec"
        array, symbols = self.ode_system(variational=variational)
        f = VectorLowLevel(arr_type, array, self.t, q=ContainerLowLevel(arr_type, *symbols), args=ContainerLowLevel(_vector, *self.args))
        return f.code("ODE_FUNC", scalar_type)
        
    def ode_generator_code(self, stack=True, scalar_type="double"):
        Tt = scalar_type
        Ty = _vec(stack) + f"<{scalar_type}>"
        line1 = f"PyODE<{Tt}, {Ty}> GetOde(const {Tt}& t0, py::array q0, const {Tt}& stepsize, const {Tt}& rtol, const {Tt}& atol, const {Tt}& min_step, py::tuple args, py::str method, const {Tt}& event_tol)"+'{\n'
        event_array = '{' + ", ".join([event.code(scalar_type, self.t, *self.q, args=self.args, stack=stack) for event in self.events]) + '}'
        stop_event_array = '{' + ", ".join([event.code(scalar_type, self.t, *self.q, args=self.args, stack=stack) for event in self.stop_events]) + '}'
        line2 = f'\treturn PyODE<{Tt}, {Ty}>(ODE_FUNC, t0, toCPP_Array<{Tt}, {Ty}>(q0), stepsize, rtol, atol, min_step, toCPP_Array<{Tt}, {_vector}<{Tt}>>(args), method.cast<std::string>(), event_tol, {event_array}, {stop_event_array});\n'+'}'
        return line1+line2

    def module_code(self, scalar_type:str, stack=True, variational=False):
        header = "#include <odepack/pyode.hpp>"
        template_alias = f'template<class T>\nusing {_stack_vec_alias} = vec<T, {self.Nsys}>;'
        ode_func = self.odefunc_code(scalar_type, stack=stack, variational=variational)
        ode_gen_code = self.ode_generator_code(stack=stack, scalar_type=scalar_type)
        Ty = f"vec<{scalar_type}, {self.Nsys}>" if stack else f"vec<{scalar_type}>"
        pybind_cond = f"PYBIND11_MODULE({self.module_name}, m)"+"{\n\tdefine_ode_module" + f'<{scalar_type}, {Ty}>(m);\n\tm.def("get_ode", GetOde);\n'+'}'
        return "\n\n".join([header, template_alias, ode_func, ode_gen_code, pybind_cond])

    def generate_cpp_file(self, directory, module_name, stack = True, variational=False, scalar_type="double"):
        if not os.path.exists(directory):
            raise RuntimeError(f'Directory "{directory}" does not exist')
        code = self.module_code(scalar_type=scalar_type, stack=stack, variational=variational)
        cpp_file = os.path.join(directory, f"{module_name}.cpp")

        with open(cpp_file, "w") as f:
            f.write(code)

        return os.path.join(directory, f'{module_name}.cpp')

    def compile(self, directory: str, module_name, stack=True, variational=False, no_math_errno=False):
        if not os.path.exists(directory):
            raise RuntimeError(f"Cannot compile ode at {directory}: Path does not exist")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            cpp_file = self.generate_cpp_file(temp_dir, module_name, stack, variational)
            tools.compile(cpp_file, directory, module_name, no_math_errno=no_math_errno)

    def get(self, t0: float, q0: np.ndarray, stepsize=1e-3, rtol=1e-6, atol=1e-12, min_step=0., args=(), method="RK45", event_tol=1e-12, variational=False, stack=True, no_math_errno=False)->LowLevelODE:
        params = (t0, q0, stepsize, rtol, atol, min_step, args, method, event_tol)
        return self.ode_map(stack=stack, no_errno=no_math_errno, variational=variational)(*params)
        
    def _ode_generator(self, stack=True, variational=False, no_math_errno=False):
        modname = self.module_name

        with tempfile.TemporaryDirectory() as so_dir:
            self.compile(so_dir, modname, stack=stack, variational=variational, no_math_errno=no_math_errno)
            temp_module = tools.import_lowlevel_module(so_dir, modname)

        self.__class__._counter += 1
        return temp_module.get_ode
    
    @cached_property
    def _lowlevel_heap(self):
        return self._ode_generator(stack=False, no_math_errno=False, variational=False)
    
    @cached_property
    def _lowlevel_heap_var(self):
        return self._ode_generator(stack=False, no_math_errno=False, variational=True)
    
    @cached_property
    def _lowlevel_heap_noerrno(self):
        return self._ode_generator(stack=False, no_math_errno=True, variational=False)
    
    @cached_property
    def _lowlevel_heap_var_noerrno(self):
        return self._ode_generator(stack=False, no_math_errno=True, variational=True)
    
    @cached_property
    def _lowlevel_stack(self):
        return self._ode_generator(stack=True, no_math_errno=False, variational=False)

    @cached_property
    def _lowlevel_stack_var(self):
        return self._ode_generator(stack=True, no_math_errno=False, variational=True)
    
    @cached_property
    def _lowlevel_stack_noerrno(self):
        return self._ode_generator(stack=True, no_math_errno=True, variational=False)


    @cached_property
    def _lowlevel_stack_var_noerrno(self):
        return self._ode_generator(stack=True, no_math_errno=True, variational=True)
    
    def ode_map(self, stack, no_errno, variational)->Callable[[float, np.ndarray, float, float, float, float, tuple, str, float], LowLevelODE]:
        _map = [ [ ['_lowlevel_heap', '_lowlevel_heap_var'],
                   ['_lowlevel_heap_noerrno', '_lowlevel_heap_var_noerrno']],

                 [ ['_lowlevel_stack', '_lowlevel_stack_var'],
                   ['_lowlevel_stack_noerrno', '_lowlevel_stack_var_noerrno']]]
        return getattr(self, _map[stack][no_errno][variational])
