from .symexpr import *
import numpy as np
import math, cmath
from ..conditional import Boolean
from typing import Iterable


class Container:

    def __init__(self, array_type: str, *args: Variable):
        self.array_type = array_type
        self.args = args
    
    def as_argument(self, scalar_type: str, name: str)->str:...
    
    def map(self, array_name: str)->dict[Variable, Variable]:
        return {self.args[i]: Variable(f'{array_name}[{i}]') for i in range(len(self.args))}
    
    def converted(self, expr: Expr, array_name: str):
        return expr.replace(self.map(array_name))
    

class PyContainer(Container):

    def as_argument(self, scalar_type: str, name: str):
        return f'{name}: {self.array_type}[{scalar_type}]'
    

class ContainerLowLevel(Container):

    def as_argument(self, scalar_type: str, name: str):
        return f'const {self.array_type}<{scalar_type}>& {name}'
        


class _CallableFunction:

    def __init__(self, *args: Variable, **containers: Container):
        self.args = args
        self.containers = containers

    def argument_list(self, scalar_type):
        arglist = [self.scalar_id(scalar_type, x) for x in self.args]
        container_list = [self.containers[name].as_argument(scalar_type, name) for name in self.containers]
        return ', '.join(arglist+container_list)
    
    def scalar_id(self, scalar_type: str, scalar_name: str)->str:...

    def _code(self, name: str, return_type: str, arg_list: str, code_impl: str)->str:...

    def code(self, name: str, scalar_type: str, *args, **kwargs)->str:...

    def lambda_code(self, scalar_type: str, *args, **kwargs)->str:...

    def return_id(self, scalar_type: str)->str:...

    def core_impl(self, *args, **kwargs)->str:...


class _BooleanCallable(_CallableFunction):

    expr: Boolean

    def __init__(self, expr: Boolean, *args: Variable, **containers: PyContainer):
        self.expr = expr
        _CallableFunction.__init__(self, *args, **containers)

    def _mapped_boolen(self)->Boolean:
        f = self.expr
        for name in self.containers:
            array = self.containers[name]
            _map = array.map(name)
            f = f.do("replace", _map)
        return f
    
    def return_id(self, scalar_type: str):
        return "bool"


class _ScalarCallable(_CallableFunction):

    def __init__(self, expr: Expr, *args: Variable, **containers: ContainerLowLevel):
        self.expr = expr
        _CallableFunction.__init__(self, *args, **containers)

    def _mapped_expr(self)->Expr:
        f = self.expr
        for name in self.containers:
            f = self.containers[name].converted(f, name)
        return f

    def return_id(self, scalar_type: str):
        return scalar_type


class _VectorCallable(_CallableFunction):

    def __init__(self, array_type: str, array: Iterable[Expr], *args: Variable, **containers: ContainerLowLevel):
        self.array_type = array_type
        self.array = array
        _CallableFunction.__init__(self, *args, **containers)

    def _convert(self, f: Expr)->Expr:
        g = f
        for name in self.containers:
            arg_array = self.containers[name]
            g = arg_array.converted(g, name)
        return g
    
    def _converted_array(self, array: list)->list:
        new = []
        for f in array:
            if isinstance(f, Expr):
                new.append(self._convert(f))
            else:
                new.append(self._converted_array(f))
        return new


class _PythonCallable(_CallableFunction):

    def scalar_id(self, scalar_type, scalar_name):
        return f'{scalar_name}: {scalar_type}'
    
    def _code(self, name, return_type, arg_list, code_impl):
        return f"def {name}({arg_list})->{return_type}:\n\t{code_impl}"
    
    def code(self, name: str, scalar_type: str, lib: str):
        return self._code(name, self.return_id(scalar_type), self.argument_list(scalar_type), self.core_impl(lib=lib))
    
    def lambda_code(self, scalar_type, lib: str):
        return f'lambda {self.argument_list(scalar_type)}: {self.core_impl(lib=lib)}'
    
    def core_impl(self, lib: str)->str:...
    


class BooleanPythonCallable(_BooleanCallable, _PythonCallable):

    def core_impl(self, lib: str):
        res = self._mapped_boolen().repr(lib=lib)
        return f"return {res}"


class ScalarPythonCallable(_ScalarCallable, _PythonCallable):

    def core_impl(self, lib: str):
        res = self._mapped_expr().repr(lib=lib)
        return f"return {res}"


class VectorPythonCallable(_VectorCallable, _PythonCallable):

    def return_id(self, scalar_type):
        return f"{self.array_type}[{scalar_type}]"

    def core_impl(self, lib):
        return f"return numpy.array({_multidim_lambda_list(self._converted_array(self.array), lib=lib)})"




def _multidim_lambda_list(arg, lib:str):
    if hasattr(arg, '__iter__'):
        return f"[{', '.join([_multidim_lambda_list(f, lib) for f in arg])}]"
    elif isinstance(arg, Expr):
        return arg.repr(lib = lib)
    else:
        raise ValueError(f"Item of type '{arg.__class__}' not supported for lambdifying")

def lambdify(*expr: Expr, symbols: list[Variable], lib='math'):
    if len(expr) == 1:
        code = ScalarPythonCallable(*expr, *symbols).code("MyFunc", scalar_type="float", lib=lib)
    else:
        code = VectorPythonCallable("numpy.array", expr, *symbols).code("MyFunc", scalar_type="float", lib=lib)
    glob_vars = {"numpy": np, "math": math, "cmath": cmath}
    exec(code, glob_vars)
    return glob_vars['MyFunc']


class ScalarLambdaExpr:

    def __init__(self, expr: Expr, *symbols: Variable):
        self.expr = expr
        self.symbols = symbols
        self._callable = self.expr.lambdify(symbols, lib='numpy')

    def __call__(self, *args)->float|complex:
        return self._callable(*args)
    
    def __str__(self):
        return str(self.expr)
    
    def __repr__(self):
        return str(self)


class VectorLambdaExpr:

    def __init__(self, expr: list, *symbols: Variable):
        self.expr = expr
        self.symbols = symbols
        self.code = f"np.array({_multidim_lambda_list(expr)})"
        self._callable = multidim_lambda_func(expr, *symbols)

    def __call__(self, *args)->np.ndarray:
        return self._callable(*args)
    
    def __str__(self):
        return self.code
    
    def __repr__(self):
        return self.code