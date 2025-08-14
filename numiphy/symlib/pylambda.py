from .symcore import *
import numpy as np
import math, cmath
from .boolean import Boolean
from typing import Iterable, Callable


class _CallableFunction:

    def __init__(self, result: Expr|Iterable[Expr], *args: Symbol, **kwargs: Symbol|Iterable[Symbol]):
        '''
        must check correct input types, and that all symbols in input are different
        '''
        self._res: Expr|list[Expr]
        if hasattr(result, '__iter__'):
            self._res = list(result)
        else:
            self._res = result
        self._map = {arg: arg for arg in args}
        self._is_array = {arg.name: False for arg in args}
        self._arg_symbols = list(args)
        for name, obj in kwargs.items():
            x = Symbol(name)
            self._arg_symbols.append(x)
            if isinstance(obj, Symbol):
                self._map.update({obj: x})
                self._is_array.update({x.name: False})
            else:
                for i, symbol in enumerate(obj):
                    self._map.update({symbol: Symbol(f'{name}[{i}]')})
                self._is_array.update({x.name: True})
        
        self._arg_symbols = tuple(self._arg_symbols)

    def argument_list(self):
        arglist = [self.scalar_id(x.name) for x in self._arg_symbols]
        return ', '.join(arglist)
    
    def scalar_id(self, scalar_name: str)->str:...

    def _code(self, name: str, return_type: str, arg_list: str, code_impl: str)->str:...

    def return_id(self)->str:...


class _BooleanCallable(_CallableFunction):

    def __init__(self, expr: Boolean, *args: Symbol, **kwargs: Symbol|Iterable[Symbol]):
        _CallableFunction.__init__(self, expr, *args, **kwargs)

    @property
    def expr(self)->Boolean:
        return self._res
    
    def return_id(self):
        return "bool"


class _ScalarCallable(_CallableFunction):

    def __init__(self, expr: Expr, *args: Symbol, **kwargs: Symbol|Iterable[Symbol]):
        self.expr = expr
        _CallableFunction.__init__(self, expr, *args, **kwargs)


class _TensorCallable(_CallableFunction):

    def __init__(self, array: Iterable, *args: Symbol, **kwargs: Symbol|Iterable[Symbol]):
        arr = np.array(array, dtype=object)
        self.shape = arr.shape
        _CallableFunction.__init__(self, arr.flatten().tolist(), *args, **kwargs)
    
    @property
    def array(self)->list[Expr]:
        return self._res
    
    @property
    def new_array(self)->list:
        return np.array([arg.varsub(self._map) for arg in self.array], dtype=object).reshape(self.shape).tolist()


class _PythonCallable(_CallableFunction):

    def scalar_id(self, scalar_name):
        return f'{scalar_name}'
    
    def return_id(self):
        return 'float'
    
    def _code(self, name, return_type, arg_list, code_impl):
        return f"def {name}({arg_list})->{return_type}:\n\t{code_impl}"
    
    def code(self, name: str, lib: str):
        return self._code(name, self.return_id(), self.argument_list(), self.core_impl(lib))
    
    def lambda_code(self, lib: str):
        return f'lambda {self.argument_list()}: {self.core_impl(lib)}'
    
    def core_impl(self, lib: str)->str:...
    

class BooleanPythonCallable(_BooleanCallable, _PythonCallable):

    def core_impl(self, lib: str):
        res = self.expr.varsub(self._map).repr(lib=lib)
        return f"return {res}"


class ScalarPythonCallable(_ScalarCallable, _PythonCallable):

    def core_impl(self, lib: str):
        res = self.expr.varsub(self._map).repr(lib=lib)
        return f"return {res}"


class TensorPythonCallable(_TensorCallable, _PythonCallable):

    def return_id(self):
        return f"numpy.ndarray[float]"

    def core_impl(self, lib: str):
        return f"return numpy.array({_multidim_lambda_list(self.new_array, lib=lib)})"




def _multidim_lambda_list(arg, lib: str):
    if hasattr(arg, '__iter__'):
        return f"[{', '.join([_multidim_lambda_list(f, lib) for f in arg])}]"
    elif isinstance(arg, Expr):
        return arg.repr(lib = lib)
    else:
        raise ValueError(f"Item of type '{arg.__class__}' not supported for lambdifying")

def lambdify(arg, lib: str, *args: Symbol, **kwargs: Iterable[Symbol])->Callable:

    if hasattr(arg, '__iter__'):
        r = TensorPythonCallable(arg, *args, **kwargs)
    elif isinstance(arg, Boolean):
        r = BooleanPythonCallable(arg, *args, **kwargs)
    else:
        r = ScalarPythonCallable(arg, *args, **kwargs)

    code = r.code("MyFunc", lib=lib)
    glob_vars = {"numpy": np, "math": math, "cmath": cmath}
    exec(code, glob_vars)
    return glob_vars['MyFunc']


class ScalarLambdaExpr:

    def __init__(self, expr: Expr, *symbols: Symbol):
        self.expr = expr
        self.symbols = symbols
        self._callable = self.expr.lambdify(*symbols, lib='numpy')

    def __call__(self, *args)->float|complex:
        return self._callable(*args)
    
    def __str__(self):
        return str(self.expr)
    
    def __repr__(self):
        return str(self)


class VectorLambdaExpr:

    def __init__(self, expr: list, *symbols: Symbol):
        self.expr = expr
        self.symbols = symbols
        self.code = f"np.array({_multidim_lambda_list(expr, lib="math")})"
        self._callable = lambdify(expr, "numpy", *symbols)

    def __call__(self, *args)->np.ndarray:
        return self._callable(*args)
    
    def __str__(self):
        return self.code
    
    def __repr__(self):
        return self.code