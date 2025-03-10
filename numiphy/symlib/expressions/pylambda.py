from .symexpr import *
import numpy as np
import math, cmath


def _multidim_lambda_list(arg):
    if hasattr(arg, '__iter__'):
        return f"[{', '.join([_multidim_lambda_list(f) for f in arg])}]"
    elif isinstance(arg, Expr):
        return arg.repr(lang="python", lib = "numpy")
    else:
        raise ValueError(f"Item of type '{arg.__class__}' not supported for lambdifying")
    
def multidim_lambda_func(arg, *symbols: Variable):
    return_expr = _multidim_lambda_list(arg)
    args = ', '.join([str(v) for v in symbols])
    code = f'def MyFunc({args}):\n\t return numpy.array({return_expr})'
    glob_vars = {"numpy": np, "math": math, "cmath": cmath}
    exec(code, glob_vars)
    return glob_vars['MyFunc']


def lambdify(*expr: Expr, symbols: list[Variable], lib='math', ode_style=False):
    raise NotImplementedError


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