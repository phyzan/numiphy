from .symexpr import *
import numpy as np
import math, cmath


class CodeGenerator:

    def __init__(self, *expr: Expr, symbols: list[Variable], args: tuple[Variable, ...]=()):
        self.expr = expr
        self.symbols = symbols
        self.args = args
        self.Nsys = len(symbols) - 1

    @property
    def is_arr(self):
        return len(self.expr) > 1

    def _core(self, lang, lib):
        return ', '.join([expr.repr(lang, lib) for expr in self.expr])

    def _return_expr(self, lang, lib, ode_style: bool):
        if ode_style:
            core = ', '.join([expr.write_as_ode(lang, lib, self.symbols, 't', 'q', self.args) for expr in self.expr])
        else:
            core = ', '.join([expr.repr(lang, lib) for expr in self.expr])
        
        return core
    
    def _param_expr(self, lang, ode_style: bool, stack: bool):
        if lang == 'python':
            if ode_style:
                args = 't, q'
                if self.args:
                    args += ', ' + ', '.join([str(i) for i in self.args])
            else:
                args = ', '.join([str(v) for v in self.symbols+list(self.args)])
        elif lang == 'c++':
            if ode_style:
                if self.is_arr:
                    if stack:
                        args = f'const double& t, const vec::StackArray<double, {self.Nsys}>& q'
                    else:
                        args = f'const double& t, const vec::HeapArray<double>& q'
                else:
                    args = f'const double& t, const double& q'
                    
                args += f', const double* args'
            else:
                args = ', '.join([f"double {v}" for v in self.symbols+list(self.args)])
        return args

        
    def get_python(self, lib='math', ode_style=False):

        args = self._param_expr('python', ode_style, None)
        core = self._return_expr('python', lib, ode_style)

        if not self.is_arr:
            code = f'def MyFunc({args}):\n\t return {core}'
        else:
            code = f'def MyFunc({args}):\n\t return numpy.array([{core}])'

        return code
    
    def get_cpp(self, namespace='std', ode_style=False, stack=True):

        args = self._param_expr('c++', ode_style, stack)
        core = self._return_expr('c++', namespace, ode_style)

        if self.is_arr:
            if stack:
                ret_type = f"vec::StackArray<double, {self.Nsys}>"
            else:
                ret_type = f"vec::HeapArray<double>"
        else:
            ret_type = 'double'
        
        code = f'{ret_type} MyFunc({args})'+'{\n\treturn {'+core+'};\n}'
        return code


    def python_callable(self, lib='math', ode_style=False):

        code = self.get_python(lib, ode_style)
        glob_vars = {"math": math, "numpy": np, "cmath": cmath}
        exec(code, glob_vars)
        return glob_vars['MyFunc']
        


def lambdify(*expr: Expr, symbols: list[Variable], lib='math', ode_style=False):
    return CodeGenerator(*expr, symbols=symbols).python_callable(lib, ode_style)

