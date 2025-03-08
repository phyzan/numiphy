from .symexpr import *
import numpy as np
import math, cmath
from typing import Iterable
from ..conditional import Boolean
from ...toolkit import tools



class ContainerLowLevel:

    def __init__(self, array_type: str, *args: Variable):
        self.array_type = array_type
        self.args = args
    
    def as_argument(self, scalar_type: str, name: str):
        return f'const {self.array_type}<{scalar_type}>& {name}'
    
    def map(self, array_name: str)->dict[Variable, Variable]:
        return {self.args[i]: Variable(f'{array_name}[{i}]') for i in range(len(self.args))}
    
    def converted(self, expr: Expr):
        return expr.replace(self.map())


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
            f = self.containers[name].converted(name, f)
        return f.lowlevel_repr(scalar_type)

    def return_type(self, scalar_type: str):
        return scalar_type


class VectorLowLevel(FunctionLowLevel):

    def __init__(self, array_type: str, array: Iterable[Expr], *args: Variable, **containers: ContainerLowLevel):
        self.array_type = array_type
        self.array = array

        super().__init__(*args, **containers)

    def return_expression(self):
        r = ", ".join([self.containers[name].converted(name) for name in self.containers])
        return '{'+r+'}'
        
    def return_type(self, scalar_type: str):
        return f'{self.array_type}<{scalar_type}>'












class SymbolicOde:

    def __init__(self, ode_sys: Iterable[Expr], t: Variable, *q: Variable, args: Iterable[Variable]):
        self.ode_sys = tuple(ode_sys)
        self.args = tuple(args)
        self.t = t
        self.q = q

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
        
    def lowlevel_code(self, stack=True, variational=False):
        pass



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
                args = ', '.join([str(v) for v in list(self.symbols)+list(self.args)])
        elif lang == 'c++':
            if ode_style:
                if self.is_arr:
                    if stack:
                        args = f'const double& t, const vec<double, {self.Nsys}>& q'
                    else:
                        args = f'const double& t, const vec<double>& q'
                else:
                    args = f'const double& t, const double& q'
                    
                args += f', const std::vector<double>& args'
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
                ret_type = f"vec<double, {self.Nsys}>"
            else:
                ret_type = f"vec<double>"
        else:
            ret_type = 'double'
        
        if stack:
            code = f'{ret_type} MyFunc({args})'+'{\n\treturn {'+core+'};\n}'
        else:
            code = f'{ret_type} MyFunc({args})'+'{'+f'\n\tvec<double> y({len(self.expr)}); y << {core};\n\treturn y;'+'\n}'
        return code


    def python_callable(self, lib='math', ode_style=False):

        code = self.get_python(lib, ode_style)
        glob_vars = {"math": math, "numpy": np, "cmath": cmath}
        exec(code, glob_vars)
        return glob_vars['MyFunc']


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
    return CodeGenerator(*expr, symbols=symbols).python_callable(lib, ode_style)


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