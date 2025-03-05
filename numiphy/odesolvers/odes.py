from __future__ import annotations
from ._lowlevelode import *

from ..symlib import expressions as sym
from ..symlib.conditional import *
from ..toolkit import tools
from functools import cached_property
import tempfile
import os

# def argument_list(t: sym.Variable, *q:sym.Variable, ode_style=True)->str:
#     res = f'const double& {t}, '
#     if ode_style:
#         x = [sym.Variable(f'q[{i}]') for i in range(len(q))]



# class SymbolicEvent:

#     def __init__(self, name, event: sym.Expr, check_if: Boolean):
#         self.name = name
#         self.event = event
#         self.check = check_if

#     def code(self, t: sym.Variable, *q: sym.Variable):



class SymbolicOde:

    _counter = 0

    def __init__(self, *odesys: sym.Expr, symbols: list[sym.Variable], args: tuple[sym.Variable,...]=()):
        given = symbols + list(args)
        tvar = symbols[0]
        assert tools.all_different(given)
        odesymbols = []
        for ode in odesys:
            for arg in ode.variables:
                if arg not in odesymbols:
                    odesymbols.append(arg)

        if len(odesys) != len(symbols)-1:
            raise ValueError('')
        if tvar in odesymbols:
            assert len(odesymbols) <= len(given)
        else:
            assert len(odesymbols) <= len(given) - 1

        self._odesys = odesys
        self._symbols = tuple(symbols)
        self.args = args

    @property
    def Nsys(self):
        return len(self._odesys)

    def ode_sys(self, variational=False)->tuple[tuple[sym.Expr, ...], tuple[sym.Variable,...]]:
        if not variational:
            return self._odesys, self._symbols
        else:
            assert not any([x.name.startswith('delta_') for x in self._symbols])
            q = self._symbols[1:]
            delq = [sym.Variable('delta_'+qi.name) for qi in q]
            n = len(self._odesys)
            var_odesys = []
            for i in range(n):
                var_odesys.append(sum([self._odesys[i].diff(q[j])*delq[j] for j in range(n)]))
            
            odesys = self._odesys + tuple(var_odesys)
            symbols = self._symbols + tuple(delq)
            return odesys, symbols
    
    def codegen(self, variational=False):
        odesys, symbols = self.ode_sys(variational=variational)
        return sym.CodeGenerator(*odesys, symbols=symbols, args=self.args)
    
    def ode(self, lowlevel=True, stack=True, variational=False):
        if lowlevel:
            return self.to_lowlevel(stack=stack, variational=variational)
        else:
            return self.to_python(variational=variational)

    def to_python(self, variational=False):
        df = self.codegen(variational).python_callable(ode_style=True)
        return LowLevelODE(df)
    
    def to_lowlevel(self, stack=True, variational=False)->LowLevelODE:
        if variational:
            return self._lowlevel_stack_var.copy() if stack else self._lowlevel_heap_var.copy()
        else:
            return self._lowlevel_stack.copy() if stack else self._lowlevel_heap.copy()
    
    def generate_cpp_file(self, directory, module_name, stack: bool, variational=False):
        if not os.path.exists(directory):
            raise RuntimeError(f'Directory "{directory}" does not exist')
        code = self.codegen(variational).get_cpp(ode_style=True, stack=stack)
        cpp_code = f'#include <odepack/pyode.hpp>\n\n{code}\n\n'
        cpp_code += f"PYBIND11_MODULE({module_name}, m){{\ndefine_lowlevel_ode(m, MyFunc);\n}}"
        cpp_file = os.path.join(directory, f"{module_name}.cpp")

        with open(cpp_file, "w") as f:
            f.write(cpp_code)

        return os.path.join(directory, f'{module_name}.cpp')
    
    def compile(self, directory: str, module_name, stack=True, variational=False):
        if not os.path.exists(directory):
            raise RuntimeError(f"Cannot compile ode at {directory}: Path does not exist")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            cpp_file = self.generate_cpp_file(temp_dir, module_name, stack, variational)
            tools.compile(cpp_file, directory, module_name)
        
    def _to_lowlevel(self, stack=True, variational=False)->LowLevelODE:
        c = self.__class__._counter
        modname = f"ode_module{c}"

        with tempfile.TemporaryDirectory() as so_dir:
            self.compile(so_dir, modname, stack=stack, variational=variational)
            temp_module = tools.import_lowlevel_module(so_dir, modname)

        self.__class__._counter += 1
        return temp_module.ode()
    
    @cached_property
    def _lowlevel_stack(self):
        return self._to_lowlevel(stack=True, variational=False)
    
    @cached_property
    def _lowlevel_heap(self):
        return self._to_lowlevel(stack=False, variational=False)

    @cached_property
    def _lowlevel_stack_var(self):
        return self._to_lowlevel(stack=True, variational=True)
    
    @cached_property
    def _lowlevel_heap_var(self):
        return self._to_lowlevel(stack=False, variational=True)
