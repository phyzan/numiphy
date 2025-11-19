from __future__ import annotations
from ..symlib.symcore import *
from ..symlib.pylambda import *
from typing import Iterable, Any, TypeAlias
from ..toolkit import tools
import os
import tempfile
from ..symlib.pylambda import _CallableFunction, _BooleanCallable, _ScalarCallable, _TensorCallable

Pointer: TypeAlias = Any

class LowLevelCallable(_CallableFunction):

    def core_impl(self)->str:...

    def scalar_id(self, scalar_name):
        if (self._is_array[scalar_name]):
            return f'const double* {scalar_name}'
        else:
            return f'const double& {scalar_name}'
    
    def _code(self, name, return_type, arg_list, code_impl):
        return f'{return_type} {name}({arg_list})'+'{\n'+code_impl+'\n}'
    
    def code(self, name: str):
        return self._code(name, self.return_id(), self.argument_list()+', const void*', self.core_impl())
    
    def lambda_code(self):
        return f'[]({self.argument_list()}) -> {self.return_id()} ' + '{' +f'{self.core_impl()}'+'}'
    
    def compile(self, directory: str = None, module_name: str = None)->Pointer:
        return compile_funcs([self], directory=directory, module_name=module_name)[0]
    
    def to_python_callable(self)->PythonCallable:
        raise NotImplementedError('')


class BooleanLowLevelCallable(_BooleanCallable, LowLevelCallable):

    def core_impl(self):
        res = self.expr.varsub(self._map).lowlevel_repr("double")
        return f"return {res};"
    
    def to_python_callable(self):
        p = self._constuctor_params
        return BooleanPythonCallable(p[0], *p[1], **p[2])


class ScalarLowLevelCallable(_ScalarCallable, LowLevelCallable):

    def core_impl(self):
        res = self.expr.varsub(self._map).lowlevel_repr("double")
        return f"return {res};"
    
    def return_id(self):
        return 'double'
    
    def to_python_callable(self):
        p = self._constuctor_params
        return ScalarPythonCallable(p[0], *p[1], **p[2])


class TensorLowLevelCallable(_TensorCallable, LowLevelCallable):

    @property
    def new_array(self)->list[Expr]:
        return [arg.varsub(self._map) for arg in self.array]

    def return_id(self):
        return 'void'
    
    def core_impl(self):
        res = self.new_array
        return '\n'.join([f'result[{i}] = {res[i].lowlevel_repr("double")};' for i in range(len(res))])

    def lambda_code(self):
        return f'[](double* result, {self.argument_list()})' + '{' +f'{self.core_impl()}'+'}'

    def argument_list(self):
        arglist = 'double* result, '+_CallableFunction.argument_list(self)
        return arglist
    
    def to_python_callable(self):
        p = self._constuctor_params
        return TensorPythonCallable(p[0], *p[1], **p[2])


class CompileTemplate:

    def __init__(self, module_name: str = None, directory: str = None):
        self.__module_name = module_name
        self.__directory = directory if directory is not None else tools.get_source_dir()
    
    @property
    def directory(self):
        return self.__directory
    
    @property
    def module_name(self):
        return self.__module_name

    @cached_property
    def lowlevel_callables(self)->tuple[LowLevelCallable, ...]:
        '''
        override
        '''
        raise NotImplementedError('')
    
    @cached_property
    def _code(self):
        names = [f'f_{i}' for i in range(len(self.lowlevel_callables))]
        return '\n\n'.join([g.code(name) if g is not None else '' for g, name in zip(self.lowlevel_callables, names)])
    
    @property
    def _funcs_path(self):
        return os.path.join(self.directory, f"ode_callables.txt")
    
    def compile(self)->tuple:
        result = compile_funcs(self.lowlevel_callables, self.directory, self.module_name)
        with open(self._funcs_path, "w") as f:
            f.write(self._code)
        return result
    
    @cached_property
    def pointers(self)->tuple[Pointer,...]:
        if os.path.exists(self._funcs_path):
            try:
                with open(self._funcs_path, "r") as f:
                    saved_code = f.read()
                
                if saved_code == self._code:
                    # Callables match, try to import the module without recompiling
                    return tools.import_lowlevel_module(self.directory, self.module_name).pointers()
            except:
                pass
        # If no match or error, recompile
        return self.compile()


def generate_cpp_file(code, directory, module_name):
    if not os.path.exists(directory):
        raise RuntimeError(f'Directory "{directory}" does not exist')
    cpp_file = os.path.join(directory, f"{module_name}.cpp")
    with open(cpp_file, "w") as f:
        f.write(code)
    
    return os.path.join(directory, f'{module_name}.cpp')


def compile_funcs(functions: Iterable[LowLevelCallable], directory: str = None, module_name: str = None)->tuple[Pointer,...]:
    none_modname = module_name is None
    if (none_modname):
        module_name = tools.random_module_name()
    header = "#include <pybind11/pybind11.h>\n\n#include <complex>\n\nusing std::complex, std::imag, std::real, std::numbers::pi;\n\nnamespace py = pybind11;"

    names = [f"_lowlevel_func_{i}" for i in range(len(functions))]

    code_block = '\n\n'.join([f.code(name) if f is not None else '' for f, name in zip(functions, names)])

    array = "py::make_tuple("+", ".join([f'reinterpret_cast<const void*>({name})' if f is not None else 'nullptr' for f, name in zip(functions, names)])+")"

    py_func = '\n\tm.def("pointers", [](){return '+array+';});'
    pybind_cond = f"PYBIND11_MODULE({module_name}, m)"+'{'+py_func+'\n}'
    items = [header, code_block, pybind_cond]
    code = "\n\n".join(items)
    if none_modname:
        with tempfile.TemporaryDirectory() as so_dir:
            with tempfile.TemporaryDirectory() as temp_dir:
                cpp_file = generate_cpp_file(code, temp_dir, module_name)
                tools.compile(cpp_file, so_dir, module_name)
            temp_module = tools.import_lowlevel_module(so_dir, module_name)
    else:
        so_dir = directory if directory is not None else tools.get_source_dir()
        with tempfile.TemporaryDirectory() as temp_dir:
            cpp_file = generate_cpp_file(code, temp_dir, module_name)
            tools.compile(cpp_file, so_dir, module_name)
        temp_module = tools.import_lowlevel_module(so_dir, module_name)
    return temp_module.pointers()
