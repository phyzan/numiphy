from __future__ import annotations
from ..symlib.symcore import *
from ..symlib.pylambda import *
from typing import Iterable, Any, TypeAlias
from ..toolkit import tools
import os
import tempfile
from ..symlib.pylambda import _CallableFunction, _BooleanCallable, _ScalarCallable, _TensorCallable

Pointer: TypeAlias = Any
DEFAULT_SCALAR_TYPE = "double"

class LowLevelCallable(_CallableFunction):

    def __init__(self, result, *args, scalar_type = DEFAULT_SCALAR_TYPE, **kwargs):
        self._scalar_type = scalar_type
        super().__init__(result, *args, **kwargs)

    def core_impl(self)->str:...

    def scalar_id(self, scalar_name):
        if (self._is_array[scalar_name]):
            return f'const {self._scalar_type}* {scalar_name}'
        else:
            return f'const {self._scalar_type}& {scalar_name}'
    
    def _code(self, name, return_type, arg_list, code_impl):
        return f'{return_type} {name}({arg_list})'+'{\n'+code_impl+'\n}'
    
    def code(self, name: str):
        return self._code(name, self.return_id(), self.argument_list()+', const void*', self.core_impl())
    
    def lambda_code(self):
        return f'[]({self.argument_list()}) -> {self.return_id()} ' + '{' +f'{self.core_impl()}'+'}'
    
    def compile(self, directory: str = None, module_name: str = None)->Pointer:
        return compile_funcs([self], directory=directory, module_name=module_name)[0][0]
    
    def to_python_callable(self)->PythonCallable:
        raise NotImplementedError('')


class BooleanLowLevelCallable(LowLevelCallable, _BooleanCallable):

    def __init__(self, expr: Boolean, *args: Symbol, scalar_type = DEFAULT_SCALAR_TYPE, **kwargs: Symbol|Iterable[Symbol]):
        LowLevelCallable.__init__(self, expr, *args, scalar_type=scalar_type, **kwargs)

    def core_impl(self):
        res = self.expr.varsub(self._map).lowlevel_repr(self._scalar_type)
        return f"return {res};"
    
    def to_python_callable(self):
        p = self._constructor_params
        return BooleanPythonCallable(p[0], *p[1], **p[2])


class ScalarLowLevelCallable(LowLevelCallable, _ScalarCallable):

    def __init__(self, expr: Expr, *args: Symbol, scalar_type = DEFAULT_SCALAR_TYPE, **kwargs: Symbol|Iterable[Symbol]):
        LowLevelCallable.__init__(self, expr, *args, scalar_type=scalar_type, **kwargs)

    def core_impl(self):
        res = self.expr.varsub(self._map).lowlevel_repr(self._scalar_type)
        return f"return {res};"
    
    def return_id(self):
        return self._scalar_type
    
    def to_python_callable(self):
        p = self._constructor_params
        return ScalarPythonCallable(p[0], *p[1], **p[2])


class TensorLowLevelCallable(LowLevelCallable, _TensorCallable):

    def __init__(self, array: Iterable, *args: Symbol, scalar_type = DEFAULT_SCALAR_TYPE, **kwargs: Symbol|Iterable[Symbol]):
        self._scalar_type = scalar_type
        _TensorCallable.__init__(self, array, *args, **kwargs)

    @property
    def new_array(self)->list[Expr]:
        return [arg.varsub(self._map) for arg in self.array]

    def return_id(self):
        return 'void'
    
    def core_impl(self):
        res = self.new_array
        return '\n'.join([f'result[{i}] = {res[i].lowlevel_repr(self._scalar_type)};' for i in range(len(res))])

    def lambda_code(self):
        return f'[]({self._scalar_type}* result, {self.argument_list()})' + '{' +f'{self.core_impl()}'+'}'

    def argument_list(self):
        arglist = f'{self._scalar_type}* result, '+_CallableFunction.argument_list(self)
        return arglist
    
    def to_python_callable(self):
        p = self._constructor_params
        return TensorPythonCallable(p[0], *p[1], **p[2])


class CompileTemplate:

    def __init__(self, module_name: str = None, directory: str = None):
        self.__module_name = module_name
        self.__directory = directory if directory is not None else tools.get_source_dir()
        self.__nan_dir = directory is None
    
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
        return generate_cpp_code(self.lowlevel_callables, self.module_name)
    
    @property
    def _funcs_path(self):
        return os.path.join(self.directory, f"ode_callables.cpp")
    
    def compile(self)->tuple:
        if not self.__nan_dir:
            with open(self._funcs_path, "w") as f:
                f.write(self._code)
        result = compile_funcs(self.lowlevel_callables, self.directory, self.module_name)
        return result[0]
    
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

def generate_cpp_code(functions: Iterable[LowLevelCallable], module_name: str, extra_header_block: str = "", extra_code_block: str = "", extra_funcs: Iterable[tuple[str, str]]=())->str:

    has_mpreal = any([f._scalar_type == "mpreal" for f in functions])
    mpreal_include = '#include <mpreal.h>\n\n' if has_mpreal else ''
    mpreal_use = 'using mpfr::mpreal;\n\n' if has_mpreal else ''
    header = "#include <pybind11/pybind11.h>\n\n" + mpreal_include + extra_header_block + "\n#include <complex>\n\nusing std::complex, std::imag, std::real, std::numbers::pi;\n\nnamespace py = pybind11;\n\n" + mpreal_use

    names = [f"func{i}" for i in range(len(functions))]

    code_block = '\n\n'.join([f.code(name) for f, name in zip(functions, names)])
    
    code_block = extra_code_block+'\n\n' + code_block + '\n\n'

    array = "py::make_tuple("+", ".join([f'reinterpret_cast<const void*>({name})' for f, name in zip(functions, names)])+")"

    py_func = '\n\tm.def("pointers", [](){return '+array+';});'
    
    for func_name, func_code in extra_funcs:
        py_func += f'\n\tm.def("{func_name}", {func_code});'
    pybind_cond = f"PYBIND11_MODULE({module_name}, m)"+'{'+py_func+'\n}'
    items = [header, code_block, pybind_cond]
    code = "\n\n".join(items)
    return code

def compile_funcs(functions: Iterable[LowLevelCallable], directory: str = None, module_name: str = None, extra_header_block: str = "", extra_code_block: str = "", extra_funcs: Iterable[tuple[str, str]] = (), links: Iterable[tuple[str, str]] = (), extra_flags: Iterable[str]=())->tuple[tuple[Pointer,...], ...]:
    '''
    Converts a list of expressions into C++ syntax, and compiles them as separate functions

    Parameters
    --------------------
    directory: 
        The directory to store the compiled python module that contains the pointers to the compiled functions.
    module_name:
        The name of the compiled module

    If the module name is None, the module is compiled in a temporary directory
    If only the directory is None, the module is compiled in the current working directory
    Otherwise, it is compiled in the provided directory

    Returns
    ---------------------------
    pointers: tuple of void pointers, each pointing to a compiled function, in the same order
        as they were provided
    '''
    none_modname = module_name is None
    if (none_modname):
        module_name = tools.random_module_name()
    code = generate_cpp_code(functions, module_name, extra_header_block=extra_header_block, extra_code_block=extra_code_block, extra_funcs=extra_funcs)
    if none_modname:
        with tempfile.TemporaryDirectory() as so_dir:
            with tempfile.TemporaryDirectory() as temp_dir:
                cpp_file = generate_cpp_file(code, temp_dir, module_name)
                tools.compile(cpp_file, so_dir, module_name, links=links, extra_flags=extra_flags)
            temp_module = tools.import_lowlevel_module(so_dir, module_name)
    else:
        so_dir = directory if directory is not None else os.getcwd()
        with tempfile.TemporaryDirectory() as temp_dir:
            cpp_file = generate_cpp_file(code, temp_dir, module_name)
            tools.compile(cpp_file, so_dir, module_name, links=links, extra_flags=extra_flags)
        temp_module = tools.import_lowlevel_module(so_dir, module_name)
    return temp_module.pointers(), *[getattr(temp_module, func_name) for func_name, _ in extra_funcs]