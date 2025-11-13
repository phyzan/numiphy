from __future__ import annotations

from .symcore import Expr, asexpr, Number, Symbol, S
from typing import Dict
import numpy as np
from typing import Callable
import numpy.typing as npt
import torch


class Boolean(Expr):

    operator: str
    torch_repr: str

    def __and__(self, other: Boolean):
        return And(self, other)
    
    def __or__(self, other: Boolean):
        return Or(self, other)
    
    def __rand__(self, other):
        return And(self, other)
    
    def __ror__(self, other):
        return Or(self, other)
    
    def __invert__(self):
        return Not(self)

    def _diff(self, var):
        return S.Zero


class Comparison(Boolean):

    def __new__(cls, a, b, simplify=True):
        a, b = asexpr(a), asexpr(b)
        if isinstance(a, Number) and isinstance(b, Number):
            return cls.evaluate(a.value, b.value)
        else:
            return Boolean.__new__(cls, a, b)

    @property
    def a(self)->Expr:
        return self.args[0]
    
    @property
    def b(self)->Expr:
        return self.args[1]

    def repr(self, lib="", **kwargs)->str:
        if lib == '':
            if isinstance(self, (Eq, Neq)):
                return f'{self.__class__.__name__}({self.a}, {self.b})'
            else:
                return f'({self.a} {self.operator} {self.b})'
        elif lib == 'torch' and not self.isNumber and kwargs.get('out', False):
            return f'torch.{self.torch_repr}({self.a.repr(lib, **kwargs)}, {self.b.repr(lib, **kwargs)}, out=out)'
        return f'({self.a.repr(lib, **kwargs)} {self.operator} {self.b.repr(lib, **kwargs)})'

    def lowlevel_repr(self, scalar_type="double"):
        return f'({self.a.lowlevel_repr(scalar_type)} {self.operator} {self.b.lowlevel_repr(scalar_type)})'

    @classmethod
    def evaluate(cls, a: float, b: float)->bool:
        raise NotImplementedError('')

    def get_ndarray(self, x: Dict[Symbol, np.ndarray], **kwargs)->np.ndarray[bool]:
        return self.evaluate(self.a.get_ndarray(x, **kwargs), self.b.get_ndarray(x, **kwargs))
    
    def eval(self):
        a, b = self.a.eval(), self.b.eval()
        if isinstance(a, Number) and isinstance(b, Number):
            return asexpr(self.evaluate(a.value, b.value))
        else:
            return self.init(a, b)


class Logical(Boolean):

    cpp_op: str
    np_logical: Callable[[npt.NDArray, npt.NDArray], npt.NDArray[np.bool_]]

    def __new__(cls, left: Boolean, right: Boolean, simplify=True):
        if cls is Logical:
            raise ValueError("The Logical class cannot be directly instanciated")
        return Boolean.__new__(cls, left, right)
    
    @classmethod
    def _assert(cls, a, b):
        if not isinstance(a, (Boolean, bool)) or not isinstance(b, (Boolean, bool)):
            raise ValueError(f"Left and right expressions need to be instances of the Boolean class, not {a.__class__} and {b.__class__}")
        
    @classmethod
    def evaluate(cls, a: bool, b: bool)->bool:
        raise NotImplementedError('')

    @property
    def left(self)->Boolean:
        return self.args[0]
    
    @property
    def right(self)->Boolean:
        return self.args[1]
    
    def repr(self, lib="", **kwargs):
        if lib == 'torch' and not self.isNumber and kwargs.get('out', False):
            return f'torch.{self.torch_repr}({self.left.repr(lib, **kwargs)}, {self.right.repr(lib, **kwargs)}, out=out)'
        return f'(({self.left.repr(lib, **kwargs)}) {self.operator} ({self.right.repr(lib, **kwargs)}))'
    
    def lowlevel_repr(self, scalar_type="double"):
        return f'(({self.left.lowlevel_repr(scalar_type)}) {self.cpp_op} ({self.right.lowlevel_repr(scalar_type)}))'
    
    def get_ndarray(self, x, **kwargs):
        left = self.left.get_ndarray(x, **kwargs)
        right = self.right.get_ndarray(x, **kwargs)
        return self.np_logical(left, right)
    
    def eval(self):
        a, b = self.left.eval(), self.right.eval()
        if isinstance(a, Number):
            a = a.value
        if isinstance(b, Number):
            b = b.value
        return self.init(a, b)


class Not(Boolean):

    operator = '~'
    torch_repr = 'logical_not'
    _priority = 17

    def __new__(cls, arg: Boolean, simplify=True):
        if isinstance(arg, bool):
            return not arg
        elif not isinstance(arg, Boolean):
            raise ValueError("Argument must be of the Boolean class")
        return Boolean.__new__(cls, arg)

    @property
    def arg(self)->Boolean:
        return self._args[0]

    def repr(self, lib="", **kwargs):
        if lib == 'torch' and not self.isNumber and kwargs.get('out', False):
            return f'torch.{self.torch_repr}({self.arg.repr(lib, **kwargs)}, out=out)'
        return f'({self.operator} ({self.arg.repr(lib, **kwargs)}))'
    
    def lowlevel_repr(self, scalar_type="double"):
        return f'!({self.arg.lowlevel_repr(scalar_type)})'
    
    def get_ndarray(self, x, **kwargs):
        arr = self.arg.get_ndarray(x, **kwargs)
        return np.logical_not(arr)
    
    def eval(self):
        a = self.arg.eval()
        if isinstance(a, Number):
            return asexpr(not a.value)
        else:
            return self.init(a)


class Gt(Comparison):

    operator = '>'
    torch_repr = 'gt'
    _priority = 18

    @classmethod
    def evaluate(cls, a: float, b: float) -> bool:
        return a>b


class Lt(Comparison):

    operator = '<'
    torch_repr = 'lt'
    _priority = 19

    @classmethod
    def evaluate(cls, a: float, b: float) -> bool:
        return a<b


class Ge(Comparison):

    operator = '>='
    torch_repr = 'ge'
    _priority = 20

    @classmethod
    def evaluate(cls, a: float, b: float) -> bool:
        return a>=b


class Le(Comparison):

    operator = '<='
    torch_repr = 'le'
    _priority = 21

    @classmethod
    def evaluate(cls, a: float, b: float) -> bool:
        return a<=b


class Eq(Comparison):

    operator = '=='
    torch_repr = 'eq'
    _priority = 22

    @classmethod
    def evaluate(cls, a: float, b: float) -> bool:
        return a==b


class Neq(Comparison):

    operator = '!='
    torch_repr = 'ne'
    _priority = 23

    @classmethod
    def evaluate(cls, a: float, b: float) -> bool:
        return a!=b


class And(Logical):

    cpp_op = '&&'
    operator = '&'
    torch_repr = 'logical_and'
    np_logical = np.logical_and
    _priority = 24

    def __new__(cls, left, right, simplify=True):
        cls._assert(left, right)
        if left is False or right is False:
            return False
        elif left is True and right is True:
            return True
        else:
            return Logical.__new__(cls, left, right, simplify=simplify)
        
    @classmethod
    def evaluate(cls, a, b):
        return a and b


class Or(Logical):

    cpp_op = '||'
    operator = '|'
    torch_repr = 'logical_or'
    np_logical = np.logical_or
    _priority = 25

    def __new__(cls, left, right, simplify=True):
        cls._assert(left, right)
        if left is True or right is True:
            return True
        elif left is False and right is False:
            return False
        else:
            return Logical.__new__(cls, left, right, simplify=simplify)
        
    @classmethod
    def evaluate(cls, a, b):
        return a or b

