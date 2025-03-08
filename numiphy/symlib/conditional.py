from __future__ import annotations

from .symcore import *
from .symcore import _Expr, _Symbol
from typing import Dict
import numpy as np
from typing import Callable
import numpy.typing as npt


class Boolean:

    operator: str
    Args: tuple

    def __str__(self):
        return self.repr()
    
    def __repr__(self):
        return str(self)

    def __eq__(self, other: Boolean):
        return type(self) is type(other) and self.Args == other.Args
    
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

    def repr(self, lib="")->str:...

    def lowlevel_repr(self, scalar_type="double")->str:...

    def elementwise_eval(self, x: Dict[_Symbol, np.ndarray], **kwargs)->np.ndarray[bool]:...

    def do(self, func: str, *args, **kwargs)->Boolean:...


class Comparison(Boolean):

    Args: tuple[_Expr, _Expr]

    def __new__(cls, a, b):
        if isinstance(a, _Expr) and not isinstance(b, _Expr):
            b = a._asexpr(b)
        elif not isinstance(a, _Expr) and isinstance(b, _Expr):
            a = b._asexpr(a)
        if a.isNumber and b.isNumber:
            return cls.evaluate(a.eval().value, b.eval().value)
        else:
            obj = super().__new__(cls)
            obj.Args = (a, b)
            return obj
    
    @property
    def a(self):
        return self.Args[0]
    
    @property
    def b(self):
        return self.Args[1]

    def repr(self, lib="")->str:
        if lib == '':
            if isinstance(self, (Eq, Neq)):
                return f'{self.__class__.__name__}({self.a}, {self.b})'
            else:
                return f'{self.a} {self.operator} {self.b}'
        return f'{self.a.repr(lib)} {self.operator} {self.b.repr(lib)}'

    def lowlevel_repr(self, scalar_type="double"):
        return f'{self.a.lowlevel_repr(scalar_type)} {self.operator} {self.b.lowlevel_repr(scalar_type)}'

    @classmethod
    def evaluate(cls, a: float, b: float)->bool:...

    def elementwise_eval(self, x: Dict[_Symbol, np.ndarray], **kwargs)->np.ndarray[bool]:
        return self.evaluate(self.a.get_ndarray(x, **kwargs), self.b.get_ndarray(x, **kwargs))

    def do(self, func: str, *args, **kwargs):
        a = getattr(self.a, func)(*args, **kwargs)
        b = getattr(self.b, func)(*args, **kwargs)
        return self.__class__(a, b)


class Logical(Boolean):

    Args: tuple[Boolean, Boolean]
    cpp_op: str
    np_logical: Callable[[npt.NDArray, npt.NDArray], npt.NDArray[np.bool_]]

    def __new__(cls, left: Boolean, right: Boolean):
        if cls is Logical:
            raise ValueError("The Logical class cannot be directly instanciated")
        
        obj = super().__new__(cls)
        obj.Args = (left, right)
        return obj
    
    @classmethod
    def _assert(cls, a, b):
        if not isinstance(a, (Boolean, bool)) or not isinstance(b, (Boolean, bool)):
            raise ValueError(f"Left and right expressions need to be instances of the Boolean class, not {a.__class__} and {b.__class__}")

    @property
    def left(self):
        return self.Args[0]
    
    @property
    def right(self):
        return self.Args[1]
    
    def repr(self, lib=""):
        return f'({self.left.repr(lib)}) {self.operator} ({self.right.repr(lib)})'
    
    def lowlevel_repr(self, scalar_type="double"):
        return f'({self.left.lowlevel_repr(scalar_type)}) {self.cpp_op} ({self.right.lowlevel_repr(scalar_type)})'

    def elementwise_eval(self, x, **kwargs):
        left = self.left.elementwise_eval(x, **kwargs)
        right = self.right.elementwise_eval(x, **kwargs)
        return self.np_logical(left, right)
    
    def do(self, func: str, *args, **kwargs):
        a = self.left.do(func, *args, **kwargs)
        b = self.right.do(func, *args, **kwargs)
        return self.__class__(a, b)


class Not(Boolean):

    operator = '~'
    arg: Boolean

    def __new__(cls, arg: Boolean):
        if isinstance(arg, bool):
            return not bool
        elif not isinstance(arg, Boolean):
            raise ValueError("Argument must be of the Boolean class")
        obj = super().__new__(cls)
        obj.arg = arg
        return obj

    def repr(self, lib=""):
        return f'({self.operator} ({self.arg.repr(lib)}))'
    
    def lowlevel_repr(self, scalar_type="double"):
        return f'!({self.arg.lowlevel_repr(scalar_type)})'
    
    def elementwise_eval(self, x, **kwargs):
        arr = self.arg.elementwise_eval(x, **kwargs)
        return np.logical_not(arr)
    
    def do(self, func, *args, **kwargs):
        f = self.arg.do(func, *args, **kwargs)
        return Not(f)


class Gt(Comparison):

    operator = '>'

    @classmethod
    def evaluate(cls, a: float, b: float) -> bool:
        return a>b


class Lt(Comparison):

    operator = '<'

    @classmethod
    def evaluate(cls, a: float, b: float) -> bool:
        return a<b


class Ge(Comparison):

    operator = '>='

    @classmethod
    def evaluate(cls, a: float, b: float) -> bool:
        return a>=b


class Le(Comparison):

    operator = '<='

    @classmethod
    def evaluate(cls, a: float, b: float) -> bool:
        return a<=b


class Eq(Comparison):

    operator = '=='

    @classmethod
    def evaluate(cls, a: float, b: float) -> bool:
        return a==b


class Neq(Comparison):

    operator = '!='

    @classmethod
    def evaluate(cls, a: float, b: float) -> bool:
        return a!=b


class And(Logical):

    cpp_op = '&&'
    operator = '&'
    np_logical = np.logical_and

    def __new__(cls, left, right):
        cls._assert(left, right)
        if left is False or right is False:
            return False
        elif left is True and right is True:
            return True
        else:
            return super().__new__(cls, left, right)


class Or(Logical):

    cpp_op = '||'
    operator = '|'
    np_logical = np.logical_or

    def __new__(cls, left, right):
        cls._assert(left, right)
        if left is True or right is True:
            return True
        elif left is False and right is False:
            return False
        else:
            return super().__new__(cls, left, right)



'''
TODO

In the future, implement And, Or classes:

e.g. (x<0) and (x>0) is a condition

'''