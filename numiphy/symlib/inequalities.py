from __future__ import annotations

from .symcore import *
from .symcore import _Expr, _Symbol
from typing import Dict
import numpy as np
from typing import Callable


class Condition:

    sign: str
    a: _Expr
    b: _Expr

    def __new__(cls, a, b):
        if isinstance(a, _Expr) and not isinstance(b, _Expr):
            b = a._asexpr(b)
        elif not isinstance(a, _Expr) and isinstance(b, _Expr):
            a = b._asexpr(a)
        if a.isNumber and b.isNumber:
            return cls.evaluate(a.eval().value, b.eval().value)
        else:
            obj = super().__new__(cls)
            obj.a = a
            obj.b = b
            return obj
    
    def repr(self, lang="python", lib=""):
        if lang == 'python' and lib == '':
            if isinstance(self, Eq):
                return f'Eq({self.a}, {self.b})'
            else:
                return f'{self.a} {self.sign} {self.b}'
        return f'{self.a.repr(lang, lib)} {self.sign} {self.b.repr(lang, lib)}'
        

    def __str__(self):
        return self.repr()
    
    def __repr__(self):
        return str(self)
    
    def __eq__(self, other: Condition):
        return type(self) is type(other) and self.a == other.a and self.b == other.b

    @classmethod
    def evaluate(cls, a: float, b: float)->bool:...

    def elementwise_eval(self, x: Dict[_Symbol, np.ndarray], **kwargs)->np.ndarray[bool]:
        return self.evaluate(self.a.get_ndarray(x, **kwargs), self.b.get_ndarray(x, **kwargs))

    def do(self, func: str, *args, **kwargs):
        a = getattr(self.a, func)(*args, **kwargs)
        b = getattr(self.b, func)(*args, **kwargs)
        return self.__class__(a, b)


class Gt(Condition):

    sign = '>'

    @classmethod
    def evaluate(cls, a: float, b: float) -> bool:
        return a>b


class Lt(Condition):

    sign = '<'

    @classmethod
    def evaluate(cls, a: float, b: float) -> bool:
        return a<b


class Ge(Condition):

    sign = '>='

    @classmethod
    def evaluate(cls, a: float, b: float) -> bool:
        return a>=b


class Le(Condition):

    sign = '<='

    @classmethod
    def evaluate(cls, a: float, b: float) -> bool:
        return a<=b


class Eq(Condition):

    sign = '=='

    @classmethod
    def evaluate(cls, a: float, b: float) -> bool:
        return a==b



'''
TODO

In the future, implement And, Or classes:

e.g. (x<0) and (x>0) is a condition

'''