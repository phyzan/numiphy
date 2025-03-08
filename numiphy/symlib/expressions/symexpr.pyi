from ..symcore import *
from ..symcore import _Expr, _Singleton, _Subs, _Add, _Mul, _Pow, _Float, _Rational, _Integer, _Special, _Complex, _Symbol, _Derivative, _Integral, _ScalarField, _DummyScalarField, _Piecewise, _Any
from typing import Type, Self, Dict, Callable
import numpy as np
from functools import cached_property
from ...findiffs import grids


class Expr(_Expr):

    S: Singleton

    def __add__(self, other)->Expr:...
    
    def __sub__(self, other)->Expr:...
    
    def __mul__(self, other)->Expr:...
    
    def __truediv__(self, other)->Expr:...
    
    def __pow__(self, other)->Expr:...
    
    def __neg__(self)->Expr:...
    
    def __radd__(self, other)->Expr:...
    
    def __rsub__(self, other)->Expr:...
    
    def __rmul__(self, other)->Expr:...
    
    def __rtruediv__(self, other)->Expr:...
    
    def __rpow__(self, other)->Expr:...
    
    def __abs__(self)->Expr:...

    def __repr__(self)->str:...

    def __str__(self)->str:...
    
    def __hash__(self)->int:...

    def __eq__(self, other)->bool:...

    def _diff(self, var: Variable)->Expr:...
    
    def _equals(self, other: Type[Self])->bool:... #other must be same class as self

    @classmethod
    def _asexpr(cls, arg)->Expr:...# also checks if it is operator or not e.g.
    
    @classmethod
    def _add(cls, *args, simplify=True)->Expr:...# return Add.init(...)

    @classmethod
    def _mul(cls, *args, simplify=True)->Expr:...

    @classmethod
    def _pow(cls, base, power, simplify=True)->Expr:...

    @classmethod
    def _sin(cls, arg: Expr, simplify=True)->Expr:...

    @classmethod
    def _cos(cls, arg: Expr, simplify=True)->Expr:...

    @classmethod
    def _exp(cls, arg: Expr, simplify=True)->Expr:...

    @classmethod
    def _log(cls, arg: Expr, simplify=True)->Expr:...

    @classmethod
    def _tan(cls, arg: Expr, simplify=True)->Expr:...

    @classmethod
    def _abs(cls, arg: Expr, simplify=True)->Expr:...

    @classmethod
    def _rat(cls, m: int, n: int)->Expr:...##### def __new__ in Rational and RationalOp

    @classmethod
    def _derivative(cls, f: Expr, *vars: Variable, simplify=True)->Expr:...

    @classmethod
    def _subs(cls, expr: Expr, vals: Dict[Expr, Expr], simplify=True)->Expr:...

    @classmethod
    def _dummy(cls, arr: np.ndarray, grid: grids.Grid, *vars: Variable)->DummyScalarField:...

    @property
    def args(self)->tuple[Expr, ...]:...

    def doit(self, deep=True)->Expr:...

    def get_ndarray(self, x: Dict[Variable, np.ndarray], **kwargs)->np.ndarray:...

    def ndarray(self, varorder: list[Variable], grid: grids.Grid, acc=1, fd='central')->np.ndarray:...

    def body(self)->Expr:...
    
    def coef(self)->Expr:...
    
    def addargs(self)->tuple[Expr,...]:...

    def mulargs(self)->tuple[Expr,...]:...

    def powargs(self)->tuple[Expr, Expr]:...

    def neg(self)->Expr:...
    
    def replace(self, items: Dict[Expr, Expr])->Expr:...

    def subs(self, vals: Dict[Expr, Expr])->Expr:...

    def diff(self, var: Variable, order=1)->Expr:...

    def eval(self)->Expr:...
    
    def get_grids(self, var: Variable)->tuple[grids.Grid1D,...]:...
    
    @cached_property
    def variables(self)->tuple[Variable,...]:...
    
    def expand(self)->Expr:...

    def init(self, *args, simplify=True)->Expr:...

    def array(self, varorder: list[Variable], grid: grids.Grid, acc=1, fd='central')->np.ndarray:...
    
    def integral(self, varorder: list[Variable], grid: grids.Grid, acc=1, fd='central')->float:...

    def dummify(self, varorder=None, grid: grids.Grid=None, acc=1, fd='central')->DummyScalarField:...

    def plot(self, varorder: list[Variable], grid: grids.Grid, acc=1, fd='central', ax=None, **kwargs):...
    
    def animate(self, var: Variable, varorder: list[Variable], duration: float, save: str, grid: grids.Grid, display = True, **kwargs):...

    def write_as_ode(self, lang: str, lib: str, symbols: list[Variable], tvar: str, qvar: str, args: tuple[Variable, ...]=())->str:...

    def lambdify(self, varnames, lib='math')->Callable[..., np.ndarray|float]:...

    def powsimp(self)->Expr:...

    def trigexpand(self)->Expr:...
    
    def split_trig(self)->Expr:...

    def split_int_trigcoefs(self)->Expr:...


class Add(Expr, _Add):

    def __new__(cls, *args, simplify=True)->Expr:...


class Mul(Expr, _Mul):

    def __new__(cls, *args, simplify=True)->Expr:...


class Pow(Expr, _Pow):

    def __new__(cls, base, power, simplify=True)->Expr:...


class Float(Expr, _Float):...


class Rational(Expr, _Rational):

    def __new__(cls, m: int, n: int)->Expr:...


class Integer(Expr, _Integer):

    def __new__(cls, m)->Expr:...


class Special(Expr, _Special):...


class Complex(Expr, _Complex):...


class Variable(Expr, _Symbol):...


class Subs(Expr, _Subs):

    def __new__(cls, expr: Expr, vals: Dict[Variable, Expr], simplify=True)->Expr:...

    @property
    def expr(self)->Expr:...

    @property
    def vals(self)->Dict[Variable, Expr]:...


class Derivative(Expr, _Derivative):

    def __new__(cls, f: Expr, *vars: Variable, simplify=True)->Expr:...

    @property
    def f(self)->Expr:...
    
    @cached_property
    def symbols(self)->tuple[Variable, ...]:...

    @cached_property
    def diffcount(self)->Dict[Variable, int]:...


class Integral(Expr, _Integral):

    def __new__(cls, f: Expr, var: Variable, x0, simplify=True)->Expr:...

    @property
    def f(self)->Expr:...
    
    @property
    def symbol(self)->Variable:...


class ScalarField(Expr, _ScalarField):

    def __init__(self, ndarray: np.ndarray, grid: grids.Grid, name: str, *vars: Variable):...

    @property
    def _variables(self)->tuple[Variable,...]:...


class DummyScalarField(Expr, _DummyScalarField):...


class Piecewise(Expr, _Piecewise):

    def __new__(cls, *cases: tuple[Expr, Boolean], default: _Expr, simplify=True)->Expr:...

    @property
    def default(self)->Expr:...

    @property
    def cases(self)->tuple[tuple[Expr, Boolean],...]:...


class Singleton(_Singleton):
    One: Integer
    Zero: Integer
    I: Complex
    pi: Special


class Any(Expr, _Any):...


def binomial(n, k)->Integer:...

def powsimp(expr: Expr)->Expr:...

def trigexpand(expr: Expr)->Expr:...

def write_as_common_denominator(expr: Expr):...

def split_trig(expr: Expr):...

def variables(arg: str)->tuple[Variable, ...]:...


from ..conditional import Boolean