from __future__ import annotations
from ..symcore import _Add, _Mul, _Pow, _Float, _Integer, _Rational, _Special, _Complex, _Symbol, _Subs, _Derivative, _Integral, _ScalarField, _Singleton, _Any, _DummyScalarField, _Piecewise, _Expr
from ..mathbase import *
from typing import Dict
import itertools
import math


class Expr(_Expr):

    is_symexpr = True
    args: tuple[Expr,...]

    def __add__(self, other)->Expr:
        return Add(self, self._asexpr(other))
    
    def __sub__(self, other)->Expr:
        return Add(self, -self._asexpr(other))
    
    def __mul__(self, other)->Expr:
        return Mul(self, self._asexpr(other))
    
    def __truediv__(self, other)->Expr:
        return Mul(self, self._asexpr(other)**-1)
    
    def __pow__(self, other)->Expr:
        return Pow(self, self._asexpr(other))
    
    def __neg__(self)->Expr:
        return -1*self

    @classmethod
    def _asexpr(cls, arg)->Expr:
        if isinstance(arg, Expr):
            return arg
        elif type(arg) is int:
            return Integer(arg)
        elif type(arg) is float:
            if arg == int(arg):
                return Integer(int(arg))
            else:
                return Float(arg)
        elif type(arg) is complex:
            return Complex(arg.real, arg.imag)
        else:
            raise ValueError(f'The object {arg} of type {arg.__class__} is not compatible with the Expr class')

    @classmethod
    def _add(cls, *args, simplify=True)->Expr:
        return Add(*args, simplify=simplify)

    @classmethod
    def _mul(cls, *args, simplify=True)->Expr:
        return Mul(*args, simplify=simplify)

    @classmethod
    def _pow(cls, base, power, simplify=True)->Expr:
        return Pow(base, power, simplify=simplify)
    
    @classmethod
    def _sin(cls, arg, simplify=True):
        return sin(arg, simplify=simplify)
    
    @classmethod
    def _cos(cls, arg, simplify=True):
        return cos(arg, simplify=simplify)
    
    @classmethod
    def _exp(cls, arg, simplify=True):
        return exp(arg, simplify=simplify)
    
    @classmethod
    def _log(cls, arg, simplify=True):
        return log(arg, simplify=simplify)
    
    @classmethod
    def _tan(cls, arg, simplify=True):
        return tan(arg, simplify=simplify)

    @classmethod
    def _abs(cls, arg, simplify=True):
        return Abs(arg, simplify=simplify)

    @classmethod
    def _rat(cls, m: int, n: int):
        return Rational(m, n)
    
    @classmethod
    def _derivative(cls, f, *vars, simplify=True) -> Expr:
        return Derivative(f, *vars, simplify=simplify)
    
    @classmethod
    def _subs(cls, expr, vals, simplify=True):
        return Subs(expr, vals, simplify=simplify)        

    @classmethod
    def _dummy(cls, arr, grid, *vars):
        return DummyScalarField(arr, grid, *vars)

    def lambdify(self, varnames, lib='math'):
        return lambdify(self, symbols=varnames, lib=lib)

    def powsimp(self):
        return powsimp(self)

    def trigexpand(self):
        return trigexpand(self)
    
    def split_trig(self):
        return split_trig(self)

    def split_int_trigcoefs(self):
        return split_int_trigcoefs(self)


class Subs(Expr, _Subs):...


class Add(Expr, _Add):...


class Mul(Expr, _Mul):...


class Pow(Expr, _Pow):...
        

class Float(Expr, _Float):...


class Rational(Expr, _Rational):...


class Integer(Expr, _Integer):...


class Special(Expr, _Special):...


class Complex(Expr, _Complex):...


class Variable(Expr, _Symbol):...


class Derivative(Expr, _Derivative):...


class Integral(Expr, _Integral):...


class ScalarField(Expr, _ScalarField):...


class DummyScalarField(Expr, _DummyScalarField):...


class Piecewise(Expr, _Piecewise):...


class Singletons(_Singleton):

    One = Integer(1)
    Zero = Integer(0)
    I = Complex(0, 1)
    pi = Special('pi', 3.141592653589793)


class Any(Expr, _Any):...

def sqrt(x):
    return x**Rational(1, 2)

def binomial(n, k)->Integer:
    return Rational(math.factorial(n), math.factorial(k)*math.factorial(n-k))



def powsimp(expr: Expr)->Expr:
    
    assert isinstance(expr, Expr)

    if isinstance(expr, Mul):
        base: Dict[Expr, list[Expr]] = {} #base = {power1: [base1, base2,...], power2: [base3, base4,...],...}
        nonpow_base = []

        for item in expr.args:
            item = item.powsimp()
            body, power = item.powargs()
            if power == 1:
                nonpow_base.append(item)
            elif power in base:
                base[power].append(body)
            else:
                base[power] = [body]
        
        muls = [expr._pow(expr._mul(*base[power]), power) for power in base]
        return expr._mul(*nonpow_base, *muls, simplify=False)
    elif isinstance(expr, Pow):
        return expr.init(expr.base, expr.power)
    elif isinstance(expr, Node):
        return expr.init(*[arg.powsimp() for arg in expr.args], simplify=False)
    else:
        return expr


def trigexpand(expr: Expr)->Expr:
    assert isinstance(expr, Expr)
    if not isinstance(expr, Operation):
        return expr
    else:
        expr = expr.expand()
        if isinstance(expr, Add):
            return Add(*[trigexpand(arg) for arg in expr.args])
        elif isinstance(expr, Mul):
            trigs = []
            pows = []
            other = []
            for arg in expr.args:
                base, power = arg.powargs()
                if isinstance(base, (cos, sin)) and isinstance(power, Integer):
                    trigs.append(base)
                    pows.append(power.value)
                    continue
                other.append(arg)
            if not trigs or (len(trigs)==1 and sum(pows)<2):
                return expr
            else:
                return Mul(_expand_sin_cos(trigs, pows), *other).expand()
        else:
            expr: Pow
            base, power = expr.args
            if isinstance(base, (cos, sin)) and isinstance(power, Integer):
                if power.value > 1:
                    return _expand_sin_cos([base], [power.value])
            return expr

def write_as_common_denominator(expr: Expr):
    expr = expr.expand()
    if not isinstance(expr, _Add):
        return expr
    nums, dens = [], []
    for arg in expr.args:
        if isinstance(arg, _Mul):
            nums.append(arg.numerator)
            dens.append(arg.denominator)
        else:
            nums.append(arg)
            dens.append(S.One)
    Num = []
    for i in range(len(nums)):
        Num.append(Mul(nums[i], *dens[:i], *dens[i+1:]))
    return Add(*Num).expand()/Mul(*dens)

def _s(trig):
    if isinstance(trig, sin):
        return 1
    else:
        return 0
        
def _expand_sin_cos(trigterms: list[sin|cos], powers:list[int])->Expr:        
    powers = [n.value if isinstance(n, Integer) else n for n in powers]
    biniter = [[binomial(n, k) for k in range(n+1)] for n in powers]
    phaseiter = [[(2*k-n)*x.Arg+_s(x)*(k+Rational(n,2))*pi for k in range(n+1)] for x, n in zip(trigterms, powers)]
    biniter = itertools.product(*biniter)
    phaseiter = itertools.product(*phaseiter)
    adds = []
    for coef, phase in zip(biniter, phaseiter):
        adds.append(Mul(*coef, cos(Add(*phase).expand())))
    return Rational(1, 2**sum(powers)) * Add(*adds)


def split_trig(expr: Expr):
    if isinstance(expr, Operation):
        return expr.init(*[split_trig(arg) for arg in expr.args])
    elif isinstance(expr, (sin, cos)):
        if isinstance(expr.Arg, Add):
            return split_trig(expr.addrule(expr.Arg))
        else:
            return expr
    else:
        return expr


def split_int_trigcoefs(expr: Expr):
    if isinstance(expr, Operation):
        return expr.init(*[split_int_trigcoefs(arg) for arg in expr.args])
    elif isinstance(expr, (sin, cos)):
        if isinstance(expr.Arg, Mul):
            if isinstance(expr.Arg.args[0], Integer):
                return split_int_trigcoefs(expr.split_intcoef(expr.Arg))
        return expr
    else:
        return expr

def variables(arg: str):
    x = arg.split(', ')
    y = []
    for i in x:
        if i != '':
            y.append(i)
    n = len(y)
    symbols: list[Variable] = []
    for i in range(n):
        symbols.append(Variable(y[i]))
    return tuple(symbols)

Expr.S = Singletons()
S = Expr.S
pi = Expr.S.pi

from .symmath import sin, cos, exp, log, tan, Abs, Real, Imag
from .pylambda import lambdify