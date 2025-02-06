from __future__ import annotations
from ..symcore import _Add, _Mul, _Pow, _Number, _Float, _Integer, _Rational, _Special, _Complex, _Symbol, _Subs, _Derivative, _ScalarField, _Singleton, _Any, _DummyScalarField, _Expr, _Function, Atom
from ..mathbase import *
from ..mathbase import _Mathfunc
from typing import Dict
from ...toolkit import tools
import scipy.sparse as sp
from ...findiffs import grids
from functools import cached_property


class Operator(_Expr):

    is_operator = True
    args: tuple[Operator,...]

    def __add__(self, other)->Operator:
        return AddOp(self, self._asexpr(other))
    
    def __sub__(self, other)->Operator:
        return AddOp(self, -self._asexpr(other))
    
    def __mul__(self, other)->Operator:
        return MulOp(self, self._asexpr(other))
    
    def __truediv__(self, other)->Operator:
        return MulOp(self, self._asexpr(other)**-1)
    
    def __pow__(self, other)->Operator:
        return PowOp(self, self._asexpr(other))
    
    def __neg__(self)->Operator:
        return -1*self

    @classmethod
    def _asexpr(cls, arg)->Operator:
        if isinstance(arg, Operator):
            return arg
        elif type(arg) is int:
            return IntegerOp(arg)
        elif type(arg) is float:
            if arg == int(arg):
                return IntegerOp(int(arg))
            else:
                return FloatOp(arg)
        elif type(arg) is complex:
            return ComplexOp(arg.real, arg.imag)
        else:
            raise ValueError(f'The object {arg} of type {arg.__class__} is not compatible with the Operator class')

    @classmethod
    def _add(cls, *args, simplify=True)->Operator:
        return AddOp(*args, simplify=simplify)

    @classmethod
    def _mul(cls, *args, simplify=True)->Operator:
        return MulOp(*args, simplify=simplify)

    @classmethod
    def _pow(cls, base, power, simplify=True)->Operator:
        return PowOp(base, power, simplify=simplify)
    
    @classmethod
    def _sin(cls, arg, simplify=True):
        return Sin(arg, simplify=simplify)
    
    @classmethod
    def _cos(cls, arg, simplify=True):
        return Cos(arg, simplify=simplify)
    
    @classmethod
    def _exp(cls, arg, simplify=True):
        return Exp(arg, simplify=simplify)
    
    @classmethod
    def _log(cls, arg, simplify=True):
        return Log(arg, simplify=simplify)
    
    @classmethod
    def _tan(cls, arg, simplify=True):
        return Tan(arg, simplify=simplify)

    @classmethod
    def _abs(cls, arg, simplify=True):
        return AbsOp(arg, simplify=simplify)

    @classmethod
    def _rat(cls, m: int, n: int):
        return RationalOp(m, n)
    
    @classmethod
    def _derivative(cls, f, *vars, simplify=True) -> Operator:
        return DerivOp(f, *vars, simplify=simplify)
    
    @classmethod
    def _subs(cls, expr, vals, simplify=True):
        return SubsOp(expr, vals, simplify=simplify)        

    @classmethod
    def _dummy(cls, arr, grid, *vars):
        return DummyScalarFieldOp(arr, grid, *vars)
    
    def has_diff(self)->bool:
        return self.contains_type(Diff)
    
    @property
    def variables(self)->tuple[VariableOp, ...]:
        v: tuple[VariableOp, ...] = super().variables
        return tools.sort(v, [x.axis for x in v])[0]

    @property
    def hasdiff_wrt(self)->tuple[VariableOp,...]:
        res = ()
        for item in self.deepsearch():
            if isinstance(item, Diff):
                if item.symbol not in res:
                    res += (item.symbol,)
        return res

    @cached_property
    def isNumber(self):
        if self.has_diff():
            return False
        else:
            return super().isNumber
    
    def is_const_wrt(self, x: VariableOp):
        if self.has_diff():
            return False
        else:
            return super().is_const_wrt(x)
    
    def trivially_commutes_with(self, other: Operator)->bool:
        if not self.has_diff() and not other.has_diff():
            return True
        
        if all([d_dxi not in other.variables for d_dxi in self.hasdiff_wrt]) and all([d_dxi not in self.variables for d_dxi in other.hasdiff_wrt]):
            return True
        
        if self == other:
            return True
        
        return False

    def commutes_with(self, other: Operator)->bool:
        if not self.trivially_commutes_with(other):
            return (self*other-other*self).Expand().apply(TestFunction('f', *list(set(self.variables+other.variables)))) == 0
        else:
            return True
    
    def Expand(self)->Operator:
        res: Operator
        if isinstance(self, AddOp):
            res = AddOp(*[arg.Expand() for arg in self.args])
        elif isinstance(self, (MulOp, PowOp)):
            if any([isinstance(arg, AddOp) for arg in self.mulargs()]):
                oper: Operator = self.expand()
                res = oper.Expand()
            else:
                args: list[Operator] = list(self.mulargs())
                if len(args) == 1:
                    return self
                res = self
                for i in range(len(args)-2, -1, -1):
                    if isinstance(args[i], Diff) and not args[i+1].has_diff():
                        oper = args[i].apply(args[i+1]) + args[i+1]*args[i]
                        args.pop(i+1)
                        args[i] = oper
                        res = MulOp(*args)
                        res = res.Expand()
                        break
        else:
            res = self
        return res.expand()

    def apply(self, other)->Operator:
        other = self._asexpr(other)
        if isinstance(self, Diff):
            return other.diff(*self.Args)
        elif isinstance(self, AddOp):
            return AddOp(*[arg.apply(other) for arg in self.args])
        elif isinstance(self, (MulOp, PowOp)):
            args = self.mulargs()
            if len(args) == 1:
                return self*other
            else:
                return _apply_seq(args, other)
        else:
            return self*other        
        
    def difforder(self, variable: VariableOp):
        raise NotImplementedError('')

    def adjoint(self, weight: VariableOp = 1):
        weight = self._asexpr(weight)
        if weight.has_diff():
            raise ValueError('Weight must not contain any differential operators')
        if isinstance(self, Diff):
            return (-1)**self.order * weight * self * weight
        elif isinstance(self, AddOp):
            return AddOp(*[arg.adjoint(weight) for arg in self.args])
        elif isinstance(self, MulOp):
            n = len(self.args)
            return MulOp(*[self.args[i].adjoint(weight) for i in range(n-1, -1, -1)])
        elif isinstance(self, PowOp):
            if self.has_diff():
                return self.base.adjoint(weight) ** self.power
            else:
                return self
        elif isinstance(self, ComplexOp):
            return ComplexOp(self.value.conjugate())
        else:
            return self
    
    def matrix(self, grid: grids.Grid, acc=1, fd='central')->sp.csr_matrix:
        if isinstance(self, Diff):
            return grid.partialdiff_matrix(order=self.order, axis=self.symbol.axis, acc=acc, fd=fd)
        elif isinstance(self, AddOp):
            return sum([arg.matrix(grid, acc, fd) for arg in self.args], start=grid.empty_matrix())
        elif isinstance(self, MulOp):
            return tools.multi_dot_product(*[f.matrix(grid, acc, fd) for f in self.args])
        elif isinstance(self, PowOp):
            if self.has_diff():
                return tools.multi_dot_product(*(self.power.value*[self.base.matrix(grid, acc, fd)]))
            else:
                return tools.as_sparse_diag(self.array(grid, acc, fd))
        elif isinstance(self, _Number):
            return self.value * sp.identity(grid.n, format='csr')
        else:
            return tools.as_sparse_diag(self.array(grid, acc=acc, fd=fd))
        
    def separate(self)->tuple[VariableOp]:
        if len(self.variables) == 0:
            return self,
        else:
            x = self.variables[0]
            res[x] = 0
        args = self.expand().addargs()
        res: Dict[VariableOp, Operator] = {}
        for arg in args:
            if len(arg.variables) > 1:
                return self,
            elif len(arg.variables) == 1:
                v = arg.variables[0]
                if v not in res:
                    res[v] = arg
                else:
                    res[v] += arg
            else:
                res[x] += arg

class SubsOp(Operator, _Subs):...


class AddOp(Operator, _Add):...


class MulOp(Operator, _Mul):

    is_commutative = False

    @classmethod
    def simplify(cls, *seq: Operator)->tuple[Operator,...]:

        flatseq: tuple[_Expr,...] = ()
        for arg in seq:
            if arg == 0:
                return cls.S.Zero,
            elif arg.sgn == -1 and isinstance(arg, _Add):
                flatseq += (cls._asexpr(-1),) + cls._add(*[-f for f in arg.args], simplify=False).mulargs()
            else:
                flatseq += arg.mulargs()

        _float = 1.
        rational = cls.S.One


        newseq: list[Operator] = []
        for arg in flatseq:
            if isinstance(arg, _Float):
                _float *= arg.value
            elif isinstance(arg, _Rational):
                rational = cls._rat(rational.n*arg.n, rational.d*arg.d)
            else:
                newseq.append(arg)
        
        simp_seq: list[Operator] = []
        for arg in newseq:
            simp_seq.append(arg)
            j = len(simp_seq)-2
            while arg.trivially_commutes_with(simp_seq[j]) and j >= 0:
                base1, p1 = simp_seq[j].powargs()
                base2, p2 = arg.powargs()
                if base1 == base2:
                    power = p1 + p2
                    if power != 0:
                        simp_seq[j] = base1**power
                    else:
                        simp_seq.pop(j)
                    simp_seq.pop(-1)
                    break
                elif j == 0:
                    break
                else:
                    j -= 1
        
        res = []
        if _float != 1:
            res.append(cls._asexpr(_float))
        if rational != 1:
            res.append(rational)
        res = res + simp_seq

        if not res:
            return cls.S.One,
        else:
            return tuple(res)
        
    def _equals(self, other: Operator):
        if not self.has_diff() and not other.has_diff():
            return tools.contain_same_elements(self.args, other.args)
        else:
            return Operation._equals(self, other)


class PowOp(Operator, _Pow):

    base: Operator
    power: Operator

    def __new__(cls, base, power, simplify=True) -> _Expr:
        base, power = cls._asexpr(base), cls._asexpr(power)
        if power.has_diff():
            raise ValueError('Cannot raise an expression to an exponent that is a differential operator')
        elif base.has_diff() and power.isNegInt:
            raise ValueError('Cannot raise a differential operator to a power that is not a positive integer')
        else:
            return super().__new__(cls, base, power)

    def _diff(self, var):
        if self.has_diff():
            return Diff(var)*self
        else:
            return _Pow._diff(self, var)

    def mulargs(self):
        if self.has_diff():
            power: IntegerOp = self.power
            return power.value * self.base.mulargs()
        else:
            args = tuple([arg**self.power for arg in self.base.mulargs()])
            if len(args) == 1 and isinstance(self.base, AddOp) and self.power.isPosInt:
                return self.power.value * [self.base]
            else:
                return args
        

class FloatOp(Operator, _Float):...


class RationalOp(Operator, _Rational):...


class IntegerOp(Operator, _Integer):...


class SpecialOp(Operator, _Special):...


class ComplexOp(Operator, _Complex):...


class VariableOp(Operator, _Symbol):

    def __init__(self, name: str, axis: int):
        self.Args = (name, axis)

    @property
    def axis(self)->int:
        return self.Args[1]


class DerivOp(Operator, _Derivative):

    def __new__(cls, f: Operator, *vars: VariableOp, simplify=True) -> Operator:
        return super().__new__(cls, f, *vars, simplify=simplify)


class Diff(Operator, Atom):

    def __new__(cls, var: VariableOp, order=1):
        if isinstance(order, IntegerOp):
            order = order.value
        if not isinstance(var, VariableOp):
            raise ValueError(f'var must be of an instance of Variable, not of type {type(var)}')
        elif not isinstance(order, int) or order < 0:
            raise ValueError(f'The Diff order can only be integer and positive')
        if order == 0:
            return cls.S.One
        else:
            obj = super().__new__(cls)
            obj.Args = (var, order)
            return obj
    
    @property
    def symbol(self)->VariableOp:
        return self.Args[0]
    
    @property
    def order(self)->int:
        return self.Args[1]
    
    def mulargs(self):
        return self.order*(Diff(self.symbol),)
        
    def _diff(self, var: VariableOp):
        return Diff(var)*self

    def repr(self, lib: str):
        if lib != '':
            raise NotImplementedError('Differential operators do not support repr()')
        else:
            if self.order == 1:
                return f'Diff({self.symbol})'
            else:
                return f'Diff({self.symbol}, {self.order})'
            
    def powargs(self) -> tuple[Operator, Operator]:
        return Diff(self.symbol), self.order

    def raiseto(self, power):
        return Diff(self.symbol, self.order*power)
    
    def get_ndarray(self, x, **kwargs):
        raise NotImplementedError('ndarray not defined for differential operators')


class ScalarFieldOp(Operator, _ScalarField):...


class DummyScalarFieldOp(Operator, _DummyScalarField):...


class TestFunction(Operator, _Function):

    def __init__(self, name: str, *variables: VariableOp):
        self.Args = (name, *variables)

    @property
    def name(self)->str:
        return self.Args[0]
    
    @property
    def _variables(self)->tuple[VariableOp,...]:
        return self.Args[1:]


class SingletonsOp(_Singleton):

    One = IntegerOp(1)
    Zero = IntegerOp(0)
    I = ComplexOp(0, 1)
    pi = SpecialOp('pi', 3.141592653589793)


class AnyOp(Operator, _Any):...

def _apply_seq(seq: list[Operator], other: Operator):
    res = other
    for i in range(len(seq)-1, -1, -1):
        res = seq[i].apply(res)
    return res

Operator.S = SingletonsOp()



from .opermath import Sin, Cos, Exp, Log, Tan, AbsOp

