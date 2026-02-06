from __future__ import annotations
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import scipy.sparse as sp
from ..findiffs import grids
from ..findiffs.grids import InterpedArray
from ..toolkit import tools
from ..toolkit.plotting import plot, animate
from typing import Type, Dict
import itertools
from functools import cached_property
import math
import uuid



class Expr:

    _args: tuple #the only object attribute
    _priority: int #class attribute
    _repr_priority = 3


    def __new__(cls, *args):
        obj = object.__new__(cls)
        obj._args = args
        return obj

    def __add__(self, other)->Expr:
        return Add(self, asexpr(other))
    
    def __sub__(self, other)->Expr:
        return Add(self, -asexpr(other))
    
    def __mul__(self, other)->Expr:
        return Mul(self, asexpr(other))
    
    def __truediv__(self, other)->Expr:
        return Mul(self, asexpr(other)**-1)
    
    def __pow__(self, other)->Expr:
        return Pow(self, asexpr(other))
    
    def __neg__(self)->Expr:
        return -1*self
    
    def __radd__(self, other)->Expr:
        return asexpr(other) + self
    
    def __rsub__(self, other)->Expr:
        return asexpr(other) - self
    
    def __rmul__(self, other)->Expr:
        return asexpr(other) * self
    
    def __rtruediv__(self, other)->Expr:
        return asexpr(other) / self
    
    def __rpow__(self, other)->Expr:
        return asexpr(other) ** self
    
    def __abs__(self)->Expr:
        return Abs(self)
    
    def __gt__(self, other)->Expr:
        return Gt(self, other)
    
    def __lt__(self, other)->Expr:
        return Lt(self, other)
    
    def __ge__(self, other)->Expr:
        return Ge(self, other)
    
    def __le__(self, other)->Expr:
        return Le(self, other)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return self.repr(lib='')
    
    def __eq__(self, other):
        other = asexpr(other)
        if other is self:
            return True
        elif other.__class__ == self.__class__:
            return self._args == other._args
        else:
            return False
    
    def __hash__(self):
        return hash((self.__class__,) + self._hashable_content)

    @property
    def repr_priority(self):
        return self.__class__._repr_priority

    @cached_property
    def is_operator(self)->bool:
        return self.contains_type(Diff)

    @property
    def args(self):
        return self._args
    
    @property
    def branches(self):
        return tuple([arg for arg in self._args if isinstance(arg, Expr)])
    
    @property
    def atoms(self)->tuple[Atom,...]:
        '''
        performs a full search in all branches, and reaches to the outer edges where the Atoms lie
        '''
        return self.deepsearch()
    
    @cached_property
    def symbols(self)->tuple[Symbol, ...]:
        res = ()
        for x in self.atoms:
            if isinstance(x, Symbol) and x not in res:
                res += (x,)
        return res
    
    @cached_property
    def isNumber(self):
        return len(self.variables) == 0 and not self.is_operator

    @property
    def isRealNumber(self):
        n = self.eval()
        if isinstance(n, Number):
            return not isinstance(n, Complex)
        else:
            return False
        
    @property
    def isPosInt(self):
        if isinstance(self, Integer):
            if self.value > 0:
                return True
        return False

    @property
    def isNegInt(self):
        if isinstance(self, Integer):
            if self.value < 0:
                return True
        return False
    
    @cached_property
    def sgn(self):
        return 1
    
    def _remake_branches(self, filter: Type[Expr], attr: str, *args, **kwargs):
        '''
        same as init, but only targets the filtered objects in args,
        and replaces them with arg.attr(*args, **kwargs)
        '''
        newargs = list(self._args)
        for i in range(len(self._args)):
            arg = newargs[i]
            if isinstance(arg, filter):
                newargs[i] = getattr(arg, attr)(*args, **kwargs)
        return self.init(*newargs, simplify=True)
    
    def _replace(self, items: dict[Expr, Expr]):
        '''
        different from .replace()
        '''
        if self in items:
            return items[self]
        else:
            return self._remake_branches(Expr, '_replace', items)

    def diff(self, var: Symbol, order=1)->Expr:
        res = Derivative(self, *(order*[var]))
        if isinstance(res, Derivative):
            return res.doit(deep=False)
        else:
            return res
        
    def deepsearch(self, mask: Type[Expr] = None)->tuple[Expr,...]:

        if mask is not None:
            msk = mask
        else:
            msk = Atom

        if isinstance(self, msk):
            return (self,) if isinstance(self, msk) else ()

        args: tuple[Expr,...] = ()
        for arg in self.branches:
            if isinstance(arg, msk):
                args += (arg,)
            else:
                args += arg.deepsearch(msk)
        return args

    def contains_type(self, cls: Type):
        '''
        goes through al branches
        '''
        if isinstance(self, cls):
            return True
        else:
            for arg in self.branches:
                if arg.contains_type(cls):
                    return True
            return False
    
    def ndarray(self, symbols: tuple[Symbol, ...], grid: grids.Grid, acc=1, fd='central')->np.ndarray:
        return self.get_ndarray({x: arr for x, arr in zip(symbols, grid.x)}, acc=acc, fd=fd)
    
    def array(self, symbols: tuple[Symbol, ...], grid: grids.Grid, acc=1, fd='central'):
        return self.ndarray(symbols, grid, acc, fd).flatten(order='F')
    
    def varsub(self, data: dict[Symbol, Symbol]):
        for (x, v) in data.items():
            if not isinstance(x, Symbol) or not isinstance(v, Symbol):
                raise ValueError('')
        return self._replace(data)
    
    def is_const_wrt(self, x: Symbol):
        return x not in self.variables
    
    def lambdify(self, *symbols: Symbol, lib = "math", **kwargs):
        return lambdify(self, lib, *symbols, **kwargs)

    def powsimp(self):
        return powsimp(self)

    def trigexpand(self):
        return trigexpand(self)
    
    def split_trig(self):
        return split_trig(self)

    def split_int_trigcoefs(self):
        return split_int_trigcoefs(self)
    
    def get_grids(self, var: Symbol)->tuple[grids.Grid1D]:
        gs: list[grids.Grid1D] = []
        for item in self.atoms:
            if hasattr(item, 'grid') and var in item.variables:
                g: grids.Grid = getattr(item, 'grid')
                if g not in gs:
                    gs.append(g[item.variables.index(var)])
        return tuple(gs)
    
    def subs(self, vals: Dict[Symbol, Expr])->Expr:
        res = Subs(self, vals)
        if isinstance(res, Subs):
            return res.doit(deep=False)
        else:
            return res

    def integral(self, symbols: tuple[Symbol, ...], grid: grids.Grid, acc=1, fd='central'):
        return tools.full_multidim_simpson(self.ndarray(symbols, grid, acc, fd), *grid.x)

    def dummify(self, symbols: tuple[Symbol, ...], grid: grids.Grid=None, acc=1, fd='central'):
        if grid is None:
            gs = []
            for x in self.variables:
                g = self.get_grids(x)
                if len(g) == 1:
                    gs.append(g[0])
                else:
                    raise ValueError(f'No 1D-grid along the {x} variable inside the expression to dummify')
            grid = grids.NdGrid(*gs)
        
        return DummyScalarField(self.ndarray(symbols, grid, acc, fd), grid, *self.variables)

    def plot(self, symbols: tuple[Symbol,...], grid: grids.Grid, acc=1, fd='central', ax=None, **kwargs):
        return plot(self.ndarray(symbols, grid, acc, fd), grid, ax, **kwargs)
    
    def animate(self, var: Symbol, symbols: tuple[Symbol,...], duration: float, save: str, grid: grids.Grid, display = True, **kwargs):
        if 'fps' in kwargs:
            fps = kwargs['fps']
            del kwargs['fps']
            grid = grid.replace(axis, grids.Uniform1D(*grid.limits[axis], fps*duration, grid.periodic[axis]))
            f = self.ndarray(symbols, grid)
        else:
            f = self.ndarray(symbols, grid)
        axis = self.variables.index(var)
        return animate(str(var), f, duration, save, grid, axis, display, **kwargs)
        
    def _diff(self, var: Symbol)->Expr:
        raise NotImplementedError('')
    
    def get_ndarray(self, x: Dict[Symbol, np.ndarray], **kwargs)->np.ndarray:
        raise NotImplementedError('')
    
    def repr(self, lib = "")->str:
        raise NotImplementedError('')

    def lowlevel_repr(self, scalar_type="double")->str:
        raise NotImplementedError('')
    
    @classmethod
    def simplify(cls, *args: Expr)->tuple[Expr,...]:
        raise NotImplementedError('')








    def body(self):
        '''
        override
        '''
        return self
    
    def coef(self):
        '''
        override
        '''
        return S.One
    
    def addargs(self):
        '''
        override
        '''
        return self,

    def mulargs(self):
        '''
        override
        '''
        return self,

    def powargs(self)->tuple[Expr, Expr]:
        '''
        override
        '''
        return self, S.One

    def neg(self)->Expr:
        '''
        override
        '''
        return -self
    
    def expand(self):
        '''
        override
        '''
        return self
    
    def _repr_from(self, arg: str, oper: Type[Operation], lowlevel=False)->str:
        '''
        override
        '''
        base = self.repr(arg) if not lowlevel else self.lowlevel_repr(arg)
        if oper._repr_priority <= self.repr_priority:
            return base
        else:
            return f'({base})'
        






    

    @property
    def _hashable_content(self):
        '''
        override (e.g. ScalarField overrides this, and creates _HashableNdArray)
        '''
        return self._args

    @cached_property
    def variables(self)->tuple[Symbol, ...]:
        '''
        override
        '''
        res = ()
        for arg in self.branches:
            for x in arg.variables:
                if x not in res:
                    res += (x,)
        return res
        
    def init(self, *args, simplify=True):
        '''
        override

        This property is defined so that obj.init(*obj.args) is equivalent to obj
        '''
        if isinstance(self, Atom):
            return self.__class__(*args)
        else:
            return self.__class__(*args, simplify=simplify)
    
    def doit(self, deep=True)->Expr:
        '''
        override
        '''
        if isinstance(self, Atom):
            return self
        elif deep:
            return self._remake_branches(Expr, "doit", deep=True)
        else:
            return self.init(*[arg for arg in self._args], simplify=True)#simply reinstanciate simplified
    
    def eval(self)->Expr:
        '''
        override: _Number, _mathFunc, Operation, Subs
        '''
        return self._remake_branches(Expr, "eval")
        
    def _subs(self, vals: Dict[Symbol, Expr])->Expr:
        '''
        override. e.g. The Derivative or Diff override it, because the variable we differentiate wrt cannot change to another Expr.
        '''
        if self in vals:
            return vals[self]
        else:
            return self._remake_branches(Expr, "_subs", vals)

    def matrix(self, symbols: tuple[Symbol, ...], grid: grids.Grid, acc=1, fd='central')->sp.csr_matrix:
        '''
        override
        '''
        return tools.as_sparse_diag(self.array(symbols, grid, acc=acc, fd=fd))


    @cached_property
    def oper_symbols(self)->tuple[Symbol, ...]:
        res = ()
        for arg in self.branches:
            for x in arg.variables:
                if x not in res:
                    res += (x,)

        Dx: Diff
        for Dx in self.deepsearch(Diff):
            if Dx.symbol not in res:
                res += (Dx.symbol,)
        return tools.sort(res, [Hashable(x) for x in res])[0]


    @property
    def hasdiff_wrt(self)->tuple[Symbol,...]:
        res = ()
        items: tuple[Diff, ...] = self.deepsearch(Diff)
        for item in items:
                if item.symbol not in res:
                    res += (item.symbol,)
        return res
    
    def trivially_commutes_with(self, other: Expr)->bool:
        if not self.is_operator and not other.is_operator:
            return True
        
        if all([d_dxi not in other.symbols for d_dxi in self.hasdiff_wrt]) and all([d_dxi not in self.symbols for d_dxi in other.hasdiff_wrt]):
            return True
        
        if self == other:
            return True
        
        return False
    
    def commutes_with(self, other: Expr)->bool:
        if not self.trivially_commutes_with(other):
            return (self*other-other*self).Expand().apply(Function('f', *list(set(self.variables+other.variables)))) == 0
        else:
            return True

    def Expand(self)->Expr:
        res: Expr
        if isinstance(self, Add):
            res = Add(*[arg.Expand() for arg in self.args])
        elif isinstance(self, (Mul, Pow)):
            if any([isinstance(arg, Add) for arg in self.mulargs()]):
                oper = self.expand()
                res = oper.Expand()
            else:
                args: list[Expr] = list(self.mulargs())
                if len(args) == 1:
                    return self
                res = self
                for i in range(len(args)-2, -1, -1):
                    if isinstance(args[i], Diff) and not args[i+1].is_operator:
                        oper = args[i].apply(args[i+1]) + args[i+1]*args[i]
                        args.pop(i+1)
                        args[i] = oper
                        res = Mul(*args)
                        res = res.Expand()
                        break
        else:
            res = self
        return res.expand()

    def apply(self, other)->Expr:
        other = asexpr(other)
        if isinstance(self, Diff):
            return other.diff(self.symbol, self.order)
        elif isinstance(self, Add):
            return Add(*[arg.apply(other) for arg in self.args])
        elif isinstance(self, (Mul, Pow)):
            args = self.mulargs()
            if len(args) == 1:
                return self*other
            else:
                return _apply_seq(args, other)
        else:
            return self*other
        
    def adjoint(self, weight: Symbol = 1):
        weight = asexpr(weight)
        if weight.is_operator:
            raise ValueError('Weight must not contain any differential operators')
        if isinstance(self, Diff):
            return (-1)**self.order * weight * self * weight
        elif isinstance(self, Add):
            return Add(*[arg.adjoint(weight) for arg in self.args])
        elif isinstance(self, Mul):
            n = len(self.args)
            return Mul(*[self.args[i].adjoint(weight) for i in range(n-1, -1, -1)])
        elif isinstance(self, Pow):
            if self.is_operator:
                return self.base.adjoint(weight) ** self.power
            else:
                return self
        elif isinstance(self, Complex):
            return Complex(self.value.conjugate())
        else:
            return self
    
    def separate(self)->tuple[Symbol]:
        if len(self.variables) == 0:
            return self,
        else:
            x = self.variables[0]
            res[x] = 0
        args = self.expand().addargs()
        res: Dict[Symbol, Expr] = {}
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
            
        return tuple(res.values())



    


class Atom(Expr):
    
    @property
    def branches(self)->tuple[Expr, ...]:
        return ()
    
    @property
    def atoms(self)->Atom:
        return (self,)
    
    @classmethod
    def simplify(cls, *args):
        return cls(*args)


class Operation(Expr):

    binary: str
    torch_binary: str
    _args: tuple[Expr, ...]

    def __new__(cls, *args: Expr, simplify=True):
        if len(args) == 1:
            return args[0]
        if simplify:
            args = cls.simplify(*args)
            if len(args) == 1:
                return args[0]
        return Expr.__new__(cls, *args)
    
    @property
    def args(self)->tuple[Expr,...]:
        return self._args
    
    @property
    def is_commutative(self):
        return True
    
    @classmethod
    def matheval(cls, a, b):
        raise NotImplementedError('')
    
    def repr(self, lib = "", **kwargs)->str:
        if lib == 'torch' and kwargs.get('out', False):
            if len(self.args) == 2 and not (self._args[0].isNumber and self._args[1].isNumber):
                return f'torch.{self.torch_binary}({self._args[0].repr(lib, **kwargs)}, {self._args[1].repr(lib, **kwargs)}, out=out)'
            elif len(self.args) > 2:
                return self.__class__(self.__class__(*self.args[:2], simplify=False), *self.args[2:], simplify=False).repr(lib, **kwargs)

        return self.binary.join([f._repr_from(lib, self.__class__) for f in self._args])
    
    def lowlevel_repr(self, scalar_type="double"):
        return self.binary.join([f._repr_from(scalar_type, self.__class__, lowlevel=True) for f in self._args])
    
    def eval(self):
        args = [arg.eval() for arg in self.args]
        if isinstance(self, Pow):
            a, b = self.base.eval(), self.power.eval()
            if isinstance(a, Number) and isinstance(b, Number):
                return asexpr(a.value**b.value)
            else:
                return Pow(a, b)
        else:
            nums, rest = [], []
            for arg in args:
                if isinstance(arg, Number):
                    nums.append(arg.value)
                else:
                    rest.append(arg)
            return self.init(self.matheval(*nums), *rest)
        

class Add(Operation):

    _repr_priority = 0
    torch_binary = 'add'
    binary = '+'
    _priority = 0

    def __new__(cls, *args, simplify=True):
        args = tuple([asexpr(arg) for arg in args])
        args = sort_by_hash(*args)
        return Operation.__new__(cls, *args, simplify=simplify)

    @classmethod
    def simplify(cls, *seq: Expr):
        flatseq: tuple[Expr,...] = ()
        
        for arg in seq:
            flatseq += arg.addargs()
        
        _float = 0.
        rational = S.Zero

        coef: Dict[Expr, list[Expr]] = {}

        for arg in flatseq:
            if isinstance(arg, Float):
                _float += arg.value
            elif isinstance(arg, Rational):
                rational = Rational(rational.n*arg.d+arg.n*rational.d, rational.d*arg.d)
            else:
                base, c = arg.body(), arg.coef()
                if base in coef:
                    coef[base].append(c)
                else:
                    coef[base] = [c]

        res = []
        if _float != 0:
            res.append(asexpr(_float))
        if rational.n != 0:
            res.append(rational)
        for base in coef:
            c = Add(*coef[base])
            if c != 0:
                res.append(c*base)

        if not res:
            return S.Zero,
        else:
            return tuple(res)
        
    @cached_property
    def sgn(self):
        signs = [arg.sgn for arg in self._args]
        if signs.count(1) >= signs.count(-1):
            return 1
        else:
            return -1

    def _diff(self, var):
        return Add(*[arg._diff(var) for arg in self._args])

    def neg(self):
        return self.init(*[-arg for arg in self._args])

    def addargs(self):
        return self._args
    
    @classmethod
    def matheval(cls, *args: float)->float:
        return sum(args)
    
    def _remove_minus(self, func: str, _arg: str)->str:
        res = getattr(self._args[0], func)(_arg)
        for arg in self._args[1:]:
            r: str = getattr(arg, func)(_arg)
            if r.startswith('-'):
                res += ' - ' + r[1:]
            else:
                res += ' + ' + getattr(arg, func)(_arg)
        return res
    
    def repr(self, lib = "", **kwargs)->str:
        if lib == 'torch' and ((not (self._args[0].isNumber and self._args[1].isNumber)) or len(self._args) > 2) and kwargs.get('out', False):
            return super().repr(lib, **kwargs)
        return self._remove_minus("repr", lib)
    
    def lowlevel_repr(self, scalar_type="double"):
        return self._remove_minus("lowlevel_repr", scalar_type)
    
    def expand(self)->Expr:
        return Add(*[arg.expand() for arg in self._args])

    def get_ndarray(self, x, **kwargs):
        return np.sum([arg.get_ndarray(x, **kwargs) for arg in self._args], axis=0)
    
    def matrix(self, symbols: tuple[Symbol, ...], grid, acc=1, fd='central'):
        return sum([arg.matrix(symbols, grid, acc, fd) for arg in self.args], start=grid.empty_matrix())
    

class Mul(Operation):

    _repr_priority = 1
    torch_binary = 'mul'
    binary = '*'
    _priority = 1

    def __new__(cls, *args, simplify=True):
        args = tuple([asexpr(arg) for arg in args])
        if not any([arg.is_operator for arg in args]):
            args = sort_by_hash(*args)
        return Operation.__new__(cls, *args, simplify=simplify)
    
    @classmethod
    def simplify(cls, *seq: Expr):
        flatseq: tuple[Expr,...] = ()
        for arg in seq:
            if arg == 0:
                return S.Zero,
            elif arg.sgn == -1 and isinstance(arg, Add):
                flatseq += (asexpr(-1),) + Add(*[-f for f in arg._args], simplify=False).mulargs()
            else:
                flatseq += arg.mulargs()

        _float = 1.
        rational = S.One
        if any([arg.is_operator for arg in flatseq]):
            newseq: list[Expr] = []
            for arg in flatseq:
                if isinstance(arg, Float):
                    _float *= arg.value
                elif isinstance(arg, Rational):
                    rational = Rational(rational.n*arg.n, rational.d*arg.d)
                else:
                    newseq.append(arg)
            
            j = len(newseq) - 1
            while j > 0:
                k = j-1
                while newseq[j].trivially_commutes_with(newseq[k]) and k >= 0:
                    base1, p1 = newseq[j].powargs()
                    base2, p2 = newseq[k].powargs()
                    if base1 == base2:
                        power = p1+p2
                        newseq.pop(j)
                        if power != 0:
                            newseq[k] = base1**power
                        else:
                            newseq.pop(k)
                        j = len(newseq) - 1
                        k = j-1
                    else:
                        k -= 1
                j -= 1

            res = []
            if _float != 1:
                res.append(asexpr(_float))
            if rational != 1:
                res.append(rational)
            res = res + newseq
        else:
            pow: Dict[Expr, list[Expr]] = {}
            for arg in flatseq:
                if isinstance(arg, Float):
                    _float *= arg.value
                elif isinstance(arg, Rational):
                    rational = Rational(rational.n*arg.n, rational.d*arg.d)
                else:
                    base, p = arg.powargs()
                    if base in pow:
                        pow[base].append(p)
                    else:
                        pow[base] = [p]
        
            res = []
            if isinstance(rational, Integer) and _float != 1.:
                res.append(asexpr(rational.n*_float))
            else:
                if _float != 1:
                    res.append(asexpr(_float))
                if rational != 1:
                    res.append(rational)
            for base in pow:
                c = Add(*pow[base])
                if c == 0:
                    continue
                res.append(base**c)

        if not res:
            return S.One,
        else:
            return tuple(res)
        
    @property
    def is_commutative(self):
        return not self.is_operator
        
    @cached_property
    def numerator(self):
        res = []
        for arg in self._args:
            if isinstance(arg, Pow):
                if isinstance(arg.power, (Float, Integer)):
                    if arg.power.value < 0:
                        continue
            res.append(arg)
        if not res:
            return S.One
        else:
            return Mul(*res, simplify=False)

    @cached_property
    def denominator(self):
        res = []
        for arg in self._args:
            if isinstance(arg, Pow):
                if isinstance(arg.power, (Float, Integer)):
                    if arg.power.value < 0:
                        res.append(arg**-1)
        if not res:
            return S.One
        else:
            return Mul(*res, simplify=False)

    @cached_property
    def _coef(self):
        res = []
        for arg in self._args:
            if isinstance(arg, Number) and not isinstance(arg, Special):
                res.append(arg)
        if len(res) == len(self._args):
            return S.One
        else:
            return Mul(*res)

    @cached_property
    def _body(self):
        res = []
        for arg in self._args:
            if not isinstance(arg, Number) or isinstance(arg, Special):
                res.append(arg)
        if not res:
            return self
        else:
            return Mul(*res)

    @cached_property
    def sgn(self):
        return self._args[0].sgn
    
    def _diff(self, var):
        coef = Mul(*[arg for arg in self._args if arg.isNumber])
        body = [arg for arg in self._args if not arg.isNumber]
        toadd = []
        for i in range(len(body)):
            prod = body[:i]+[body[i]._diff(var)]+body[i+1:]
            toadd.append(Mul(*prod))
        return coef*Add(*toadd)

    def body(self):
        return self._body
    
    def coef(self):
        return self._coef

    @classmethod
    def matheval(cls, *args: float)->float:
        return math.prod(args)
    
    def _remove_one(self, func: str, arg: str):
        lowlevel = (func == "lowlevel_repr")
        num, den = self.numerator, self.denominator
        if den != 1:
            if den.repr_priority == self.repr_priority:
                s = f'{num._repr_from(arg, Mul, lowlevel)}/({den._repr_from(arg, Mul, lowlevel)})'
            else:
                s = f'{num._repr_from(arg, Mul, lowlevel)}/{den._repr_from(arg, Mul, lowlevel)}'
        else:
            s = getattr(Operation, func)(self, arg)
        if s.startswith('-1*'):
            s = '-'+s[3:]
        elif  s.startswith('-1.*'):
            s = '-'+s[4:]
        return s

    def repr(self, lib="", **kwargs):
        if lib == 'torch' and ((not (self._args[0].isNumber and self._args[1].isNumber)) or len(self._args) > 2) and kwargs.get('out', False):
            return super().repr(lib, **kwargs)
        return self._remove_one("repr", lib)
    
    def lowlevel_repr(self, scalar_type="double"):
        return self._remove_one("lowlevel_repr", scalar_type)

    def mulargs(self):
        return sum([arg.mulargs() for arg in self._args], start=())

    def expand(self)->Expr:
        if not self.contains_type(Add):
            return self
        args = [arg.expand() for arg in self._args]
        muls = itertools.product(*[arg._args if isinstance(arg, Add) else (arg,) for arg in args])
        s = [Mul(*m) for m in muls]
        return Add(*s)

    def get_ndarray(self, x, **kwargs):
        return np.prod([arg.get_ndarray(x, **kwargs) for arg in self._args], axis=0)
    
    def matrix(self, symbols: tuple[Symbol, ...], grid, acc=1, fd='central'):
        return tools.multi_dot_product(*[f.matrix(symbols, grid, acc, fd) for f in self.args])


class Pow(Operation):

    _repr_priority = 2
    torch_binary = 'pow'
    binary = '**'
    _priority = 2

    def __new__(cls, base, power, simplify=True):
        base = asexpr(base)
        power = asexpr(power)
        if power.is_operator:
            raise ValueError('Cannot raise an expression to an exponent that is a differential operator')
        elif base.is_operator and power.isNegInt:
            raise ValueError('Cannot raise a differential operator to a power that is not a positive integer')
        return Operation.__new__(cls, base, power, simplify=simplify)
    
    @classmethod
    def simplify(cls, a: Expr, b: Expr):
        abase, apower = a.powargs()
        a, b = abase, apower*b
        if a == 1 or b == 0:
            return S.One,
        elif b == 1:
            return a,
        elif a == 0:
            if isinstance(b, (Float, Integer)):
                if b.value < 0:
                    raise ValueError('Cannot divide by zero')
            return S.Zero,
        elif isinstance(a, Rational) and isinstance(b, Integer):
            if b > 0:
                return Rational(a.n**b.value, a.d**b.value),
            else:
                return Rational(a.d**-b.value, a.n**-b.value),
        elif hasattr(a, 'raiseto'):
            return getattr(a, 'raiseto')(b),
        else:
            return a, b
    
    def repr(self, lib = "", **kwargs):
        if lib == 'torch' and not self.isNumber and kwargs.get('out', False):
            return super().repr(lib, **kwargs)
        elif self.base.repr_priority == self.repr_priority:
            return f'({self.base._repr_from(lib, Pow)})**{self.power._repr_from(lib, Pow)}'
        else:
            return super().repr(lib, **kwargs)

    def lowlevel_repr(self, scalar_type="double"):
        if isinstance(self.power, Integer):
            if self.power.value > 0:
                return '(' + Mul(*self.power.value*[self.base], simplify=False).lowlevel_repr(scalar_type) + ')'
            else:
                return f'({Integer(1).lowlevel_repr(scalar_type)}/({Mul(*(-self.power.value)*[self.base], simplify=False).lowlevel_repr(scalar_type)}))'
        return f"pow({self.base.lowlevel_repr(scalar_type)}, {self.power.lowlevel_repr(scalar_type)})"
    
    @property
    def is_commutative(self):
        return False

    @property
    def base(self):
        return self._args[0]
    
    @property
    def power(self):
        return self._args[1]

    def _diff(self, var):
        if self.is_operator:
            return Diff(var)*self
        elif var not in self.power.variables:
            return Mul(self.power, self.base**(self.power-1), self.base._diff(var))
        elif var not in self.base.variables:
            return self * log(self.base) * self.power._diff(var)
        else:
            return self * (log(self.base) * self.power._diff(var) + self.power/self.base*self.base._diff(var))

    def mulargs(self):
        if self.is_operator:
            return self.power.value * self.base.mulargs()

        res = self.base.mulargs()
        if len(res) == 1:
            return self,
        else:
            return tuple([arg**self.power for arg in res])
        
    def powargs(self):
        return self._args

    @classmethod
    def matheval(cls, a, b):
        return a**b
        
    def expand(self)->Expr:
        if isinstance(self.power, Integer):
            if self.power.value > 0:
                args = self.power.value*[self.base.expand()]
                muls = itertools.product(*[arg._args if isinstance(arg, Add) else (arg,) for arg in args])
                return Add(*[Mul(*m) for m in muls])
        return self
    
    def get_ndarray(self, x, **kwargs):
        return np.power(*[arg.get_ndarray(x, **kwargs) for arg in self._args])
    
    def matrix(self, symbols: tuple[Symbol, ...], grid, acc=1, fd='central'):
        if self.is_operator:
            return tools.multi_dot_product(*(self.power.value*[self.base.matrix(symbols, grid, acc, fd)]))
        else:
            return tools.as_sparse_diag(self.array(grid, acc, fd))


class Function(Expr):

    def __new__(cls, name: str, *x: Symbol):
        assert tools.all_different(x)
        assert all([isinstance(xi, Symbol) for xi in x])
        return Expr.__new__(cls, name, *x)

    
    @property
    def name(self)->str:
        return self._args[0]
    
    # @property
    # def nd(self):
    #     return (len(self._args)-1) // 2

    def repr(self, lib="", **kwargs):
        if lib != '':
            raise NotImplementedError('Function objects do not support representation with an external library')
        else:
            return f'{self.name}({", ".join([str(x) for x in self.symbols])})'
        
    def _diff(self, var):
        return Derivative(self, var) if var in self.symbols else S.Zero
        

class Number(Atom):

    @property
    def value(self)->int|float|complex:
        raise NotImplementedError('')
    
    @cached_property
    def sgn(self):
        if not isinstance(self, Complex):
            if self.value < 0:
                return -1
        return 1

    def get_ndarray(self, x, **kwargs):
        shape = sum([x[v].shape for v in x], start=())
        return self.value * np.ones(shape=shape)
    
    def _diff(self, var):
        return S.Zero
    
    def get_ndarray(self, x, **kwargs):
        shape = sum([x[v].shape for v in x], start=())
        return self.value * np.ones(shape=shape)
    
    def eval(self):
        return asexpr(self.value)
    
    def matrix(self, symbols: tuple[Symbol, ...], grid, acc=1, fd='central'):
        return self.value * sp.identity(grid.n, format='csr')

    

class Float(Number):

    _priority = 6

    def __new__(cls, value):
        assert isinstance(value, (int, float))
        if int(value) == value:
            value = int(value)
            return Integer(value)
        else:
            return Expr.__new__(cls, value)

    @property
    def value(self)->float:
        return self._args[0]
    
    @property
    def repr_priority(self):
        if self.value < 0:
            return 1
        else:
            return self.__class__._repr_priority
    
    def repr(self, lib="", **kwargs):
        return str(self.value)
    
    def lowlevel_repr(self, scalar_type="double"):
        return str(self.value)
    
    
class Rational(Number):

    _repr_priority = 1
    _priority = 5

    def __new__(cls, m: int, n: int):
        assert n!= 0 and isinstance(m, int) and isinstance(n, int)
        sgn = int(np.sign(m*n))
        m, n = sgn*abs(m), abs(n)
        gcd = math.gcd(m, n)
        m, n = m//gcd, n//gcd
        if n == 1 and not issubclass(cls, Integer):
            return asexpr(m)
        elif issubclass(cls, Integer):
            return Expr.__new__(cls, m)
        else:
            return Expr.__new__(cls, m, n)
    
    @property
    def n(self)->int:
        return self._args[0]
    
    @property
    def d(self)->int:
        return self._args[1]

    @property
    def value(self)->int|float:
        return self._args[0]/self._args[1]
    
    @property
    def repr_priority(self):
        if self.value < 0:
            return 1
        else:
            return self.__class__._repr_priority
    
    def repr(self, lib="", **kwargs):
        return f'{self.n}/{self.d}'
        
    def lowlevel_repr(self, scalar_type="double"):
        if scalar_type == "double":
            return f'{self.n}./{self.d}.'
        else:
            T = scalar_type if " " not in scalar_type else f'({scalar_type})'
            return f'{T}({self.n})/{T}({self.d})'
    
    def mulargs(self):
        if isinstance(self, Integer):
            return self,
        else:
            return asexpr(self.n), Rational(1, self.d)
        

class Integer(Rational):

    _repr_priority = 3
    _priority = 4

    def __new__(cls, m):
        return Rational.__new__(cls, m, 1)

    def repr(self, lib="", **kwargs):
        return f'{self.value}'

    def lowlevel_repr(self, scalar_type="double"):
        if scalar_type == "double":
            return f'{self.value}.'
        else:
            T = scalar_type if " " not in scalar_type else f'({scalar_type})'
            return f'{T}({self.value})'

    @property
    def value(self)->int:
        return self._args[0]

    @property
    def d(self)->int:
        return 1
    

class Special(Number):

    _priority = 3

    def __new__(cls, name: str, value: float):
        assert value > 0 and isinstance(value, float)
        return Expr.__new__(cls, value, name)
    
    @property
    def name(self)->str:
        return self._args[1]

    @property
    def value(self)->float:
        return self._args[0]

    def repr(self, lib="", **kwargs):
        if lib == '':
            return self.name
        else:
            return f'{lib}.{self.name}'
        
    def lowlevel_repr(self, scalar_type="double"):
        return self.name
    
    def init(self, value, name, simplify=True):
        return self.__class__(name, value)
    

class Complex(Number):

    _repr_priority = 3
    _priority = 7

    def __new__(cls, real: float, imag: float):
        assert isinstance(real, (int, float)) and isinstance(imag, (int, float))
        if int(real) == real:
            real = int(real)
        if int(imag) == imag:
            imag = int(imag)
        return Expr.__new__(cls, real, imag)

    @property
    def real(self)->int|float:
        return self._args[0]
    
    @property
    def imag(self)->int|float:
        return self._args[1]
    
    @property
    def value(self)->complex:
        return self.real + 1j*self.imag

    def repr(self, lib="", **kwargs):
        if self.real == 0:
            return f'{self.imag}j'
        else:
            return f'({self.real}+{self.imag}j)'
            
    def lowlevel_repr(self, scalar_type="double"):
        return f"complex<{scalar_type}>({Float(self.real).lowlevel_repr(scalar_type)}, {Float(self.imag).lowlevel_repr(scalar_type)})"
    

class Symbol(Atom):

    _priority = 8

    def __new__(cls, name: str, axis=0):
        assert isinstance(name, str) and isinstance(axis, int)
        return Expr.__new__(cls, axis, name)

    @property
    def name(self)->str:
        return self._args[1]

    @property
    def axis(self)->int:
        return self._args[0]
    
    def _diff(self, var):
        if self == var:
            return S.One
        else:
            return S.Zero

    def get_ndarray(self, x, **kwargs):
        #assumes self in x, which is necessary to call get_ndarray
        return np.meshgrid(*[x[v] for v in x], indexing ='ij')[list(x.keys()).index(self)]
    
    @property
    def variables(self):
        return self,

    @property
    def symbols(self):
        return self,

    def to_dummy(self):
        return Dummy(self.name)
    
    def repr(self, lib="", **kwargs):
        return self.name
    
    def lowlevel_repr(self, scalar_type="double"):
        return self.name
    
    def init(self, axis, name, simplify=True):
        return self.__class__(name, axis)
    

class Dummy(Symbol):

    _priority = 9

    def __new__(cls, name: str):
        obj = Expr.__new__(cls, -1, name)
        obj._args += (uuid.uuid4().hex,)
        return obj
    
    @property
    def axis(self):
        raise NotImplementedError('')

    def eval(self):
        return self

    def _replace(self, items):
        if self in items:
            return items[self]
        return self

    def init(self, *args, simplify=True):
        res = Dummy(args[1])
        res._args = args
        return res


class Subs(Expr):

    _priority = 10

    def __new__(cls, expr: Expr, vals: Dict[Symbol, Expr], simplify=True):
        if not vals:
            return expr
        elif isinstance(expr, Subs):
            old_vals = expr.subs_data.copy()
            repeated_vals = {}
            for x in vals:
                if x not in old_vals:
                    old_vals[x] = vals[x]
                else:
                    repeated_vals[x] = vals[x]
            
            obj = Subs(expr.expr, old_vals)
            if not repeated_vals:
                return obj
            else:
                return Subs(obj, repeated_vals)
        else:
            newvals: dict[Symbol, Expr] = {}
            for x in vals:
                if not isinstance(x, Symbol):
                    raise ValueError('Keys must be Symbol objects and values must be numbers')
                if x in expr.variables:
                    newvals[x] = asexpr(vals[x])

            if not newvals:
                return expr

            dummies = [x.to_dummy() for x in newvals]
            symbols = list(newvals.keys())
            vals = list(newvals.values())
            expr = expr.varsub({x: v for x, v in zip(symbols, dummies)})
            return Expr.__new__(cls, expr, *dummies, *vals)
        
    @property
    def expr(self)->Expr:
        return self._args[0]
    
    @property
    def Nsubs(self):
        return (len(self._args)-1)//2
    
    @property
    def subs_data(self)->Dict[Symbol, Expr]:
        return {self._args[1+i]: self._args[1+i+self.Nsubs] for i in range(self.Nsubs)}
    
    @cached_property
    def variables(self):
        all_vars = super().variables
        res = []
        for x in all_vars:
            if x not in self.subs_data:
                res.append(x)
        return tuple(res)
    
    def init(self, *args, simplify=True):
        n = (len(args)-1)//2
        subs_data = {args[1+i]: args[1+i+n] for i in range(n)}
        return self.__class__(args[0], subs_data, simplify=simplify)

    def repr(self, lib="", **kwargs):
        if lib != '':
            raise NotImplementedError()
        else:
            return f'Subs({self.expr}, {self.subs_data})'
        
    def doit(self, deep=True):
        if deep:
            expr = self.expr.doit(deep=True)
        else:
            expr = self.expr

        vals = {x: self.subs_data[x] for x in self.subs_data if x in expr.variables}
        return expr._subs(vals)

    def get_ndarray(self, x, **kwargs):
        y = {}
        s = len(self.expr.variables)*[slice(None)]
        gs = []
        vars = []
        for i, v in enumerate(self.expr.variables):
            if v in self.subs_data:
                y[v] = np.array([self.subs_data[v].eval().value])
                s[i] = 0
            else:
                y[v] = x[v]
                gs.append(grids.Unstructured1D(x[v]))
                vars.append(v)
        g = grids.NdGrid(*gs)
        arr = self.expr.get_ndarray(y, **kwargs)
        arr = arr[tuple(s)]
        if arr.ndim > 0:
            return DummyScalarField(arr, g, *vars).get_ndarray(x)
        else:
            return arr
        
    def eval(self):
        if self.isNumber:
            return asexpr(float(self.get_ndarray({})))
        else:
            return Expr.eval(self)


class Derivative(Expr):

    _priority = 11

    def __new__(cls, f: Expr, *vars: Symbol, simplify=True):
        if not vars:
            return f
        elif all([f.is_const_wrt(var) for var in vars]):
            return S.Zero
        elif isinstance(f, Derivative):
            return Derivative(f.f, *vars, *f.diff_symbols)

        return Expr.__new__(cls, f, *vars)
    
    @property
    def f(self)->Expr:
        return self.args[0]
    
    @property
    def diff_symbols(self)->tuple[Symbol, ...]:
        return self.args[1:]
    
    @cached_property
    def diffcount(self)->dict[Symbol, int]:
        res = {}
        for x in self.diff_symbols:
            res[x] = res.get(x, 0) + 1
        return res
        
    @property
    def sgn(self):
        return self.f.sgn
    
    def neg(self):
        return self.init(-self.f, *self.diff_symbols)
    
    def repr(self, lib="", **kwargs):
        if lib != '':
            raise NotImplementedError('.repr() not supported by external libraries for unevaluated derivatives')
        else:
            return f"Derivative({", ".join(self.args)})"

    def get_ndarray(self, x, **kwargs):
        acc = kwargs.get('acc', 1)
        fd = kwargs.get('fd', 'central')
        y = x.copy()
        # s = self.nd*[slice(None)]

        diffgrids: Dict[Symbol, grids.Grid1D] = {}
        variables = list(x.keys())

        for i, v in enumerate(variables): #x must contain exactly the elements in self.variables
            if v in self.diffcount and self.diffcount[v] > 0:
                gs = self.get_grids(v)
                if len(gs) == 1:
                    g = gs[0]
                else:
                    g = grids.UniformFinDiffInterval(center=x[v], stepsize=1e-4, order=self.diffcount[v], acc=acc, fd=fd)
                diffgrids[v] = g
                y[v] = g.x[0]
                if len(x[v]) == 1:
                    j = g.node(x[v][0])[0]
                    # s[i] = slice(j, j+1) #this instead of just j, so that the dimensions are conserved
        arr = self.f.get_ndarray(y, **kwargs)
        for v in self.diffcount:
            if self.diffcount[v] > 0:
                arr = diffgrids[v].findiff.apply(arr, order=self.diffcount[v], acc=acc, fd=fd, axis=variables.index(v))
        return tools.interpolate(tuple(y.values()), arr, tuple(x.values()))
    
    def expand(self)->Expr:
        f = self.f.expand()
        return Add(*[self.init(arg, *self.diff_symbols) for arg in f.addargs()])
    
    def doit(self, deep=True):
        if deep:
            res = self.f.doit(deep=True)
        else:
            res = self.f

        for x in self.diffcount:
            if self.diffcount[x] == 0:
                continue
            elif res.is_const_wrt(x):
                return S.Zero
            else:
                for _ in range(self.diffcount[x]):
                    res = res._diff(x)
        return res
    
    def _subs(self, vals):
        return Subs(self, vals)


class Diff(Expr):

    _priority = 16


    def __new__(cls, var: Symbol, order=1, simplify=True):
        if isinstance(order, Integer):
            order = order.value
        if not isinstance(var, Symbol):
            raise ValueError(f'var must be of an instance of Symbol, not of type {type(var)}')
        elif not isinstance(order, int) or order < 0:
            raise ValueError(f'The Diff order can only be integer and positive')
        if order == 0:
            return S.One
        else:
            return Expr.__new__(cls, var, order)
        
    @property
    def variables(self):
        return ()

    @property
    def symbol(self)->Symbol:
        return self._args[0]
    
    @property
    def order(self)->int:
        return self._args[1]
    
    def mulargs(self):
        return self.order*(Diff(self.symbol),)
        
    def _diff(self, var):
        return Diff(var)*self

    def repr(self, lib: str):
        if lib != '':
            raise NotImplementedError('Differential operators do not support repr()')
        else:
            if self.order == 1:
                return f'Diff({self.symbol})'
            else:
                return f'Diff({self.symbol}, {self.order})'
        
    def powargs(self):
        return Diff(self.symbol), self.order

    def raiseto(self, power):
        return Diff(self.symbol, self.order*power)
    
    def matrix(self, symbols: tuple[Symbol, ...], grid, acc=1, fd='central'):
        return grid.partialdiff_matrix(order=self.order, axis=symbols.index(self.symbol), acc=acc, fd=fd)
    
    def _subs(self, vals):
        if self.symbol in vals:
            raise ValueError('')
        return self


class Integral(Expr):

    _priority = 12

    def __new__(cls, f: Expr, var: Symbol, a: Expr, b: Expr, simplify=True):
        f, a, b = asexpr(f), asexpr(a), asexpr(b)
        if var not in f.variables:
            return var * f
        elif isinstance(f, Derivative):
            if var in f.symbols:
                diffs = f.diffcount.copy()
                diffs[var] -= 1
                args = sum([diffs[var]*[var] for var in diffs], start=[])
                return Subs(Derivative(f.f, *args), {var: b}) - Subs(Derivative(f.f, *args), {var: a})
            
        s = var.to_dummy()
        f = f._subs({var: s})
        return Expr.__new__(cls, f, s, a, b)

    @property
    def f(self)->Expr:
        return self.args[0]
    
    @property
    def symbol(self)->Symbol:
        return self.args[1]
    
    @property
    def limits(self)->tuple[Expr, Expr]:
        return self.args[2:]

    @property
    def sgn(self):
        return self.f.sgn
    
    @cached_property
    def variables(self)->tuple[Symbol,...]:
        res = []
        for x in Expr.variables.__get__(self):
            if x != self.symbol:
                res.append(x)
        return tuple(res)


    def neg(self):
        return self.init(-self.f, self.symbol, *self.limits)
    
    def repr(self, lib="", **kwargs):
        if lib != '':
            raise NotImplementedError('.repr() not supported by external libraries for unevaluated integrals')
        else:
            return f'Integral({", ".join(self.args)})'
        
    def lowlevel_repr(self, scalar_type="double"):
        raise NotImplementedError('')

    def get_ndarray(self, x, **kwargs):
        raise NotImplementedError('Not yet implemented, soon to be')
        r = x[self.symbol]
        if len(r) == 1:
            return S.Zero.get_ndarray(x)
        gs = self.get_grids(self.symbol)
        if len(gs) == 1:
            gx = gs[0].x[0]
        else:
            gx = r
        y = x.copy()
        y[self.symbol] = gx

        f = self.f.get_ndarray(y, **kwargs)
        varorder = list(x.keys())
        arr = tools.cumulative_simpson(f, *varorder, initial=0, axis=varorder.index(self.symbol))
        #in the future, if r[0] != self.x0 is implemented, then after the cumulative_simpson is performed,
        #we need to properly index the array to keep only the values for the given array
        return tools.interpolate(tuple(y.values()), arr, tuple(x.values()))
    
    def _diff(self, var):
        if var not in self.f.variables:
            return Subs(self.f, {self.symbol: self.limits[0]})*self.limits[0]._diff(var) - Subs(self.f, {self.symbol: self.limits[1]})*self.limits[1]._diff(var)
        else:
            return Derivative(self, var)
        

class ScalarField(Function):

    _priority = 13

    def __new__(cls, ndarray: np.ndarray, grid: grids.Grid, name: str, *vars: Symbol, simplify=True):
        if ndarray.shape != grid.shape:
            raise ValueError(f'Grid shape is {grid.shape} while field shape is {ndarray.shape}')
        if len(vars) != grid.nd:
            raise ValueError(f'Grid shape is {grid.shape} while the given variables are {len(vars)} in total: {", ".join([str(x) for x in vars])}')
        obj = Function.__new__(cls, name, *vars)
        obj._args = (ndarray, grid) + obj._args
        return obj

    def __call__(self, *args):
        if any([isinstance(arg, Expr) and not arg.isNumber for arg in args]):
            return EvaluatedScalarField(self._ndarray, self.grid, self.name, *args)
        if hasattr(args[0], '__iter__'):
            return self.interpolator(args, method="cubic")
        else:
            args = np.array(args)
            return self.interpolator(args, method="cubic")[0]
        
    def __eq__(self, other):
        if not isinstance(other, ScalarField):
            return False
        elif self is other:
            return True
        elif (self._ndarray is other._ndarray):
            return self._args[1:] == other._args[1:]
        elif self._ndarray.shape == other._ndarray.shape:
            return np.all(self._ndarray == other._ndarray) and self._args[1:] == other._args[1:]
        else:
            return False
        
    def __hash__(self):
        return hash((self.__class__,) + self._hashable_content)
        
    @property
    def _ndarray(self)->np.ndarray:
        return self.args[0]
    
    @property
    def grid(self)->grids.Grid:
        return self.args[1]

    @property
    def name(self)->str:
        return self.args[2]

    @property
    def variables(self)->tuple[Symbol,...]:
        return self.args[3:]
    
    @property
    def ndim(self)->int:
        return self.grid.nd
    
    @cached_property
    def interpolator(self):
        return RegularGridInterpolator(self.grid.x, self._ndarray, method="cubic")
    
    @property
    def _hashable_content(self):
        return (_HashableNdArray(self._ndarray), _HashableGrid(self.grid), *self.variables)
    
    def to_dummy(self):
        return DummyScalarField(self._ndarray, self.grid, *self.variables)

    def as_interped_array(self):
        return InterpedArray(self._ndarray, self.grid)

    def get_ndarray(self, x: Dict[Symbol, np.ndarray], **kwargs):
        for v in self.variables:
            if v not in x:
                raise ValueError(f"Symbol '{v}' of {self.__class__.__name__} object not included in varorder")

        ordered_vars = list(self.variables)
        mygrid = list(self.grid.x)
        extra_grid = []
        
        for v in x:
            if v not in self.variables:
                ordered_vars.append(v)
                extra_grid.append(x[v])
        grid = mygrid+extra_grid
        interpgrid = []
        for v in ordered_vars:
            interpgrid.append(x[v])
        
        arr = tools.repeat_along_extra_dims(self._ndarray, tuple([len(xi) for xi in extra_grid]))
        arr = tools.interpolate(tuple(grid), arr, tuple(interpgrid))
        return tools.swapaxes(arr, ordered_vars, list(x.keys()))

    def rearrange_as(self, *variables: Symbol):
        assert len(variables) == len(self.variables)
        for xi in variables:
            if xi not in self.variables:
                raise ValueError(f'Symbol "{xi}" not in {self}')
            elif variables.count(xi) > 1:
                raise ValueError(f'Repeated variable "{xi}')
        newaxes = [self.variables.index(x) for x in variables]
        obj = self.as_interped_array().reorder(*newaxes)
        return self.init(obj._ndarray, obj.grid, *self.args[2:])

    def directional_diff(self, x: tuple, direction: tuple, order=1, acc=1, fd='central'):
        '''
        Computes the directional derivative of a field at given coordinates

        Parameters
        -----------------
        x: coordinates ( e.g x = (9, 12.34) )
        direction: direction used for the differentiation. e.g. direction = (1, 1)
        order: order of differentiation. If order = 1, then we get df/dn. If order = 2, we get d^f/dn^2, where n is the direction
        acc: Accuracy of finite differences
        fd (0, 1 or -1): Central, forward, or backward finite differences
            If fd = 0, central finite differences are used if possible (might not be possible near the grid edges, unless the grid is periodic).
                If not possible backward or forward finite differences will be used. This option does not raises errors.
            If fd = 1, forward finite differences are used if possible
                If not possible, an error will be raised.
            If fd = -1, backward finite differences are used if possible
                If not possible, an error will be raised.

        Returns
        -----------------
        Single number, df/dn evaluated at given coordinates

        Switching vec -> -vec and fd -> -fd, yields the exact opposite result (for odd order), because the same nodes
        are used for the finite differences, but with opposite signs (for odd order, again). This
        is because for odd order, backward finite difference coefficients are opposite to forward fdc.
        '''
        node = self.grid.node(*x)
        m = self.grid.directional_diff_element(node=node, direction=direction, order=order, acc=acc, fd=fd)
        return m.dot(self.array(self.grid, acc=acc))[self.grid.flatten_index(node)]
    
    def interpolate(self, grid: grids.Grid):
        if grid.limits != grid.limits or grid.periodic != self.grid.periodic:
            raise ValueError('Grids not compatible')
        
        return self.init(self.ndarray(self.variables, grid=grid), grid, *self.args[2:])

    def plot(self, symbols:tuple[Symbol] = None, grid: grids.Grid=None, acc=1, fd='central', ax=None, **kwargs):
        if symbols is None:
            symbols = list(self.variables)
        if grid is None:
            grid = self.grid
        return Expr.plot(self, symbols=symbols, grid=grid, acc=acc, fd=fd, ax=ax, **kwargs)
    
    def _subs(self, vals):
        return Subs(self, vals)


class EvaluatedScalarField(ScalarField):
    

    def __new__(cls, ndarray: np.ndarray, grid: grids.Grid, name: str, *values: tuple[Expr, ...], simplify=True):
        if ndarray.shape != grid.shape:
            raise ValueError(f'Grid shape is {grid.shape} while field shape is {ndarray.shape}')
        if len(values) != grid.nd:
            raise ValueError(f'Grid shape is {grid.shape} while the given variables are {len(vars)} in total: {", ".join([str(x) for x in vars])}')
        obj = Expr.__new__(cls, name, *[asexpr(value) for value in values])
        obj._args = (ndarray, grid) + obj._args
        return obj

    @property
    def variables(self)->tuple[Symbol,...]:
        return Expr.variables.__get__(self)
    
    @cached_property
    def symbols(self)->tuple[Symbol, ...]:
        return Expr.symbols.__get__(self)
    
    def eval(self):
        if len(self.symbols) == 0:
            return asexpr(ScalarField.__call__(self, *[x.eval().value for x in self._values]))
        else:
            return Expr.eval(self)
        
    def repr(self, lib="", **kwargs):
        if lib != '':
            raise NotImplementedError('Function objects do not support representation with an external library')
        else:
            return f'{self.name}({", ".join([x.repr(lib, **kwargs) for x in self._values])})'
    
    def lowlevel_repr(self, scalar_type="double"):
        return f'{self.name}({", ".join([x.lowlevel_repr(scalar_type) for x in self._values])})'
    

    @cached_property
    def _values(self) -> tuple[Expr, ...]:
        return self.args[3:]
    
    @property
    def _hashable_content(self):
        return (_HashableNdArray(self._ndarray), _HashableGrid(self.grid), *self._values)
    
    def _diff(self, var):
        '''
        d/dx P (f1(x, y), f2(x, y), ... ) = sum_i dP/dfi * dfi/dx
        '''
        res = S.Zero
        P = self.as_interped_array() # just a scalar field that can can diff wrt each axis using finite differences

        for axis, fi in enumerate(self._values):
            dP_dfi = P.diff(axis)
            dfi_dvar = fi.diff(var)
            res += EvaluatedScalarField(dP_dfi.ndarray(), dP_dfi.grid, f'{self.name}_{axis}', *self._values) * dfi_dvar
        return res

    

class DummyScalarField(ScalarField):

    _priority = 14

    def __new__(cls, ndarray: np.ndarray, grid: grids.Grid, *vars: Symbol, simplify=True):
        return ScalarField.__new__(cls, ndarray, grid, uuid.uuid4().hex, *vars)
    
    @property
    def name(self):
        return "DummyField"

    def init(self, arr: np.ndarray, grid: grids.Grid, name: str, *vars, simplify=True):
        return self.__class__(arr, grid, *vars)
    
    def _diff(self, var, acc=1, fd='central'):
        return self.diff(var, order=1, acc=acc, fd=fd)
    
    def diff(self, var: Symbol, order=1, acc=1, fd='central'):
        axis = self.variables.index(var)
        arr = InterpedArray.diff(self, axis, order, acc, fd)._ndarray
        return self.__class__(arr, self.grid, *self.variables)
    
    def integrate(self, var: Symbol):
        if var in self.variables:
            raise ValueError(f'{self.__class__.__name__} object  does not depend on "{var}"')
        arr = InterpedArray.integrate(self, axis=self.variables.index(var))
        return self.__class__(arr, self.grid, *self.variables)

    def log10(self):
        return self.init(np.log10(self.ndarray()), self.args[1:])


class Piecewise(Expr):

    _priority = 15

    def __new__(cls, *cases: tuple[Expr, Boolean], simplify=True):
        newcases = []
        for case in cases:
            cond = case[1]
            if isinstance(cond, Integer):
                if cond.value == 1:
                    cond = True
                elif cond.value == 0:
                    cond = False
                else:
                    raise ValueError(f'If the second argument of a tuple in Piecewise is an Integer, it can only be 0 or 1, not {cond.value}')
            elif not isinstance(cond, (Boolean, bool)):
                raise NotImplementedError(f'The second argument of a tuple in the PieceWise class must be Boolean, not {cond.__class__}')
            assert isinstance(cond, (Boolean, bool))
            if cond is False:
                continue
            elif cond is True:
                newcases.append((asexpr(case[0]), True))
                break
            else:
                newcases.append((asexpr(case[0]), cond))
        cases = tuple(newcases)
        assert cases[-1][1] is True
        
        if len(cases) == 1:
            return cases[0][0]
        
        default = cases[-1][0]
        if isinstance(default, Piecewise):
            cases = cases[:-1] + default.args
        return Expr.__new__(cls, *[case[0] for case in cases], *[case[1] for case in cases])

    @property
    def default(self):
        return self.args[-2]
    
    @property
    def N(self):
        return len(self.args)//2
    
    @cached_property
    def expressions(self)->tuple[Expr, ...]:
        return self._args[:self.N]
    
    @cached_property
    def booleans(self)->tuple[Boolean, ...]:
        return self._args[self.N:]
    
    @property
    def cases(self)->tuple[tuple[Expr, Boolean], ...]:
        return tuple([(self.expressions[i], self.booleans[i]) for i in range(self.N)])
    
    def remake_cases(self, attr: str, *args, **kwargs):
        cases = [getattr(arg, attr)(*args, **kwargs) for arg in self.expressions]
        return self.init(*cases, *self.booleans)
    
    def init(self, *args, simplify=True):
        n = len(args)//2
        cases = []
        for i in range(n):
            cases.append((args[i], args[i+n]))
        return self.__class__(*cases, simplify=simplify)

    def _diff(self, var: Symbol):
        return self.remake_cases("_diff", var)
    
    def _elementwise_boolean(self, x: Dict[Symbol, np.ndarray], **kwargs)->tuple[np.ndarray,...]:
        res = []
        for i in range(self.N-1):
            res.append(self.booleans[i].get_ndarray(x, **kwargs))
        res.append(Integer(1).get_ndarray(x, **kwargs))
        return tuple(res)
    
    def repr(self, lib="", **kwargs):
        if lib == '':
            return f"{self.__class__.__name__}({', '.join([f'({self.expressions[i]}, {self.booleans[i]})' for i in range(self.N)])})"
        elif lib == 'numpy':
            return f'numpy.where({self.booleans[0].repr(lib, **kwargs)}, {self.expressions[0].repr(lib, **kwargs)}, {self.init(*self.expressions[1:], *self.booleans[1:]).repr(lib, **kwargs)})'
        elif lib == 'torch':
            return f'torch.where({self.booleans[0].repr(lib, **kwargs)}, {self.expressions[0].repr(lib, **kwargs)}, {self.init(*self.expressions[1:], *self.booleans[1:]).repr(lib, **kwargs)}, out={'None' if self.isNumber else 'out'})'
        else:
            return f'({self.expressions[0].repr(lib, **kwargs)} if {self.booleans[0].repr(lib, **kwargs)} else ({self.init(*self.expressions[1:], *self.booleans[1:]).repr(lib, **kwargs)}))'

    def lowlevel_repr(self, scalar_type="double"):
        return f"(({self.booleans[0].lowlevel_repr(scalar_type)}) ? {self.expressions[0].lowlevel_repr(scalar_type)} : {self.init(*self.expressions[1:], *self.booleans[1:]).lowlevel_repr(scalar_type)})"
        
    def get_ndarray(self, x, **kwargs):
        bools = self._elementwise_boolean(x, **kwargs)
        arrs = [arg.get_ndarray(x, **kwargs) for arg in self.expressions]
        res = arrs[-1]
        for i in range(self.N-1, -1, -1):
            res = np.where(bools[i], arrs[i], res)
        return res
    
    def eval(self):
        return self._remake_branches(Expr, "eval")


class Singleton:

    One = Integer(1)
    Zero = Integer(0)
    I = Complex(0, 1)
    pi = Special('pi', 3.141592653589793)


S = Singleton()


def asexpr(arg)->Expr:
    if isinstance(arg, Expr):
        return arg
    elif isinstance(arg, (int, np.integer)):
        return Integer(int(arg))
    elif isinstance(arg, (float, np.floating)):
        if arg == int(arg):
            return Integer(int(arg))
        else:
            return Float(float(arg))
    elif isinstance(arg, (np.ndarray)):
        if arg.ndim == 0:
            return Float(float(arg))
        else:
            raise ValueError(f"The numpy array of shape {arg.shape} is incompatible with the Expr class")
    elif type(arg) is complex:
        return Complex(arg.real, arg.imag)
    else:
        raise ValueError(f'The object {arg} of type {arg.__class__} is not compatible with the Expr class')

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
        
        muls = [Pow(Mul(*base[power]), power) for power in base]
        return Mul(*nonpow_base, *muls, simplify=False)
    elif isinstance(expr, Pow):
        return expr.init(expr.base, expr.power)
    else:
        return expr._remake_branches(Expr, "powsimp")


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
    if not isinstance(expr, Add):
        return expr
    nums, dens = [], []
    for arg in expr.args:
        if isinstance(arg, Mul):
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
    phaseiter = [[(2*k-n)*x.Arg+_s(x)*(k+Rational(n,2))*S.pi for k in range(n+1)] for x, n in zip(trigterms, powers)]
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

def _apply_seq(seq: list[Expr], other: Expr):
    res = other
    for i in range(len(seq)-1, -1, -1):
        res = seq[i].apply(res)
    return res

def symbols(arg: str):
    x = arg.split(', ')
    y = []
    for i in x:
        if i != '':
            y.append(i)
    n = len(y)
    symbols: list[Symbol] = []
    for i in range(n):
        symbols.append(Symbol(y[i]))
    return tuple(symbols)


def sqrt(x: Expr):
    return x ** Rational(1, 2)


from .hashing import _HashableGrid, _HashableNdArray, Hashable, sort_by_hash
from .mathfuncs import log, sin, cos, Abs, exp, tan, Abs, Real, Imag, Mathfunc
from .boolean import Gt, Lt, Ge, Le, Boolean
from .pylambda import lambdify
