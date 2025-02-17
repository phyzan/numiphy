from __future__ import annotations
import numpy as np
from ..findiffs import grids
from ..findiffs.grids import InterpedArray
from ..toolkit import tools
from ..toolkit.plotting import plot, animate
from typing import Type, Dict, Self
import itertools
from functools import cached_property
import math


class _Expr:

    Args: tuple
    repr_priority = 3
    is_symexpr = False
    is_operator = False
    is_analytical_expression = True
    S: _Singleton

    def __add__(self, other):
        return self._add(self, self._asexpr(other))
    
    def __sub__(self, other):
        return self._add(self, -self._asexpr(other))
    
    def __mul__(self, other):
        return self._mul(self, self._asexpr(other))
    
    def __truediv__(self, other):
        return self._mul(self, self._asexpr(other)**-1)
    
    def __pow__(self, other):
        return self._pow(self, self._asexpr(other))
    
    def __neg__(self):
        return -1*self
    
    def __radd__(self, other):
        return self._asexpr(other) + self
    
    def __rsub__(self, other):
        return self._asexpr(other) - self
    
    def __rmul__(self, other):
        return self._asexpr(other) * self
    
    def __rtruediv__(self, other):
        return self._asexpr(other) / self
    
    def __rpow__(self, other):
        return self._asexpr(other) ** self
    
    def __abs__(self):
        return self._abs(self)
    
    def __gt__(self, other):
        return Gt(self, other)
    
    def __lt__(self, other):
        return Lt(self, other)
    
    def __ge__(self, other):
        return Ge(self, other)
    
    def __le__(self, other):
        return Le(self, other)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return self.repr(lang='python', lib='')
    
    def __hash__(self):
        return 1 # hashing must be determined by equality

    def __eq__(self, other):
        other = self._asexpr(other)
        if type(other) is type(self):
            return self._equals(other)
        elif isinstance(other, _Any):
            return other == self
        return False

    @classmethod
    def _asexpr(cls, arg)->_Expr:...# also checks if it is operator or not e.g.
    
    @classmethod
    def _add(cls, *args, simplify=True)->_Expr:...# return Add.init(...)

    @classmethod
    def _mul(cls, *args, simplify=True)->_Expr:...

    @classmethod
    def _pow(cls, base, power, simplify=True)->_Expr:...

    @classmethod
    def _sin(cls, arg: _Expr, simplify=True)->_Expr:...

    @classmethod
    def _cos(cls, arg: _Expr, simplify=True)->_Expr:...

    @classmethod
    def _exp(cls, arg: _Expr, simplify=True)->_Expr:...

    @classmethod
    def _log(cls, arg: _Expr, simplify=True)->_Expr:...

    @classmethod
    def _tan(cls, arg: _Expr, simplify=True)->_Expr:...

    @classmethod
    def _abs(cls, arg: _Expr, simplify=True)->_Expr:...

    @classmethod
    def _rat(cls, m: int, n: int)->_Expr:...

    @classmethod
    def _derivative(cls, f: _Expr, *vars: _Symbol, simplify=True)->_Expr:...

    @classmethod
    def _subs(cls, expr: _Expr, vals: Dict[_Expr, _Expr], simplify=True)->_Expr:...

    @classmethod
    def _dummy(cls, arr: np.ndarray, grid: grids.Grid, *vars: _Symbol)->_DummyScalarField:...

    def _diff(self, var: _Symbol):
        return self._derivative(self, var)
    
    def _equals(self, other: Type[Self])->bool: #other must be same class as self
        return self.Args == other.Args

    @property
    def args(self)->tuple[_Expr, ...]:
        return self.Args
    
    @property
    def is_AbstractFunction(self):
        return isinstance(self, _Function)
    
    @property
    def is_MathFunction(self):
        return isinstance(self, _Mathfunc)

    def makenew(self, changefunc: str, *args, **kwargs)->_Expr:
        return self.init(*[getattr(arg, changefunc)(*args, **kwargs) for arg in self.args])

    def is_const_wrt(self, x: _Symbol):
        return x not in self.variables

    def doit(self, deep=True):
        if deep:
            return self.makenew('doit', deep=True)
        else:
            return self.init(*[arg for arg in self.args], simplify=True)#simply reinstanciate simplified, no need for makenew

    def repr(self, lang="python", lib = "")->str:...

    def get_ndarray(self, x: Dict[_Symbol, np.ndarray], **kwargs)->np.ndarray:...

    def ndarray(self, varorder: list[_Symbol], grid: grids.Grid, acc=1, fd='central')->np.ndarray:
        return self.get_ndarray({x: arr for x, arr in zip(varorder, grid.x)}, acc=acc, fd=fd)

    def body(self):
        return self
    
    def coef(self):
        return self.S.One
    
    def addargs(self):
        return self,

    def mulargs(self):
        return self,

    def powargs(self)->tuple[_Expr, _Expr]:
        return self, self.S.One

    def neg(self)->_Expr:
        return -self

    def _repr_from(self, lang: str, lib: str, oper: Type[Operation])->str:
        if oper.repr_priority <= self.repr_priority:
            return f'{self.repr(lang, lib)}'
        else:
            return f'({self.repr(lang, lib)})'

    def replace(self, items: Dict[_Expr, _Expr])->_Expr:
        if self in items:
            return self._asexpr(items[self])
        elif isinstance(self, Operation):
            obj = self.makenew('replace', items)
            if obj != self:
                return obj
            else:
                rest = items.copy()
                while rest:
                    f, val = rest.popitem()
                    if type(f) is type(self):
                        q = self.includes(*f.args)
                        if q:
                            return self.argsub(val, *q).replace(rest)
                return self
        elif isinstance(self, Node):
            return self.makenew('replace', items)
        else:
            return self

    def subs(self, vals: Dict[_Expr, _Expr]):
        res = self._subs(self, vals)
        if isinstance(res, _Subs):
            return res.doit(deep=False)
        else:
            return res

    def diff(self, var: _Symbol, order=1)->_Expr:
        res = self._derivative(self, *(order*[var]))
        if isinstance(res, _Derivative):
            return res.doit(deep=False)
        else:
            return res
    
    def eval(self)->_Expr:
        if isinstance(self, _Number):
            return self._asexpr(self.value)
        elif isinstance(self, _Mathfunc):
            arg = self.args[0].eval()
            if isinstance(arg, _Number):
                return self._asexpr(self.eval_at(arg.value))
            else:
                return self.init(arg)
        elif isinstance(self, Operation):
            args = [arg.eval() for arg in self.args]
            if isinstance(self, _Pow):
                a, b = self.base.eval(), self.power.eval()
                if isinstance(a, _Number) and isinstance(b, _Number):
                    return self._asexpr(a.value**b.value)
                else:
                    return self._pow(a, b)
            else:
                nums, rest = [], []
                for arg in args:
                    if isinstance(arg, _Number):
                        nums.append(arg.value)
                    else:
                        rest.append(arg)
                return self.init(self.matheval(*nums), *rest)
        elif isinstance(self, (_Derivative, _Integral)):
            return self.init(self.f.eval())
        elif isinstance(self, _Piecewise):
            return self.makenew('eval')
        elif isinstance(self, _Subs) and self.isNumber:
            return self._asexpr(float(self.get_ndarray({})))
        else:
            return self
    
    def get_grids(self, var: _Symbol)->tuple[grids.Grid1D]:
        gs: list[grids.Grid1D] = []
        items = self.deepsearch()
        for item in items:
            if hasattr(item, 'grid') and var in item.variables:
                g: grids.Grid = getattr(item, 'grid')
                if g not in gs:
                    gs.append(g[item.variables.index(var)])
        return tuple(gs)
    
    @property
    def nd(self):
        return len(self.variables)
    
    @cached_property
    def isNumber(self):
        return len(self.variables) == 0

    @property
    def isRealNumber(self):
        n = self.eval()
        if isinstance(n, _Number):
            return not isinstance(n, _Complex)
        else:
            return False
        
    @property
    def isPosInt(self):
        if isinstance(self, _Integer):
            if self.value > 0:
                return True
        return False

    @property
    def isNegInt(self):
        if isinstance(self, _Integer):
            if self.value < 0:
                return True
        return False
    
    @cached_property
    def sgn(self):
        return 1

    @cached_property
    def variables(self)->tuple[_Symbol,...]:
        '''
        All _Symbol objects that the expression has algebraic dependence wrt.
        Those might not explicitly appear inside the expression as objects.
        E.g a _ScalarField object f(x, y) has no analytic expression wrt to x, y,
        but f.variables returns (x, y).
        '''


        if isinstance(self, _Function):
            return self._variables
        elif isinstance(self, _Symbol):
            return self,
        elif isinstance(self, _Subs):
            return tuple([x for x in self.expr.variables if x not in self.vals])
        elif isinstance(self, _Integral):
            if self.symbol in self.f.variables:
                return self.f.variables
            else:
                return *self.f.variables, self.symbol
        elif isinstance(self, _Number):
            return ()
        else:
            x_all: list[_Symbol] = []
            for arg in self.args:
                vars = arg.variables
                for v in vars:
                    if v not in x_all:
                        x_all.append(v)
            return tuple(x_all)
    
    def expand(self)->_Expr:
        return self

    def deepsearch(self, mask: Type[Atom] = None, full=False)->tuple[Atom,...]:
        '''
        Gathers all outer Atom objects of the tree structure from .args of each node.
        TODO: In future implement 2 more different deepsearch options:
        1) only search nodes up to operation level
        2) search through all Node objects, up to the level of non-analytically expressed objects (e.g. dont look inside a Derivative or Integral object)
        '''
        if mask is not None:
            if not issubclass(mask, Atom):
                raise ValueError("The mask argument needs to be subclass of Atom (not containing any other expressions)")
            msk = mask
        else:
            msk = _Expr
        
        if isinstance(self, Atom):
            return (self,) if isinstance(self, msk) else ()
        elif full:
            container = self.Args
        else:
            container = self.args

        args: tuple[Atom,...] = ()
        for arg in container:
            if isinstance(arg, Node):
                args += arg.deepsearch(mask)
            elif isinstance(arg, msk):#Atom
                args += (arg,)
        return args
    

    def contains_type(self, cls: Type):
        if isinstance(self, cls):
            return True
        else:
            for arg in self.args:
                if arg.contains_type(cls):
                    return True
            return False

    def init(self, *args, simplify=True)->_Expr:...

    def array(self, varorder: list[_Symbol], grid: grids.Grid, acc=1, fd='central'):
        return self.ndarray(varorder, grid, acc, fd).flatten(order='F')
    
    def integral(self, varorder: list[_Symbol], grid: grids.Grid, acc=1, fd='central'):
        return tools.full_multidim_simpson(self.ndarray(varorder, grid, acc, fd), *grid.x)

    def dummify(self, varorder=None, grid: grids.Grid=None, acc=1, fd='central'):
        if varorder is None:
            varorder = self.variables
        if grid is None:
            gs = []
            for x in varorder:
                g = self.get_grids(x)
                if len(g) == 1:
                    gs.append(g[0])
                else:
                    raise ValueError(f'No 1D-grid along the {x} variable inside the expression to dummify')
            grid = grids.NdGrid(*gs)
        
        return self._dummy(self.ndarray(varorder, grid, acc, fd), grid, *varorder)


    def plot(self, varorder: list[_Symbol], grid: grids.Grid, acc=1, fd='central', ax=None, **kwargs):
        return plot(self.ndarray(varorder, grid, acc, fd), grid, ax, **kwargs)
    
    def animate(self, var: _Symbol, varorder: list[_Symbol], duration: float, save: str, grid: grids.Grid, display = True, **kwargs):
        if 'fps' in kwargs:
            fps = kwargs['fps']
            del kwargs['fps']
            grid = grid.replace(axis, grids.Uniform1D(*grid.limits[axis], fps*duration, grid.periodic[axis]))
            f = self.ndarray(varorder, grid)
        else:
            f = self.ndarray(varorder, grid)
        axis = varorder.index(var)
        return animate(str(var), f, duration, save, grid, axis, display, **kwargs)


class Node(_Expr):

    @classmethod
    def simplify(cls, *args: _Expr)->tuple[_Expr,...]:...

    def init(self, *args, simplify=True):
        '''
        The function to call to reinstanciate a Node with different args.
        For most objects, this is just self.__class__(*args)

        However for a Derivarive object, its args are just (f,) the function it differentiates,
        but not the variable it differentiates in respect with. So for a Derivative object,
        we have:

        g = Derivative(f, x, x)
        g.init(f): return g.__class__(f, *g.diffvars) #g.diffvars different from g.variables
        '''
        return self.__class__(*args, simplify=simplify)
    
    def item(self, path: list[int]):
        obj = self
        for i in path:
            obj = obj.args[i]
        return obj


class Atom(_Expr):
    
    @property
    def args(self):
        return ()

    def init(self, *, simplify=True):
        return self.__class__(*self.Args)

    def repr(self, lang="python", lib = "")->str:
        return f'{self.Args[0]}'


class Operation(Node):

    is_commutative = True
    binary: str

    def __new__(cls, *args, simplify=True):
        args = tuple([cls._asexpr(arg) for arg in args])
        if len(args) == 1:
            return args[0]
        if simplify:
            args = cls.simplify(*args)
            if len(args) == 1:
                return args[0]
            else:
                obj = Node.__new__(cls)
                obj.Args = args
        else:
            obj = Node.__new__(cls)
            obj.Args = args
        return obj

    
    def _equals(self, other: Operation):
        if len(other.args) == len(self.args):
            if self.is_commutative:
                for arg in self.args:
                    if arg not in other.args:
                        return False
                return True
            else:
                n = len(self.args)
                for i in range(n):
                    if self.args[i] != other.args[i]:
                        return False
                return True
        else:
            return False

    def includes(self, *args):
        '''
        If it includes this sequence it will be found.
        But what if it is included more than once. TODO
        '''
        if len(args) > len(self.args):
            return []
        
        n = len(self.args)
        res = []
        if self.is_commutative:
            for arg in args:
                for i in range(n):
                    if arg == self.args[i]:
                        if i not in res:
                            res.append(i)
            if len(res) == len(args):
                return res
            else:
                return []
        else:
            m = len(args)
            for i in range(n-m+1):
                if self.args[i:i+m] == args:
                    return list(range(i, i+m))
            return []

    def argsub(self, arg: _Expr, *k: int):
        '''
        works for all nodes (including non commutative operations) by default

        Could be implemented in all Node objects, but there are classes that cannot define this method
        well, like the Subs class.
        '''
        if not self.is_commutative:
            assert all([k[i+1]==k[i]+1 for i in range(len(k)-1)])
        k: list[int] = sorted(k)
        seq = list(self.args)
        seq[k[0]] = arg
        del k[0]
        k.reverse()
        for i in k:
            seq.pop(i)
        return self.init(*seq)
    
    @classmethod
    def matheval(cls, *args: float)->float:...

    def repr(self, lang="python", lib = ""):
        return self.binary.join([f._repr_from(lang, lib, self.__class__) for f in self.args])


class _Add(Operation):

    repr_priority = 0
    binary = ' + '

    @classmethod
    def simplify(cls, *seq: _Expr):
        flatseq: tuple[_Expr,...] = ()
        
        for arg in seq:
            flatseq += arg.addargs()
        
        _float = 0.
        rational = cls.S.Zero

        coef: Dict[_Expr, list[_Expr]] = {}

        for arg in flatseq:
            if isinstance(arg, _Float):
                _float += arg.value
            elif isinstance(arg, _Rational):
                rational = cls._rat(rational.n*arg.d+arg.n*rational.d, rational.d*arg.d)
            else:
                base, c = arg.body(), arg.coef()
                if base in coef:
                    coef[base].append(c)
                else:
                    coef[base] = [c]

        res = []
        if _float != 0:
            res.append(cls._asexpr(_float))
        if rational.n != 0:
            res.append(rational)
        for base in coef:
            c = cls._add(*coef[base])
            if c != 0:
                res.append(c*base)

        if not res:
            return cls.S.Zero,
        else:
            return tuple(res)

    @cached_property
    def sgn(self):
        signs = [arg.sgn for arg in self.args]
        if signs.count(1) >= signs.count(-1):
            return 1
        else:
            return -1

    def _diff(self, var):
        return self._add(*[arg._diff(var) for arg in self.args])

    def neg(self):
        return self.init(*[-arg for arg in self.args])

    def addargs(self):
        return self.args
    
    @classmethod
    def matheval(cls, *args: float)->float:
        return sum(args)
    
    def repr(self, lang="python", lib = ""):
        res = self.args[0].repr(lang, lib)
        for arg in self.args[1:]:
            r = arg.repr(lang, lib)
            if r.startswith('-'):
                res += ' - ' + r[1:]
            else:
                res += ' + ' + arg.repr(lang, lib)
        return res
    
    def expand(self):
        return self._add(*[arg.expand() for arg in self.args])

    def get_ndarray(self, x, **kwargs):
        return np.sum([arg.get_ndarray(x, **kwargs) for arg in self.args], axis=0)


class _Mul(Operation):

    repr_priority = 1
    binary = '*'

    @classmethod
    def simplify(cls, *seq: _Expr):

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

        pow: Dict[_Expr, list[_Expr]] = {}
        for arg in flatseq:
            if isinstance(arg, _Float):
                _float *= arg.value
            elif isinstance(arg, _Rational):
                rational = cls._rat(rational.n*arg.n, rational.d*arg.d)
            else:
                base, p = arg.powargs()
                if base in pow:
                    pow[base].append(p)
                else:
                    pow[base] = [p]
    
        res = []
        if isinstance(rational, _Integer) and _float != 1.:
            res.append(cls._asexpr(rational.n*_float))
        else:
            if _float != 1:
                res.append(cls._asexpr(_float))
            if rational != 1:
                res.append(rational)
        for base in pow:
            c = cls._add(*pow[base])
            if c == 0:
                continue
            res.append(base**c)

        if not res:
            return cls.S.One,
        else:
            return tuple(res)

    @cached_property
    def numerator(self):
        res = []
        for arg in self.args:
            if isinstance(arg, _Pow):
                if isinstance(arg.power, (_Float, _Integer)):
                    if arg.power.value < 0:
                        continue
            res.append(arg)
        if not res:
            return self.S.One
        else:
            return self._mul(*res, simplify=False)

    @cached_property
    def denominator(self):
        res = []
        for arg in self.args:
            if isinstance(arg, _Pow):
                if isinstance(arg.power, (_Float, _Integer)):
                    if arg.power.value < 0:
                        res.append(arg**-1)
        if not res:
            return self.S.One
        else:
            return self._mul(*res, simplify=False)

    @cached_property
    def _coef(self):
        res = []
        for arg in self.args:
            if isinstance(arg, _Number) and not isinstance(arg, _Special):
                res.append(arg)
        if len(res) == len(self.args):
            return self.S.One
        else:
            return self._mul(*res)

    @cached_property
    def _body(self):
        res = []
        for arg in self.args:
            if not isinstance(arg, _Number) or isinstance(arg, _Special):
                res.append(arg)
        if not res:
            return self
        else:
            return self._mul(*res)

    @cached_property
    def sgn(self):
        return self.args[0].sgn
    
    def _diff(self, var):
        coef = self._mul(*[arg for arg in self.args if arg.isNumber])
        body = [arg for arg in self.args if not arg.isNumber]
        toadd = []
        for i in range(len(body)):
            prod = body[:i]+[body[i]._diff(var)]+body[i+1:]
            toadd.append(self._mul(*prod))
        return coef*self._add(*toadd)

    def body(self):
        return self._body
    
    def coef(self):
        return self._coef

    @classmethod
    def matheval(cls, *args: float)->float:
        return math.prod(args)
    
    def repr(self, lang="python", lib = ""):
        num, den = self.numerator, self.denominator
        if den != 1:
            if den.repr_priority == self.repr_priority:
                s = f'{num._repr_from(lang, lib, _Mul)}/({den._repr_from(lang, lib, _Mul)})'
            else:
                s = f'{num._repr_from(lang, lib, _Mul)}/{den._repr_from(lang, lib, _Mul)}'
        else:
            s = super().repr(lang, lib)
        if s.startswith('-1*'):
            s = '-'+s[3:]
        elif  s.startswith('-1.*'):
            s = '-'+s[4:]
        return s

    def mulargs(self):
        return sum([arg.mulargs() for arg in self.args], start=())

    def expand(self):
        if not self.contains_type(_Add):
            return self
        args = [arg.expand() for arg in self.args]
        muls = itertools.product(*[arg.args if isinstance(arg, _Add) else (arg,) for arg in args])
        s = [self._mul(*m) for m in muls]
        return self._add(*s)

    def get_ndarray(self, x, **kwargs):
        return np.prod([arg.get_ndarray(x, **kwargs) for arg in self.args], axis=0)


class _Pow(Operation):

    is_commutative = False
    repr_priority = 2
    binary = '**'


    def __new__(cls, base, power, simplify=True):
        return super().__new__(cls, base, power, simplify=simplify)
    
    @classmethod
    def simplify(cls, a: _Expr, b: _Expr):
        abase, apower = a.powargs()
        a, b = abase, apower*b
        if a == 1 or b == 0:
            return cls.S.One,
        elif b == 1:
            return a,
        elif a == 0:
            if isinstance(b, (_Float, _Integer)):
                if b.value < 0:
                    raise ValueError('Cannot divide by zero')
            return cls.S.Zero,
        elif isinstance(a, _Rational) and isinstance(b, _Integer):
            if b > 0:
                return cls._rat(a.n**b.value, a.d**b.value),
            else:
                return cls._rat(a.d**-b.value, a.n**-b.value),
        elif hasattr(a, 'raiseto'):
            return getattr(a, 'raiseto')(b),
        else:
            return a, b
    
    def repr(self, lang="python", lib = ""):
        if lang == 'python':
            if self.base.repr_priority == self.repr_priority:
                return f'({self.base._repr_from(lang, lib, _Pow)})**{self.power._repr_from(lang, lib, _Pow)}'
            else:
                return super().repr(lang, lib)
        elif lang == 'c++':
            res = f"pow({self.base.repr(lang, lib)}, {self.power.repr(lang, lib)})"
            if lib == "":
                return res
            else:
                return f"{lib}::" + res
    
    @property
    def base(self):
        return self.args[0]
    
    @property
    def power(self):
        return self.args[1]

    def _diff(self, var):
        if var not in self.power.variables:
            return self._mul(self.power, self.base**(self.power-1), self.base._diff(var))
        elif var not in self.base.variables:
            return self * self._log(self.base) * self.power._diff(var)
        else:
            return self * (self._log(self.base) * self.power._diff(var) + self.power/self.base*self.base._diff(var))

    def mulargs(self):
        res = self.base.mulargs()
        if len(res) == 1:
            return self,
        else:
            return tuple([arg**self.power for arg in res])
        
    def powargs(self):
        return self.args

    @classmethod
    def matheval(cls, a, b):
        return a**b
        
    def expand(self):
        if isinstance(self.power, _Integer):
            if self.power.value > 0:
                args = self.power.value*[self.base.expand()]
                muls = itertools.product(*[arg.args if isinstance(arg, _Add) else (arg,) for arg in args])
                return self._add(*[self._mul(*m) for m in muls])
        return self
    
    def get_ndarray(self, x, **kwargs):
        return np.power(*[arg.get_ndarray(x, **kwargs) for arg in self.args])


class _Function(Atom):

    is_analytical_expression = False
    #define self.Args, probably in subclasses

    @property
    def name(self)->str:...

    @property
    def _variables(self)->tuple[_Symbol,...]:...

    def repr(self, lang="python", lib=""):
        if lib != '' or lang != 'python':
            raise NotImplementedError('_Function objects do not support representation with an external library')
        else:
            return f'{self.name}({", ".join([str(x) for x in self._variables])})'

    def get_ndarray(self, x, **kwargs):
        raise NotImplementedError('')


class _Number(Atom):

    def __gt__(self, other)->bool:
        other = self._asexpr(other)
        if isinstance(other, _Number):
            return self.value > other.value
        else:
            raise NotImplementedError('Comparison only valid between real numbers')

    def __lt__(self, other)->bool:
        other = self._asexpr(other)
        if isinstance(other, _Number):
            return self.value < other.value
        else:
            raise NotImplementedError('Comparison only valid between real numbers')

    @property
    def value(self)->int|float|complex:...

    @cached_property
    def sgn(self):
        if not isinstance(self, _Complex):
            if self.value < 0:
                return -1
            else:
                return 1
        else:
            return 1

    def get_ndarray(self, x, **kwargs):
        shape = sum([x[v].shape for v in x], start=())
        return self.value * np.ones(shape=shape)
    
    def _diff(self, var):
        return self.S.Zero


class _Float(_Number):

    def __init__(self, value: float):
        assert value != 0 and isinstance(value, float)
        self.Args = (value,)
        if value < 0:
            self.repr_priority = 1

    @property
    def value(self)->int|float:
        return self.Args[0]


class _Rational(_Number):

    repr_priority = 1

    def __new__(cls, m: int, n: int):
        assert n!= 0 and isinstance(m, int) and isinstance(n, int)
        sgn = int(np.sign(m*n))
        m, n = sgn*abs(m), abs(n)
        gcd = math.gcd(m, n)
        m, n = m//gcd, n//gcd
        if n == 1 and not issubclass(cls, _Integer):
            return cls._asexpr(m)
        else:
            obj = _Number.__new__(cls)
            if issubclass(cls, _Integer):
                obj.Args = (m,)
            else:
                obj.Args = (m, n)
            if m < 0:
                obj.repr_priority = 1 #in case cls is _Integer
            return obj
    
    @property
    def n(self)->int:
        return self.Args[0]
    
    @property
    def d(self)->int:
        return self.Args[1]

    @property
    def value(self)->int|float:
        return self.Args[0]/self.Args[1]
    
    def repr(self, lang="python", lib=""):
        if lang == "python":
            return f'{self.n}/{self.d}'
        else:
            return f'{self.n}./{self.d}.'
    
    def mulargs(self):
        if isinstance(self, _Integer):
            return self,
        else:
            return self._asexpr(self.n), self._rat(1, self.d)


class _Integer(_Rational):

    repr_priority = 3

    def __new__(cls, m):
        return _Rational.__new__(cls, m, 1)

    def repr(self, lang="python", lib=""):
        if lang == "python":
            return f'{self.value}'
        else:
            return f'{self.value}.'

    @property
    def value(self)->int|float:
        return self.Args[0]

    @property
    def d(self)->int:
        return 1

class _Special(_Number):

    def __init__(self, name: str, value: float):
        assert value > 0 and isinstance(value, float)
        self.Args = (name, value)

    @property
    def name(self)->str:
        return self.Args[0]

    @property
    def value(self)->float:
        return self.Args[1]

    def repr(self, lang="python", lib=""):
        if lib == '':
            return self.name
        else:
            return f'{lib}.{self.name}'


class _Complex(_Number):

    repr_priority = 3

    def __init__(self, real: int|float, imag: int|float):
        assert isinstance(real, (int, float)) and isinstance(imag, (int, float))
        if int(real) == real:
            real = int(real)
        if int(imag) == imag:
            imag = int(imag)
        self.Args = (real, imag)

    @property
    def real(self)->int|float:
        return self.Args[0]
    
    @property
    def imag(self)->int|float:
        return self.Args[1]
    
    @property
    def value(self)->complex:
        return self.real + 1j*self.imag

    def repr(self, lang="python", lib=""):
        if lang == "python":
            if self.real == 0:
                return f'{self.imag}j'
            else:
                return f'({self.real}+{self.imag}j)'
        elif lang == "c++":
            assert lib == "" or lib == "std"
            res = f"complex<double>({self.real}, {self.imag})"
            if lib == "":
                return res
            else:
                return f"{lib}::" + res


class _Symbol(Atom):

    '''
    Real-valued variable. Base for all symbols-variables
    '''

    def __init__(self, name: str):
        self.Args = (name,)

    @property
    def name(self)->str:
        return self.Args[0]

    def _diff(self, var):
        if self == var:
            return self.S.One
        else:
            return self.S.Zero

    def get_ndarray(self, x, **kwargs):
        return np.meshgrid(*[x[v] for v in x], indexing ='ij')[list(x.keys()).index(self)]
    

class _Subs(Node):

    is_analytical_expression = False
    

    def __new__(cls, expr: _Expr, vals: Dict[_Symbol, _Expr], simplify=True):
        if not vals:
            return expr
        elif isinstance(expr, _Subs):
            # same dict keys do not replace older ones, but they are just dismissed
            oldvals = expr.vals.copy()
            for x in vals:
                if x not in oldvals:
                    oldvals[x] = vals[x]
            return cls._subs(expr.expr, oldvals)
        else:
            vals = {x: cls._asexpr(vals[x]) for x in vals}
            newvals = {}
            for x in vals:
                if not isinstance(x, _Symbol) or not vals[x].isNumber:
                    raise ValueError('Keys must be _Symbol objects and values must be numbers')
                if x in expr.variables:
                    newvals[x] = vals[x]

            if not newvals:
                return expr

            obj = super().__new__(cls)
            obj.Args = (expr, vals)
            return obj

    @property
    def expr(self)->_Expr:
        return self.Args[0]

    @property
    def vals(self)->Dict[_Symbol, _Expr]:
        return self.Args[1]
    
    @property
    def args(self):
        return self.expr,

    def init(self, expr, simplify=True):
        return self._subs(expr, self.vals.copy(), simplify=simplify)

    def repr(self, lang="python", lib=""):
        if lib != '' or lang != 'python':
            raise NotImplementedError
        else:
            return f'Subs({self.expr}, {self.vals})'
        
    def doit(self, deep=True):
        if deep:
            expr = self.expr.doit(deep=True)
        else:
            expr = self.expr

        vals = {x: self.vals[x] for x in self.vals if x in expr.variables}
        
        if expr in vals:
            return vals[expr]
        elif not expr.is_analytical_expression:
            return self
        elif isinstance(expr, Node):
            return expr.makenew('subs', vals)
        else:
            return expr

    def get_ndarray(self, x, **kwargs):
        y = {}
        s = self.expr.nd*[slice(None)]
        gs = []
        vars = []
        for i, v in enumerate(self.expr.variables):
            if v in self.vals:
                y[v] = np.array([self.vals[v].eval().value])
                s[i] = 0
            else:
                y[v] = x[v]
                gs.append(grids.Unstructured1D(x[v]))
                vars.append(v)
        g = grids.NdGrid(*gs)
        arr = self.expr.get_ndarray(y, **kwargs)
        arr = arr[tuple(s)]
        if arr.ndim > 0:
            return self._dummy(arr, g, *vars).get_ndarray(x)
        else:
            return arr


class _Derivative(Node):

    is_analytical_expression = False

    def __new__(cls, f: _Expr, *vars: _Symbol, simplify=True):
        if not vars:
            return f
        elif all([f.is_const_wrt(var) for var in vars]):
            return cls.S.Zero
        elif isinstance(f, _Integral):
            if f.symbol in vars:
                v = list(vars)
                v.pop(v.index(f.symbol))
                return cls(f.f, *v)
            else:
                arg = cls._derivative(f.f, *vars)
                if isinstance(arg, _Derivative):
                    return f.init(arg.doit())
                else:
                    return f.init(arg)

        obj = super().__new__(cls)
        if isinstance(f, _Derivative):
            diffs = f.diffcount.copy()
            f = f.f
        else:
            diffs = {}
        
        for x in vars:
            diffs[x] = diffs.get(x, 0) + 1
        
        obj.Args = (f, *sum([diffs[xi]*(xi,) for xi in diffs], start=()))
        return obj
    
    @property
    def args(self)->tuple[_Expr]:
        return self.Args[0],
    
    @property
    def f(self)->_Expr:
        return self.Args[0]
    
    @cached_property
    def symbols(self)->tuple[_Symbol, ...]:
        return tuple(self.diffcount.keys())

    @cached_property
    def diffcount(self)->Dict[_Symbol, int]:
        res = {}
        for x in self.Args[1:]:
            res[x] = res.get(x, 0) + 1
        return res

    @property
    def sng(self):
        return self.f.sgn
    
    def neg(self):
        return self.init(-self.f)

    @classmethod
    def newvars(cls, diffcount: Dict[_Symbol, int])->tuple[_Symbol, ...]:
        res = []
        for x in diffcount:
            res += diffcount[x]*[x]
        return tuple(res)
    
    def init(self, f: _Expr, simplify=True):
        return self.__class__(f, *self.newvars(self.diffcount), simplify=simplify)
    
    def _equals(self, other: _Derivative):
        return self.f == other.f and self.diffcount == other.diffcount
    
    def repr(self, lang="python", lib=""):
        if lib != '' or lang != 'python':
            raise NotImplementedError('.repr() not supported by external libraries for unevaluated derivatives')
        else:
            v = [str(x) for x in self.newvars(self.diffcount)]
            return f"{self.__class__.__name__}({self.f}, {', '.join(v)})"

    def get_ndarray(self, x, **kwargs):
        acc = kwargs.get('acc', 1)
        fd = kwargs.get('fd', 'central')
        y = x.copy()
        # s = self.nd*[slice(None)]

        diffgrids: Dict[_Symbol, grids.Grid1D] = {}
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

    def expand(self):
        f = self.f.expand()
        return self._add(*[self.init(arg) for arg in f.addargs()])
    
    def doit(self, deep=True):
        if deep:
            res = self.f.doit(deep=True)
        else:
            res = self.f

        for x in self.diffcount:
            if self.diffcount[x] == 0:
                continue
            elif res.is_const_wrt(x):
                return self.S.Zero
            else:
                for _ in range(self.diffcount[x]):
                    res = res._diff(x)
        return res


class _Integral(Node):

    '''
    This class needs more work. Since integration algorithms are not to be implemented,
    numerical substitution will fail in most cases, usually raising the error
    'Integration only implemented when the given array starts from self.x0'
    as defined in .get_ndarray().

    Not good to work with, besides some specific cases.
    '''
    
    is_analytical_expression = False

    def __new__(cls, f: _Expr, var: _Symbol, x0, simplify=True):

        if var not in f.variables:
            return var * f
        elif isinstance(f, _Derivative):
            if var in f.symbols:
                diffs = f.diffcount.copy()
                diffs[var] -= 1
                return _Derivative(f.f, *f.newvars(diffs))

        obj = super().__new__(cls)
        obj.Args = (f, var, x0)
        return obj

    @property
    def args(self):
        return self.f,

    @property
    def f(self)->_Expr:
        return self.Args[0]
    
    @property
    def symbol(self)->_Symbol:
        return self.Args[1]
    
    @property
    def x0(self)->float:#the point where the integral starts. This is where the ndarray sets its zero.
        return self.Args[2]

    def init(self, arg, simplify=True):
        return self.__class__(arg, self.symbol, x0=self.x0)

    @property
    def sng(self):
        return self.f.sgn
    
    def neg(self):
        return self.init(-self.f)
    
    def repr(self, lang="python", lib=""):
        if lib != '' or lang != 'python':
            raise NotImplementedError('.repr() not supported by external libraries for unevaluated integrals')
        else:
            return f'{self.__class__.__name__}({self.f}, {self.symbol}, {self.x0})'

    def get_ndarray(self, x, **kwargs):
        r = x[self.symbol]
        if r[0] != self.x0:
            raise NotImplementedError('Integration only implemented when the given array starts from self.x0')
        if len(r) == 1:
            return self.S.Zero.get_ndarray(x)
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

    def expand(self):
        f = self.f.expand()
        return self._add(*[self.init(arg) for arg in f.addargs()])


class _ScalarField(_Function):

    def __init__(self, ndarray: np.ndarray, grid: grids.Grid, name: str, *vars: _Symbol):
        if ndarray.shape != grid.shape:
            raise ValueError(f'Grid shape is {grid.shape} while field shape is {ndarray.shape}')
        if len(vars) != grid.nd:
            raise ValueError(f'Grid shape is {grid.shape} while the given variables are {len(vars)} in total')
        assert tools.all_different(vars)

        self.Args = (ndarray.copy(), grid, name, *vars)

    @property
    def _ndarray(self)->np.ndarray:
        return self.Args[0]
    
    @property
    def grid(self)->grids.Grid:
        return self.Args[1]

    @property
    def name(self)->str:
        return self.Args[2]

    @property
    def _variables(self)->tuple[_Symbol,...]:
        return self.Args[3:]
    
    @property
    def ndim(self)->int:
        return self.grid.nd
    
    def to_dummy(self):
        return self._dummy(self._ndarray, self.grid, *self._variables)

    def _equals(self, other: _ScalarField):
        return np.all(self._ndarray == other._ndarray) and self.Args[1:] == other.Args[1:]

    def as_interped_array(self):
        return InterpedArray(self._ndarray, self.grid)

    def get_ndarray(self, x: Dict[_Symbol, np.ndarray], **kwargs):
        for v in self.variables:
            if v not in x:
                raise ValueError(f"Variable '{v}' of {self.__class__.__name__} object not included in varorder")

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

    def rearrange_as(self, *variables: _Symbol):
        assert len(variables) == len(self._variables)
        for xi in variables:
            if xi not in self._variables:
                raise ValueError(f'Variable "{xi}" not in {self}')
            elif variables.count(xi) > 1:
                raise ValueError(f'Repeated variable "{xi}')
        newaxes = [self._variables.index(x) for x in variables]
        obj = self.as_interped_array().reorder(*newaxes)
        return self.__class__(obj._ndarray, obj.grid, *self.Args[2:])

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
        
        return self.__class__(self.ndarray(grid=grid), grid, *self.Args[2:])

    def plot(self, varorder: list[_Symbol]=None, grid: grids.Grid=None, acc=1, fd='central', ax=None, **kwargs):
        if varorder is None:
            varorder = list(self._variables)
        if grid is None:
            grid = self.grid
        return super().plot(varorder, grid, ax=ax, **kwargs)


class _DummyScalarField(_ScalarField):

    def __init__(self, ndarray: np.ndarray, grid: grids.Grid, *vars: _Symbol):
        super().__init__(ndarray, grid, 'DummyField', *vars)
        self.Args = (ndarray, grid, *vars)

    @property
    def name(self)->str:
        return 'DummyField'

    @property
    def _variables(self)->tuple[_Symbol,...]:
        return self.Args[2:]

    def _equals(self, other):
        return False

    def _diff(self, var, acc=1, fd='central'):
        return self.diff(var, order=1, acc=acc, fd=fd)
    
    def diff(self, var: _Symbol, order=1, acc=1, fd='central'):
        axis = self._variables.index(var)
        arr = InterpedArray.diff(self, axis, order, acc, fd)._ndarray
        return self.__class__(arr, self.grid, *self._variables)
    
    def integrate(self, var: _Symbol):
        if var in self._variables:
            raise ValueError(f'{self.__class__.__name__} object  does not depend on "{var}"')
        arr = InterpedArray.integrate(self, axis=self.variables.index(var))
        return self.__class__(arr, self.grid, *self.variables)

    def log10(self):
        return self.__class__(np.log10(self.ndarray()), self.Args[1:])


class _Piecewise(Node):
    
    Args: tuple[tuple[_Expr, Condition], ...]

    def __new__(cls, *cases: tuple[_Expr, Condition], simplify=True):
        newcases = []
        for case in cases:
            assert isinstance(case[1], (Condition, bool))
            if case[1] is False:
                continue
            elif case[1] is True:
                newcases.append((cls._asexpr(case[0]), True))
                break
            else:
                newcases.append((cls._asexpr(case[0]), case[1]))
        cases = tuple(newcases)
        assert cases[-1][1] is True
        
        if len(cases) == 1:
            return cases[0][0]
        
        default = cases[-1][0]
        if isinstance(default, _Piecewise):
            cases = cases[:-1] + default.Args
        obj = super().__new__(cls)
        obj.Args = cases
        return obj

    @cached_property
    def args(self)->tuple[_Expr,...]:
        return tuple([arg[0] for arg in self.Args])

    @property
    def default(self):
        return self.Args[-1][0]
    
    @property
    def N(self):
        return len(self.Args)-1
    
    def init(self, *args: _Expr, simplify=True):
        assert len(args) == len(self.Args)
        return self.__class__(*[(args[i], self.Args[i][1]) for i in range(self.N)], (args[-1], True), simplify=simplify)
    
    def makenew(self, changefunc, *args, **kwargs):
        return self.__class__(*[(getattr(arg[0], changefunc)(*args, **kwargs), arg[1].do(changefunc, *args, **kwargs)) for arg in self.Args[:-1]], (getattr(self.Args[-1][0], changefunc)(*args, **kwargs), True))

    def neg(self):
        return self.init(*[-arg for arg in self.args])
    
    def _diff(self, var: _Symbol):
        return self.init(*[item._diff(var) for item in self.args])
    
    def _elementwise_boolean(self, x: Dict[_Symbol, np.ndarray], **kwargs)->tuple[np.ndarray,...]:
        res = []
        for i in range(self.N):
            res.append(self.Args[i][1].elementwise_eval(x, **kwargs))
        return tuple(res)
    
    def repr(self, lang="python", lib=""):
        if lang == 'python':
            if lib == '':
                return f"{self.__class__.__name__}({', '.join([str(i) for i in self.Args])})"
            elif lib == 'numpy':
                return f'numpy.where({self.Args[0][1].repr(lang, lib)}, {self.Args[0][0].repr(lang, lib)}, {self.__class__(*self.Args[1:]).repr(lang, lib)})'
            else:
                return f'({self.Args[0][0].repr(lang, lib)} if {self.Args[0][1].repr(lang, lib)} else ({self.__class__(*self.Args[1:]).repr(lang, lib)}))'
        elif lang == 'c++':
            return f"(({self.Args[0][1].repr(lang, lib)}) ? {self.Args[0][0].repr(lang, lib)} : {self.__class__(*self.Args[1:]).repr(lang, lib)})"
        
    def get_ndarray(self, x, **kwargs):
        bools = self._elementwise_boolean(x, **kwargs)
        arrs = [arg.get_ndarray(x, **kwargs) for arg in self.args]
        res = arrs[-1]
        for i in range(self.N-1, -1, -1):
            res = np.where(bools[i], arrs[i], res)
        return res
    



class _Any(_Expr):

    args = ()

    def __init__(self, cls: Type[_Expr], *assumptions: str):
        self.cls = cls
        self.assumptions = assumptions

    def _equals(self, other: _Expr):
        if isinstance(other, self.cls):
            for attr in self.assumptions:
                if not getattr(other, attr):
                    return False
            return True
        else:
            return False

    def __hash__(self):
        return hash((self.cls, self.assumptions))

    def __str__(self):
        return 'Any'


class _Singleton:

    One: _Integer
    Zero: _Integer
    I: _Complex
    pi: _Special

from .mathbase import _Mathfunc
from .inequalities import Condition, Gt, Lt, Ge, Le

'''

1) Typhinting
2) Add And, Or in Inequalities
3) Create Line2D
4) Create VectorFields
5) Add option for c/cpp code in lambdify
'''
