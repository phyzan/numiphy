from .symcore import *
from .symcore import _Add, _Mul, _Integer, _Complex, _Expr
from typing import Callable
import math, numpy as np


class _Mathfunc(Node):
    
    name: str
    npfunc: Callable[..., np.ndarray]

    def __new__(cls, arg, simplify = True):
        arg = cls._asexpr(arg)
        cls._allows(arg)
        if simplify is True:
            obj = cls.mathsimp(arg)
        else:
            obj = super().__new__(cls)
            obj.Args = (arg,)
        return obj

    def repr(self, lang="python", lib=""):
        if lang == 'python' and lib == '':
            return f'{self.__class__.__name__}({self.Arg.repr(lang, lib)})'
        
        base = f"{self.name}({self.Arg.repr(lang, lib)})"
        if lang == 'python':
            if lib == 'math' and self.contains_type(_Complex):
                return 'cmath.'+base
            else:
                return f'{lib}.{base}'
        elif lang == 'c++':
            if lib == '':
                return base
            else:
                return f'{lib}::'+base

    def _diff_unchained(self)->_Expr:...

    def _diff(self, var):
        return self._diff_unchained() * self.Arg._diff(var)

    @classmethod
    def _allows(cls, arg: _Expr)->None:...

    @property
    def Arg(self):
        return self.args[0]
    
    @property
    def evalfunc(self):
        return getattr(math, self.name)

    def get_ndarray(self, x, **kwargs):
        return self.npfunc(self.Arg.get_ndarray(x, **kwargs))

    @classmethod
    def mathsimp(cls, arg: _Expr)->_Expr:
        return cls(arg, simplify=False)


class _Sin(_Mathfunc):

    name = 'sin'
    npfunc = np.sin
    s=1

    @classmethod
    def mathsimp(cls, arg: _Expr)->_Expr:
        pi = cls.S.pi
        if isinstance(arg, _Add):
            for i in range(len(arg.args)):
                if arg.args[i].isNumber:
                    n = arg.args[i]/(pi/2)
                    if isinstance(n, _Integer):
                        n = n.value
                        newarg = cls._add(*arg.args[:i], *arg.args[i+1:], simplify=False)
                        if n % 2 == 0:
                            trigpart = cls(newarg, simplify=False)
                            coef = (-1)**(n//2)
                        else:
                            trigpart = cls._cos(newarg, simplify=False)
                            coef = (-1)**((n-1)//2)
                        return coef*trigpart
        if isinstance(arg / cls.S.pi, _Integer):
            return cls.S.Zero
        k = arg/(cls.S.pi/2)
        if isinstance(k, _Integer):
            k = k.value
            return cls._asexpr((-1)**((k-1)//2))
        elif arg.sgn == -1:
            return -cls(arg.neg())
        else:
            return cls(arg, simplify=False)
    
    @classmethod
    def addrule(cls, arg: _Expr):
        if isinstance(arg, _Add):
            x = arg.args[0]
            y = cls._add(*arg.args[1:], simplify=False)
            return cls._cos(y)*cls._sin(x) + cls._cos(x)*cls._sin(y)
        else:
            return cls._sin(arg)

    @classmethod
    def split_intcoef(cls, arg):
        if isinstance(arg, _Mul):
            if isinstance(arg.args[0], _Integer):
                n = abs(arg.args[0].value)
                x = cls._mul(int(np.sign(arg.args[0].value)), *arg.args[1:])
                adds = []
                for m in range(int((n-1)/2)+1):
                    adds.append(cls._mul((-1)**m, cls._rat(*_bin(n, 2*m+1)), cls._sin(x)**(2*m+1)*cls._cos(x)**(n-2*m-1)))
                return cls._add(*adds)
        return cls._sin(arg)
    
    def _diff_unchained(self):
        return self._cos(self.Arg)


class _Cos(_Mathfunc):

    name = 'cos'
    npfunc = np.cos
        
    @classmethod
    def mathsimp(cls, arg: _Expr)->_Expr:
        if arg.sgn == -1:
            return cls(arg.neg())
        arg = arg+cls.S.pi/2
        sinarg: _Sin = cls._sin(0, simplify=False) #only to grab the sin class
        return sinarg.mathsimp(arg)

    @classmethod
    def addrule(cls, arg: _Expr):
        if isinstance(arg, _Add):
            x = arg.args[0]
            y = cls._add(*arg.args[1:], simplify=False)
            return cls._cos(x)*cls._cos(y) - cls._sin(x)*cls._sin(y)
        else:
            return cls._cos(arg)

    @classmethod
    def split_intcoef(cls, arg):
        if isinstance(arg, _Mul):
            if isinstance(arg.args[0], _Integer):
                n = abs(arg.args[0].value)
                x = cls._mul(int(np.sign(arg.args[0].value)), *arg.args[1:])
                adds = []
                for m in range(int(n/2)+1):
                    adds.append(cls._mul((-1)**m, cls._rat(*_bin(n, 2*m)), cls._sin(x)**(2*m)*cls._cos(x)**(n-2*m)))
                return cls._add(*adds)
        return cls._cos(arg)
    
    def _diff_unchained(self):
        return -self._sin(self.Arg)


class _Exp(_Mathfunc):

    name = 'exp'
    npfunc = np.exp

    def _diff_unchained(self):
        return self
    
    def powargs(self) -> tuple[_Expr, _Expr]:
        return self._exp(self.S.One), self.Arg

    def raiseto(self, power):
        return self.init(self.Arg*power)


class _Log(_Mathfunc):

    name = 'log'
    npfunc = np.log

    def _diff_unchained(self):
        return 1/self.Arg


class _Tan(_Mathfunc):

    name = 'tan'
    npfunc = np.tan

    def _diff_unchained(self):
        return 1/self._cos(self.Arg)**2


class _Abs(_Mathfunc):

    name = 'abs'
    npfunc = np.abs

    def _diff(self, var):
        return self._derivative(self, var)

    def repr(self, lang="python", lib=""):
        base = f'abs({self.Arg.repr(lang, lib)})'
        if lang == 'python' or lib == '':
            return base
        elif lang == 'c++':
            return f'{lib}::'+base

    @property
    def evalfunc(self):
        return abs


class _Real(_Mathfunc):

    name = 'real'
    npfunc = np.real

    @classmethod
    def mathsimp(cls, arg)->_Expr:
        if arg.isRealNumber:
            return arg
        elif isinstance(arg, _Complex):
            return arg.real
        else:
            return cls(arg, simplify=False)

    def is_complex(self):
        return False
    
    def repr(self, lang="python", lib=""):
        if lang == 'python':
            if lib != '':
                return f'({self.Arg.repr(lang, lib)}).real'
            else:
                return f'{self.__class__.__name__}({self.Arg.repr(lang, lib)})'
        elif lang == 'c++':
            return f'({self.Arg.repr(lang, lib)}).real()'
        
    def _diff(self, var):
        return self.init(self.Arg._diff(var))
    
    def evalfunc(self, x):
        return getattr(x, 'real')


class _Imag(_Mathfunc):

    name = 'imag'
    npfunc = np.imag

    @classmethod
    def mathsimp(cls, arg)->_Expr:
        if arg.isRealNumber:
            return cls.S.Zero
        elif isinstance(arg, _Complex):
            return arg.imag
        else:
            return cls(arg, simplify=False)

    def is_complex(self):
        return False
    
    def repr(self, lang="python", lib=""):
        if lang == 'python':
            if lib != '':
                return f'({self.Arg.repr(lang, lib)}).imag'
            else:
                return f'{self.__class__.__name__}({self.Arg.repr(lang, lib)})'
        elif lang == 'c++':
            return f'({self.Arg.repr(lang, lib)}).imag()'
        
    def _diff(self, var):
        return self.init(self.Arg._diff(var))
    
    def evalfunc(self, x):
        return getattr(x, 'imag')


def _bin(n, k):
    return math.factorial(n), math.factorial(k)*math.factorial(n-k)

