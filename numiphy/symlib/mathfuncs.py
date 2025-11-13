from .symcore import *
from typing import Callable
import math, cmath, numpy as np


class Mathfunc(Expr):
    
    name: str
    npfunc: Callable[..., np.ndarray]

    def __new__(cls, arg, simplify = True):
        arg = asexpr(arg)
        if arg.is_operator:
            raise NotImplementedError('')
        if simplify is True:
            return cls.mathsimp(arg)
        else:
            return Expr.__new__(cls, arg)
    
    @classmethod
    def eval_at(cls, value):
        if isinstance(value, complex):
            return getattr(cmath, cls.name)(value)
        else:
            return getattr(math, cls.name)(value)

    def repr(self, lib="", **kwargs):
        if lib == '':
            return f'{self.__class__.__name__}({self.Arg.repr(lib, **kwargs)})'
        elif lib == 'torch':
            if self.isNumber:
                lib = "math"
            else:
                return f'torch.{self.name}({self.Arg.repr(lib, **kwargs)}, out={kwargs.get("out", "None")})'
        base = f"{self.name}({self.Arg.repr(lib, **kwargs)})"
        if lib == 'math' and self.contains_type(Complex):
            return 'cmath.'+base
        else:
            return f'{lib}.{base}'
            
    def lowlevel_repr(self, scalar_type='double'):
        return f"{self.name}({self.Arg.lowlevel_repr(scalar_type)})"

    def _diff_unchained(self)->Expr:
        raise NotImplementedError('')

    def _diff(self, var):
        return self._diff_unchained() * self.Arg._diff(var)

    @property
    def Arg(self)->Expr:
        return self.args[0]

    def get_ndarray(self, x, **kwargs):
        return self.npfunc(self.Arg.get_ndarray(x, **kwargs))

    @classmethod
    def mathsimp(cls, arg: Expr)->Expr:
        return cls(arg, simplify=False)
    
    def eval(self):
        arg = self.Arg.eval()
        if isinstance(arg, Number):
            return asexpr(self.eval_at(arg.value))
        else:
            return self.init(arg)
        


class sin(Mathfunc):

    name = 'sin'
    npfunc = np.sin
    s=1
    _priority = 26

    @classmethod
    def mathsimp(cls, arg: Expr)->Expr:
        pi = S.pi
        if isinstance(arg, Add):
            for i in range(len(arg.args)):
                if arg.args[i].isNumber:
                    n = arg.args[i]/(pi/2)
                    if isinstance(n, Integer):
                        n = n.value
                        newarg = Add(*arg.args[:i], *arg.args[i+1:], simplify=False)
                        if n % 2 == 0:
                            trigpart = cls(newarg, simplify=False)
                            coef = (-1)**(n//2)
                        else:
                            trigpart = cos(newarg, simplify=False)
                            coef = (-1)**((n-1)//2)
                        return coef*trigpart
        if isinstance(arg / S.pi, Integer):
            return S.Zero
        k = arg/(S.pi/2)
        if isinstance(k, Integer):
            k = k.value
            return asexpr((-1)**((k-1)//2))
        elif arg.sgn == -1:
            return -cls(arg.neg())
        else:
            return cls(arg, simplify=False)
    
    @classmethod
    def addrule(cls, arg: Expr):
        if isinstance(arg, Add):
            x = arg.args[0]
            y = Add(*arg.args[1:], simplify=False)
            return cos(y)*sin(x) + cos(x)*sin(y)
        else:
            return sin(arg)

    @classmethod
    def split_intcoef(cls, arg):
        if isinstance(arg, Mul):
            if isinstance(arg.args[0], Integer):
                n = abs(arg.args[0].value)
                x = Mul(int(np.sign(arg.args[0].value)), *arg.args[1:])
                adds = []
                for m in range(int((n-1)/2)+1):
                    adds.append(Mul((-1)**m, Rational(*_bin(n, 2*m+1)), sin(x)**(2*m+1)*cos(x)**(n-2*m-1)))
                return Add(*adds)
        return sin(arg)
    
    def _diff_unchained(self):
        return cos(self.Arg)


class cos(Mathfunc):

    name = 'cos'
    npfunc = np.cos
    _priority = 27
        
    @classmethod
    def mathsimp(cls, arg: Expr)->Expr:
        if arg.sgn == -1:
            return cls(arg.neg())
        arg = arg+S.pi/2
        sinarg = sin(0, simplify=False) #only to grab the sin class
        return sinarg.mathsimp(arg)

    @classmethod
    def addrule(cls, arg: Expr):
        if isinstance(arg, Add):
            x = arg.args[0]
            y = Add(*arg.args[1:], simplify=False)
            return cos(x)*cos(y) - sin(x)*sin(y)
        else:
            return cos(arg)

    @classmethod
    def split_intcoef(cls, arg):
        if isinstance(arg, Mul):
            if isinstance(arg.args[0], Integer):
                n = abs(arg.args[0].value)
                x = Mul(int(np.sign(arg.args[0].value)), *arg.args[1:])
                adds = []
                for m in range(int(n/2)+1):
                    adds.append(Mul((-1)**m, Rational(*_bin(n, 2*m)), sin(x)**(2*m)*cos(x)**(n-2*m)))
                return Add(*adds)
        return cos(arg)
    
    def _diff_unchained(self):
        return -sin(self.Arg)


class exp(Mathfunc):

    name = 'exp'
    npfunc = np.exp
    _priority = 28

    def _diff_unchained(self):
        return self
    
    def powargs(self) -> tuple[Expr, Expr]:
        return exp(S.One), self.Arg

    def raiseto(self, power):
        return self.init(self.Arg*power)


class log(Mathfunc):

    name = 'log'
    npfunc = np.log
    _priority = 29

    def _diff_unchained(self):
        return 1/self.Arg


class tan(Mathfunc):

    name = 'tan'
    npfunc = np.tan
    _priority = 30

    def _diff_unchained(self):
        return 1/cos(self.Arg)**2


class Abs(Mathfunc):

    name = 'abs'
    npfunc = np.abs
    _priority = 31

    def _diff(self, var):
        return Derivative(self, var)

    def repr(self, lib="", **kwargs):
        if lib == 'torch':
            if self.isNumber:
                lib = "math"
            else:
                return f'torch.abs({self.Arg.repr(lib, **kwargs)}, out={kwargs.get("out", "None")})'
        return f'abs({self.Arg.repr(lib, **kwargs)})'

    @classmethod
    def eval_at(cls, value):
        return abs(value)


class Real(Mathfunc):

    name = 'real'
    npfunc = staticmethod(np.real)
    _priority = 32

    @classmethod
    def mathsimp(cls, arg)->Expr:
        if arg.isRealNumber:
            return arg
        elif isinstance(arg, Complex):
            return arg.real
        else:
            return cls(arg, simplify=False)

    def is_complex(self):
        return False
    
    def repr(self, lib="", **kwargs):
        if lib != '':
            return f'({self.Arg.repr(lib, **kwargs)}).real'
        elif lib == 'torch':
            if self.isNumber:
                lib = "math"
            else:
                return f'torch.real({self.Arg.repr(lib, **kwargs)}, out={kwargs.get("out", "None")})'
        else:
            return f'{self.__class__.__name__}({self.Arg.repr(lib, **kwargs)})'
        
    def _diff(self, var):
        return self.init(self.Arg._diff(var))
    
    @classmethod
    def eval_at(cls, value):
        return getattr(value, 'real')


class Imag(Mathfunc):

    name = 'imag'
    npfunc = staticmethod(np.imag)
    _priority = 33

    @classmethod
    def mathsimp(cls, arg)->Expr:
        if arg.isRealNumber:
            return S.Zero
        elif isinstance(arg, Complex):
            return arg.imag
        else:
            return cls(arg, simplify=False)

    def is_complex(self):
        return False
    
    def repr(self, lib="", **kwargs):
        if lib != '':
            return f'({self.Arg.repr(lib, **kwargs)}).imag'
        elif lib == 'torch':
            if self.isNumber:
                lib = "math"
            else:
                return f'torch.imag({self.Arg.repr(lib, **kwargs)}, out={kwargs.get("out", "None")})'
        else:
            return f'{self.__class__.__name__}({self.Arg.repr(lib, **kwargs)})'
        
    def _diff(self, var):
        return self.init(self.Arg._diff(var))
    
    @classmethod
    def eval_at(cls, value):
        return getattr(value, 'imag')


def _bin(n, k):
    return math.factorial(n), math.factorial(k)*math.factorial(n-k)

