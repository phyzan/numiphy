from ..mathbase import *
from .symexpr import *
from ..mathbase import _Sin, _Cos, _Exp, _Log, _Tan, _Abs, _Real, _Imag


class sin(Expr, _Sin):
    pass

class cos(Expr, _Cos):
    pass


class exp(Expr, _Exp):
    pass


class log(Expr, _Log):
    pass


class tan(Expr, _Tan):
    pass

class Abs(Expr, _Abs):
    pass

class Real(Expr, _Real):
    pass

class Imag(Expr, _Imag):
    pass