from ..mathbase import *
from .operators import *
from ..mathbase import _Sin, _Cos, _Exp, _Log, _Tan, _Abs, _Mathfunc


class MathOperator:

    @classmethod
    def _allows(cls, arg: Operator):
        if arg.has_diff():
            raise NotImplementedError('Mathematical functions cannot take differential operators as input')

class Sin(Operator, MathOperator, _Sin):
    pass

class Cos(Operator, MathOperator, _Cos):
    pass


class Exp(Operator, MathOperator, _Exp):
    pass

class Log(Operator, MathOperator, _Log):
    pass


class Tan(Operator, MathOperator, _Tan):
    pass

class AbsOp(Operator, MathOperator, _Abs):
    pass


