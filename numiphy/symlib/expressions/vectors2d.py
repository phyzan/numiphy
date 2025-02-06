from __future__ import annotations
from .symexpr import *
from .symmath import *

class Line2D:

    def __init__(self, x: Expr, y: Expr, u: Variable, a: float, b: float):
        self._x = x.lambdify([u])
        self._y = y.lambdify([u])

        self._xdot = x.diff(u).lambdify([u])
        self._ydot = y.diff(u).lambdify([u])
        self.a, self.b = a, b


    def x(self, u):
        return self._x(u)
    
    def y(self, u):
        return self._y(u)
    
    def xdot(self, u):
        return self._xdot(u)
    
    def ydot(self, u):
        return self._ydot(u)
    

class VectorField2D:

    def __init__(self, Fx: Expr, Fy: Expr, x: Variable, y: Variable):
        self.xvar, self.yvar = x, y
        
