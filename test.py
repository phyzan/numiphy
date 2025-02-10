from numiphy.symlib.expressions import *
from numiphy.odesolvers import *
import matplotlib.pyplot as plt

t, x, y, px, py = variables('t, x, y, px, py')

symode = SymbolicOde(px, py, -x, -y, symbols=[t, x, y, px, py])
ode_fast = symode.to_lowlevel(stack=True)
ode_mid = symode.to_lowlevel(stack=False)
ode_slow = symode.to_python()
res: dict[ODE, OdeResult] = {ode_slow: None, ode_mid: None, ode_fast: None}

ics = [(0, [1.3, -2.4, 3.7, 0.2]), (0, [1.3, -2.4, 3.7, 0.2]), (0, [1.3, -2.4, 3.7, 0.2]), (0, [1.3, -2.4, 3.7, 0.2]), (0, [1.3, -2.4, 3.7, 0.2]), (0, [1.3, -2.4, 3.7, 0.2])]
ode_fast.solve_all(ics, 500, 0.1, err=1e-8, method='RK4')