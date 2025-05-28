from numiphy.odesolvers import *

t, x, y, px, py, a, b = variables('t, x, y, px, py, a, b')

ev1 = SymbolicEvent("Main event", y-1, py>0)
ev2 = SymbolicPeriodicEvent("Periodic Event", 5)

odesys = OdeSystem([px, py, -x*a, -y*b], t, x, y, px, py, args=(a, b), events=[ev1, ev2])

orbit = odesys.get(t0=0, q0=[1, -2, 3, 5], rtol=0, atol=1e-10, args=(1, 2))

orbit.integrate(1000)

from scipy.integrate import solve_ivp


res = solve_ivp(odesys.lowlevel_jacobian, (0, 1000), [1, -2, 3, 5], rtol=1e-6, atol=1e-6, args=(1, 2))

print(res.t)