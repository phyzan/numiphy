from numiphy.odesolvers import *

t, x, y, px, py, a, b = variables('t, x, y, px, py, a, b')

# ev1 = SymbolicEvent("Main event", y-1, py>0)
ev2 = SymbolicPeriodicEvent("Periodic Event", 5)

odesys = OdeSystem([px, py, -x*a, -y*b], t, x, y, px, py, args=(a, b), events=[ev2])
orbit = odesys.get(t0=0, q0=[1, -2, 3, 5], rtol=0, atol=1e-10, args=(1, 1))

res = orbit.integrate(100000, max_frames=0, event_options={"Periodic Event": (100, True, 20)})

t = np.linspace(0, 20, 100)
x, y = res(t)[:, :2].T


import matplotlib.pyplot as plt

plt.plot(orbit.q[:, 0], orbit.q[:, 1])
plt.show()
