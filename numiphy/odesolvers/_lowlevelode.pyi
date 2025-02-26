import numpy as np

class OdeResult:

    t: np.ndarray
    y: np.ndarray
    diverges: bool
    is_stiff: bool
    runtime: float


class LowLevelODE:

    def __init__(self, f):...

    def solve(self, ics: tuple, t, dt, *, rtol=1e-6, atol=1e-12, cutoff_step=0., method="RK45", max_frames=0, args=(), getcond=None, breakcond=None)->OdeResult:...

    def solve_all(self, parameter_list: list, threads=-1)->list[OdeResult]:...

    @staticmethod
    def dsolve_all(data: list, threads=-1):...

    def copy(self)->LowLevelODE:...