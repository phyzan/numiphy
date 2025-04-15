from .lowlevelsupport import *
from .odepack import _DynamicODE #type: ignore




class DynamicOde(_DynamicODE):

    _sys: list[OdeSystem] = []
    _ptrs = []

    def __init__(self, odesys: OdeSystem, t0, q0, *, rtol=0.000001, atol=1e-12, min_step=0, max_step=np.inf, first_step=0, args: tuple[float]=(), method="RK45", savedir="", save_events_only=False):
        _sys = self.__class__._sys
        _ptrs = self.__class__._ptrs
        found = False
        for i in range(len(_sys)):
            if _sys[i] == odesys:
                found = True
                f_ptr, ev_ptr, mask_ptr = _ptrs[i]
        if not found:
            f_ptr, ev_ptr, mask_ptr = odesys.pointers()
            _sys.append(odesys)
            _ptrs.append((f_ptr, ev_ptr, mask_ptr))
            
        first_args = (f_ptr(), t0, q0)
        if mask_ptr() is not None:
            first_args += (mask_ptr(),)
        super().__init__(*first_args, rtol=rtol, atol=atol, min_step=min_step, max_step=max_step, first_step=first_step, args=args, method=method, events=ev_ptr(), savedir=savedir, save_events_only=save_events_only)


class Orbit(ODE):

    def __init__(self, ode: ODE):
        ODE.__init__(self)
        self._ode_obj = ode

    def __getattribute__(self, name):
        if name == "_ode_obj":
            return object.__getattribute__(self, name)
        else:
            return getattr(self._ode_obj, name)