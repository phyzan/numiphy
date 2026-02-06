from __future__ import annotations
import numpy as np
from ..findiffs import grids


class _HashableObject:


    def _compare(self, other: _HashableObject)->int:...

    def __eq__(self, other):
        return self._compare(other) == 0

    def __gt__(self, other: Hashable):
        return self._compare(other) == 1
    
    def __lt__(self, other: Hashable):
        return self._compare(other) == -1
    
    def __eq__(self, other: Hashable):
        return self._compare(other) == 0
    
    def __ge__(self, other: Hashable):
        return self._compare(other) in (0, 1)
    
    def __le__(self, other: Hashable):
        return self._compare(other) in (0, -1)
    

class Hashable(_HashableObject):

    def __new__(cls, obj):
        if isinstance(obj, Expr):
            return super().__new__(cls)
        elif isinstance(obj, (_HashableObject, int, float, str)):
            return obj
        else:
            return hash(obj)

    def __init__(self, arg: Expr):
        self.arg = arg

    def __hash__(self):
        return hash(self.arg)

    def _compare(self, other: Hashable):
        if self.arg is other.arg:
            return 0
        elif self.arg.__class__ != other.arg.__class__:
            return 1 if sorted([other.arg.__class__._priority, self.arg.__class__._priority]) == [other.arg.__class__._priority, self.arg.__class__._priority] else -1
        n1, n2 = len(self.arg._hashable_content), len(other.arg._hashable_content)
        if n1 > n2:
            return 1
        elif n1 < n2:
            return -1
        else:
            for (obj1, obj2) in zip(self.arg._hashable_content, other.arg._hashable_content):
                if (Hashable(obj1) < Hashable(obj2)):
                    return -1
                elif (Hashable(obj1) > Hashable(obj2)):
                    return 1
            return 0


class _HashableNdArray(_HashableObject):

    def __init__(self, array: np.ndarray):
        # Assumes the array will not be modified.
        self.array = array

    def __hash__(self):
        return hash((self.array.shape, str(self.array.dtype), self.array.tobytes()))
        
    def _compare(self, other: _HashableNdArray):
        if other.array is self.array:
            return 0
        elif np.all(self.array == other.array):
            return 0
        else:
            ptr1, ptr2 = self.array.__array_interface__['data'][0], other.array.__array_interface__['data'][0]
            if ptr1 < ptr2:
                return -1
            elif ptr1 > ptr2:
                return 1
            else:
                return 0


class _HashableGrid(_HashableObject):

    def __init__(self, grid: grids.Grid):
        self.grid = grid

    def __hash__(self):
        return hash(self.grid)

    def _compare(self, other: _HashableGrid):
        if self.grid.nd > other.grid.nd:
            return 1
        elif self.grid.nd < other.grid.nd:
            return -1
        else:
            obj1, obj2 = _HashableNdArray(np.array(self.grid.x)), _HashableNdArray(np.array(other.grid.x))
            if obj1 > obj2:
                return 1
            elif obj1 < obj2:
                return -1
            else:
                # Compare tuples lexicographically
                return (self.grid.periodic > other.grid.periodic) - (self.grid.periodic < other.grid.periodic)
            
def sort_by_hash(*args: Expr):
    res = sorted([Hashable(arg) for arg in args])
    return tuple([arg.arg for arg in res])


from .symcore import Expr