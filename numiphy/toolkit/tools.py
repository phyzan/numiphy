from __future__ import annotations
from typing import Callable, Literal
import numpy as np
import scipy.sparse as sp
import scipy.integrate as scint
import scipy.sparse.linalg as spl
from scipy.interpolate import RegularGridInterpolator, interp1d
import scipy.linalg as sl
import sys
import inspect
from scipy.special import erf, erfinv
import timeit
from matplotlib.figure import Figure as Fig
import os
import pickle
import copy
import sysconfig
import importlib.util
from typing import Any
from .compile_tools import *
from pathlib import Path


def get_source_dir():
    """
    Get the directory of the currently running script.
    Works for both .py files and .ipynb notebooks.
    Returns the directory of the caller, not this module.
    """
    # Check if we're in a Jupyter/IPython environment
    try:
        exec('get_ipython()')
        # We're in IPython/Jupyter - use cwd for notebooks
        return os.getcwd()
    except NameError:
        # Not in IPython, we're in a regular Python script
        # Inspect the call stack to find the caller's file
        frame = inspect.currentframe()
        if frame is None:
            return os.getcwd()
        
        try:
            caller_frame = frame.f_back
            while caller_frame:
                caller_file = caller_frame.f_code.co_filename
                # Skip frames from this module and internal frames
                if caller_file != __file__ and not caller_file.startswith('<') and os.path.exists(caller_file):
                    return str(Path(caller_file).parent.resolve())
                caller_frame = caller_frame.f_back
            
            # Fallback to __main__ module's file
            if hasattr(sys.modules['__main__'], '__file__'):
                main_file = sys.modules['__main__'].__file__
                if main_file and os.path.exists(main_file):
                    return str(Path(main_file).parent.resolve())
        finally:
            del frame
    
    # Ultimate fallback
    return os.getcwd()

def call_with_consumed(func, **kwargs):
    sig = inspect.signature(func)
    valid_keys = set(sig.parameters)
    # separate kwargs into two parts
    used = {k: v for k, v in kwargs.items() if k in valid_keys}
    rest = {k: v for k, v in kwargs.items() if k not in valid_keys}
    return func(**used), rest

def call_builtin_with_consumed(func, defaults: dict[str, Any], *args, **kwargs):
    used = {}
    for param in defaults:
        if param in kwargs:
            used[param] = kwargs.pop(param)
        else:
            used[param] = defaults[param]
    return func(*args, **used), kwargs

def import_lowlevel_module(directory: str, module_name):
    so_file = os.path.join(directory, module_name)
    so_full_path = so_file + suffix()
    spec = importlib.util.spec_from_file_location(module_name, so_full_path)
    temp_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(temp_module)
    return temp_module


def interpolate1D(x: np.ndarray, y: np.ndarray, x_new: np.ndarray)->np.ndarray:
    f = interp1d(x, y, kind='cubic', fill_value='extrapolate')
    return f(x_new)

def contain_same_elements(a: list, b: list):
    if len(a) == len(b):
        for arg in a:
            if arg not in b:
                return False
        return True
    else:
        return False

def savefig(fig: Fig, name: str, folder=None, **kwargs):
    if folder is None:
        folder = os.getcwd()
    fig_path_png = os.path.join(folder, f"{name}.png")
    fig_path_pdf = os.path.join(folder, f"{name}.pdf")
    fig.tight_layout()
    fig.savefig(fig_path_png, bbox_inches='tight', **kwargs)
    fig.savefig(fig_path_pdf, bbox_inches='tight')


def try_read(file, error=False):
    if os.path.exists(file):
        with open(file, 'rb') as handle:
            data = pickle.load(handle)
            return data
    else:
        if error:
            raise FileNotFoundError(f'File "{file}" not found')
        else:
            with open(file, 'wb') as file:
                return None

def try_create(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def write_binary_data(file, data):
    with open(file, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def swapaxes(arr: np.ndarray, old: list, new: list):
    assert len(old) == len(new) == arr.ndim
    assert np.all([v in new for v in old])
    old = old.copy()
    new = new.copy()
    i = 0
    while i < arr.ndim:
        if old[i] == new[i]:
            i += 1
        else:
            j = new.index(old[i])
            old[i], old[j] = old[j], old[i]
            arr = arr.swapaxes(i, j)
    return arr

def repeat_along_extra_dims(arr: np.ndarray, shape: tuple[int]):
    s = arr.ndim*[slice(None)] + len(shape)*[np.newaxis]
    tiled_axis = len(arr.shape)*(1,) + shape
    arr = arr[tuple(s)]
    return np.tile(arr, tiled_axis)


def interpolate(x: tuple[np.ndarray], f: np.ndarray, x_new: tuple[np.ndarray]):
    interpolator = RegularGridInterpolator(x, f)
    res = interpolator(tuple(np.meshgrid(*x_new, indexing='ij')))
    return res

def compare_runtime(f, g, args, N):
    t1 = timeit.default_timer()
    for _ in range(N):
        f(*args)
    t2 = timeit.default_timer()
    for _ in range(N):
        g(*args)
    t3 = timeit.default_timer()
    return t2-t1, t3-t2

def all_different(a: list|tuple):
    for ai in a:
        if a.count(ai) > 1:
            return False
    return True

def prod(args):
    '''
    Argument
    ------------
    args (iterable): array of numbers

    Returns
    ------------
    Product of numbers inside 'args'
    '''
    a = 1
    for i in args:
        a *= i  
    return a

def merge(lists: list[list]):
    a = []
    for b in lists:
        a += b
    return a

def sort(a, b):
    '''
    sort a with respect to b (b is sorted, and a follows)
    '''
    if a:
        a, b = zip(*sorted(zip(a, b), key=lambda x: x[1]))
        return a, b
    else:
        return (), ()

def params_of(func):
    return inspect.signature(func).parameters

def has_args_or_kwargs(func):
    parameters = params_of(func)
    has_args = any(param for param in parameters.values() if param.kind == param.VAR_POSITIONAL)
    has_kwargs = any(param for param in parameters.values() if param.kind == param.VAR_KEYWORD)
    return has_args or has_kwargs

def func_repr(f: str, **diff: int)->str:
    if not diff:
        return f
    tot_ord = 0
    dx_tot = ''
    for x in diff:
        if diff[x] < 0 or not isinstance(diff[x], int):
            raise ValueError('Diff order can only be a non negative integer')
        if diff[x] == 0:
            dx = ''
        elif diff[x] == 1:
            dx = f'd{x}'
        else:
            dx = f'd{x}^{diff[x]}'
        tot_ord += diff[x]
        dx_tot += dx
    if tot_ord == 0:
        return f
    elif tot_ord == 1:
        df = f'd{f}'
    else:
        df = f'd^{tot_ord}{f}'
    return f'{df}/{dx_tot}'

def _is_timedependent(func: Callable[..., float]):
    nargs = func.__code__.co_argcount
    args = func.__code__.co_varnames[:nargs]
    return 't' in args

def _nullfunc(*args):
    return 0

def _floatfunc(n):
    
    def f(*args):
        return n
    
    return f

def isnumeric(n):
    return np.issubdtype(type(n), np.number)

def sign(n):
    if n > 0:
        return 1
    elif n < 0:
        return -1
    else:
        return 0

def format_number(n, k=3):
    '''
    argument: n (float)

    returns a representation of the number with 3 significant digits.
    If the number is very small, it is returned in exponential form (latex format)
    '''
    _str: str = f'%.{k}g' % n
    if 'e' in _str:
        l, r = _str.split('e')
        if r[0] == '0':
            r = r[1:]
        elif r[0] == '-' and r[1] == '0':
            r = '-' + r[2:]
        return '$l \\cdot 10^{r}$'.replace('l', l).replace('r', str(r))
    else:
        return '$'+_str+'$'
    
def _output_progress(i, steps):
    if int(steps/1000) == 0:
        _print = True
    elif i % int(steps/1000) == 0:
        _print = True
    else:
        _print = False
    
    if _print:
        sys.stdout.flush()
        sys.stdout.write('\rComputing: {} %'.format(round(100*i/steps, 2), end=""))

def fprint(a):
    sys.stdout.flush()
    sys.stdout.write('\r{}'.format(a, end=""))

def flatten_index(index: tuple[int], shape: tuple[int], order: Literal['F', 'C']):
    '''
    Arguments
    ------------
    index (tuple): index of multidimensional matrix
    shape (tuple): dimensions of matrix
    order ('F' or 'C'): If the left index increases fastest, then 'F'. If the right index increases fastest, then 'C'.

    Returns
    ------------
    index (int) of corresponding flattened array
    '''
    if len(index) != len(shape):
        raise ValueError("'Shape' and 'index' shape mismatch")
    nd = len(shape)
    res = 0
    if order == 'C':
        for k in range(nd):
            res += index[k]*prod(shape[k+1:])
    else:
        for k in range(nd):
            res += index[k]*prod(shape[:k])        
    return res

def reshape_toindex(k: int, shape: tuple, order: Literal['F', 'C']):
    '''
    Reverse procedure of unravel_index

    Arguments
    -------------
    k: Unraveled index of a flattened array
    shape (n_1, n_2, n_3): Shape of corresponding grid
    order ('F' or 'C'): If the left index increases fastest, then 'F'. If the right index increases fastest, then 'C'.

    Returns
    -------------
    multidimensional index (tuple[int]) of corresponding grid

    '''
    nd = len(shape)
    ijk = nd*[0]
    _k = k
    if order == 'C':
        for i in range(nd):
            ijk[i] = _k // prod(shape[i+1:])
            _k -= ijk[i]*prod(shape[i+1:])
    elif order == 'F':
        for i in range(nd):
            ijk[nd-1-i] = _k // prod(shape[:nd-1-i])
            _k -= ijk[nd-1-i]*prod(shape[:nd-1-i])        
    else:
        raise ValueError('Acceptable values for "order" are "F" or "C"')
    return tuple(ijk)

def maximum_nodes_along_dir(shape: tuple[int], point: tuple[int], direction: tuple[int]):
    '''
    Given an N-dimensional grid, we can extend a line along a direction (base vector with integers) from a specific point
    only until the line meets the grid boundaries.
    e.g.
    The "0" is located at point = (8, 2). The line along the direction (1, 2) from that point reaches a total of 4 points
    (3 "o", and 1 more including "0"). Then, the "0" is located at index 1 along this direction. If we choose the opposite direction, (-1, -2),
    the "0" is located at index = 2
    . . . . . . . . . . . .
    . . . . . . . . . . o .
    . . . . . . . . . . . .
    . . . . . . . . . o . .
    . . . . . . . . . . . .
    . . . . . . . . 0 . . .
    . . . . . . . . . . . .
    . . . . . . . o . . . .

    Parameters:
    -------------
    shape: shape of grid (n_x, n_y, n_z)
    point: coordinates of a point (i_x, i_y, i_z)
    direction: base vector for direction e.g (2, 0, 1)

    Returns
    ------------
    n: number of nodes along the given direction (but both ways), including the given point
    i: index of point, if we put these nodes side by side
    '''
    nd = len(shape)
    _point, _direction, _shape = [], [], []
    for i in range(nd):
        if direction[i] != 0:
            _point.append(point[i])
            _direction.append(direction[i])
            _shape.append(shape[i])
    _nd = len(_shape)
    
    l_pos, l_neg = [], []
    for i in range(_nd):
        sgn = [np.sign(_direction[i]), np.sign(-_point[i]*_direction[i])]
        l1 = sgn[0]*((_shape[i]-_point[i]-1)//abs(_direction[i]))
        l2 = sgn[1]*(abs(-_point[i])//abs(_direction[i]))
        l_neg.append(min(l1, l2))
        l_pos.append(max(l1, l2))
    
    n, i = min(l_pos) - max(l_neg) + 1, -max(l_neg)
    return n, i


def kronsum(*arrs: np.ndarray[float], reverse = False)->np.ndarray:
    '''
    kronsum(x, y) = (x kron ones(y)) + (ones(x) kron y)
    '''
    n = len(arrs)
    if reverse:
        args = (n-1, -1, -1)
    else:
        args = (n,)
    ones = [np.ones(arr.shape[0]) for arr in arrs]
    res = 0
    for i in range(n):
        toprod = []
        for j in range(*args):
            if j == i:
                toprod.append(arrs[i])
            else:
                toprod.append(ones[j])
        res += multi_kron_prod(*toprod)
    return res


def multi_kron_prod(*arrs: np.ndarray):
    res = arrs[0]
    for arr in arrs[1:]:
        res = np.kron(res, arr)
    return res


def multi_kronecker_product(*matrices: sp.csr_matrix)->sp.csr_matrix:
    """
    Calculate the multi Kronecker product of a list of sparse matrices.

    Parameters:
    - matrices (list): List of sparse matrices.

    Returns:
    - result (sparse matrix): Multi Kronecker product of the input matrices.
    """
    result = matrices[0]

    for matrix in matrices[1:]:
        result = sp.kron(result, matrix)

    return result.tocsr()

def multi_kron_prod(*arr: np.ndarray)->np.ndarray:
    """
    Calculate the multi Kronecker product of a list of sparse matrices.

    Parameters:
    - matrices (list): List of sparse matrices.

    Returns:
    - result (sparse matrix): Multi Kronecker product of the input matrices.
    """
    result = arr[0]

    for a in arr[1:]:
        result = np.kron(result, a)

    return result

def multi_dot_product(*matrices: sp.csr_matrix):
    m = matrices[0]
    for f in matrices[1:]:
        m = m.dot(f)
    return m

def tensordot(a: sp.csr_matrix, b: np.ndarray, axis: int)->np.ndarray:
    if axis != 0:
        c = b.swapaxes(0, axis)
    else:
        c = b
    
    if b.ndim == 2:
        res = a.dot(c)
    else:
        cr = c.reshape(c.shape[0], -1)
        res = a.dot(cr).reshape(a.shape[0], *c.shape[1:])
    
    if axis != 0:
        res = res.swapaxes(0, axis)
    return res

def generalize_operation(axis: int, shape: tuple, matrix: sp.csr_matrix, edges: bool = True):
    '''
    Generalizes the action of a matrix operator to more dimensions, so that it can act
    on a flattened array that when reshaped, represents an N-dimensional grid.


    Arguments
    -------------
    axis (int): The axis of the multidimensional grid that the operator should act on
        axis = 0: 'x' axis
        axis = 1: 'y' axis
        axis = 2: 'z' axis
    shape (nx, ny, nz): The shape of the multidimensional grid (reverse order)
    matrix (sparse matrix): The matrix to be generalized
    edges (bool): Whether or not to apply the procedure at the two edges of the axis


    returns
    -------------
    Generalized matrix operator
    '''
    nd = len(shape)
    if shape[axis] != matrix.shape[0]:
        raise ValueError('Matrix shape mismatch')
    matrices = []
    for i in range(nd-1, -1, -1):
        if i == axis:
            matrices.append(matrix)
        else:
            n = shape[i]
            diag = np.ones(n)
            if not edges:
                diag[0], diag[n-1] = 0, 0
            matrices.append(sp.dia_matrix((diag, 0), shape=(n,n)))

    return multi_kronecker_product(*matrices)

def eye(nods: list[int], n):
    a = np.zeros(n, dtype=int)
    a[nods] = 1
    return sp.dia_matrix((a, 0), (n, n))

def as_sparse_diag(arr: np.ndarray)->sp.csr_matrix:
    return sp.dia_matrix((arr, 0), shape=(len(arr), len(arr))).tocsr()

def sparse_det(m: sp.csr_matrix):
    """
    Compute the determinant of a sparse matrix using LU decomposition.
    
    Parameters:
    sparse_matrix (scipy.sparse.csc_matrix or similar): The input sparse matrix.
    
    Returns:
    float: The determinant of the matrix.
    """
    # Perform LU decomposition
    lu = spl.splu(m)
    L, U = lu.L, lu.U
    
    det_L = np.prod(L.diagonal())
    det_U = np.prod(U.diagonal())
    # Consider the permutation vectors perm_r and perm_c
    sign = np.prod(lu.perm_r) * np.prod(lu.perm_c)
    # The determinant of the original matrix
    return sign * det_L * det_U

def is_sparse_herm(m: sp.csr_matrix):
    m_dag = m.conj().transpose()
    
    # Check if the original matrix is equal to its conjugate transpose
    return (m != m_dag).nnz == 0

def is_tridiag(matrix: sp.csr_matrix):
    n = matrix.shape[0]
    
    # Extract diagonals
    main_diag = matrix.diagonal(0)
    upper_diag = matrix.diagonal(1)
    lower_diag = matrix.diagonal(-1)
    
    # Create a new sparse matrix with the three diagonals
    diagonals = [main_diag, upper_diag, lower_diag]
    offsets = [0, 1, -1]
    tridiagonal_matrix = sp.diags(diagonals, offsets, shape=(n, n), format='csr')

    difference = matrix - tridiagonal_matrix
    # Check if the difference matrix has any non-zero elements
    return difference.nnz == 0

def full_multidim_simpson(arr: np.ndarray, *x: np.ndarray):
    res = arr.copy()
    for i in range(len(x)):
        res = scint.simpson(res, x=x[i], axis=0)
    return res

def cumulative_simpson(arr: np.ndarray, *x: np.ndarray, initial=0., axis=0):
    return scint.cumulative_simpson(arr, x=x[axis], axis=axis, initial=initial)

def eig(a: np.ndarray, b=None, eigvals_only=False):
    if eigvals_only:
        return sl.eigvals(a, b)
    else:
        return sl.eig(a, b)


def randomly_spaced_array(a, b, n):
    """
    Generate a numpy array of 'n' numbers from 'a' to 'b' in increasing order with random spacing.

    Parameters:
    - n: int
        Number of elements in the array.
    - a: float
        The starting value of the range.
    - b: float
        The ending value of the range.

    Returns:
    - np.ndarray
        An array of 'n' numbers from 'a' to 'b' in increasing order with random spacing.
    """
    # Generate n random numbers and sort them
    nums = np.sort(np.random.rand(n))
    
    # Scale these numbers to the range [a, b]
    nums = a + (b - a) * nums
    nums[0] = a
    nums[-1] = b
    
    return nums


def inv_gaussian_dist(p, a: float, b: float, center=0., sigma=1.):
    erf1 = erf((center-a)/(np.sqrt(2)*sigma))
    erf2 = erf((center-b)/(np.sqrt(2)*sigma))

    res = center - np.sqrt(2)*sigma*erfinv(erf1*(1-p) + erf2*p)
    res[0] = a
    res[-1] = b

    return res

def suffix():
    return sysconfig.get_config_var("EXT_SUFFIX")

def bisect(f, a, b, xtol):
    err = 2*xtol
    _a = a
    _b = b
    c = a
    if f(a)*f(b) > 0:
        raise ValueError("Root not bracketed")
    
    while err > xtol:
        c = (_a+_b)/2
        if c==a or c==_b:
            break
        fm = f(c)
        if f(a) * fm > 0:
            _a = c
        else:
            _b = c
        err = abs(_b-_a)

        return (_a, c, _b)


class Iterator:
    '''
    Creates objects that iterates through all combinations of integers
    from given ranges. It differs from itertools.product on the fact
    that it can start the iteration with the left item icreasing fastest,
    and since it only conserns integers, it does not need entire arrays of them,
    but only their start and end points
    '''
    __slots__ = ['limits', 'range', 'nd', 'n', 'n_tot', 'item', '_progress']

    def __init__(self, order: Literal['F', 'C'], *ranges: list[int]):
        '''
        Parameters
        ------------
        order ('F' or 'C'): If the first index range increases fastest, then 'F'. If the last index increases fastest, then 'C'.
        ranges: list of 2 integers, representing the start and end of an array.
        '''
        self.limits = ranges
        self.nd = len(ranges)
        self.n = tuple([ranges[i][1]-ranges[i][0]+1 for i in range(self.nd)])
        self.n_tot = prod(self.n)
        self.item = [ranges[i][1] for i in range(self.nd)]
        self._progress = self.n_tot-1
        if order == 'F':
            self.range = range(self.nd)
        elif order == 'C':
            self.range = range(self.nd-1, -1, -1)

    def next(self):
        if self._progress < self.n_tot-1:
            for i in self.range:
                if self.item[i] == self.limits[i][1]:
                    self.item[i] = self.limits[i][0]
                else:
                    self.item[i] += 1
                    self._progress += 1
                    break
        else:
            self.item = [self.limits[i][0] for i in range(self.nd)]
            self._progress = 0
        return tuple(self.item)

    def __iter__(self):
        for _ in range(self.n_tot):
            yield self.next()


class KronProd:

    _nonslice = slice(None, None, None)

    def __init__(self, *f: np.ndarray):
        for fi in f:
            assert len(fi.shape) == 2
            assert fi.shape[0] == fi.shape[1]
        self.f = f
        self.N = len(f)
        self.n = tuple([fi.shape[0] for fi in f])
        self.shape = (prod(self.n), prod(self.n))
        self.sorted = False
        self.sorted_axis = 0

    def __getitem__(self, i)->np.ndarray|float:
        if type(i) is not tuple:
            raise NotImplementedError('KronProd object takes a 2d index as argument')
        
        a, b = i
        if isinstance(a, slice) and isinstance(b, int):
            if self.sorted and self.sorted_axis == 1:
                b = self.argsort[b]
            ndindex = reshape_toindex(b, self.n, order='C')
            f_tokron = [self.f[j][a, ndindex[j]] for j in range(self.N)]
            res =  multi_kron_prod(*f_tokron)
            if self.sorted and self.sorted_axis == 0:
                return res[self.argsort[a]]
            else:
                return res
        elif isinstance(a, int) and isinstance(b, slice):
            if self.sorted and self.sorted_axis == 0:
                a = self.argsort[a]
            ndindex = reshape_toindex(a, self.n, order='C')
            f_tokron = [self.f[j][ndindex[j], :] for j in range(self.N)]
            res = multi_kron_prod(*f_tokron)
            if self.sorted and self.sorted_axis == 1:
                return res[self.argsort[b]]
            else:
                return res
        elif isinstance(a, int) and isinstance(b, int):
            if self.sorted:
                if self.sorted_axis == 0:
                    a = self.argsort[a]
                else:
                    b = self.argsort[b]
            a_ndindex = reshape_toindex(a, self.n, order='C')
            b_ndindex = reshape_toindex(b, self.n, order='C')
            to_prod = [self.f[j][a_ndindex[j], b_ndindex[j]] for j in range(self.N)]
            return prod(to_prod)
        else:
            raise NotImplementedError('Slicing in an KronProd object not implemented for the slice {i}')
        
    def dot(self, other: np.ndarray):
        res = np.zeros_like(other)
        for i in range(self.shape[0]):
            res[i] += self[i, :].dot(other)
        return res
    
    def transpose(self):
        res = KronProd(*[f.transpose() for f in self.f])
        if self.sorted:
            res.sort(self.argsort, 1-self.sorted_axis)
        return res
    
    def construct(self):
        return multi_kron_prod(*self.f)
    
    def sort(self, argsort: np.ndarray[int], axis=0):
        if axis not in (0, 1):
            raise ValueError('axis can only take the value 0 or 1')
        self.argsort = argsort
        self.sorted_axis = axis
        self.sorted = True

    
    def __repr__(self):
        return str(self.construct())


class KronSum:

    '''
    The sorting algorithm still needs work. This class must not be used yet.
    '''

    def __init__(self, *arrs: np.ndarray[float]):
        for arri in arrs:
            assert arri.ndim == 1
        
        self.arrs = arrs
        self.n = tuple([len(arri) for arri in arrs])
        self.shape = (prod(self.n),)
        self.N = len(arrs)
        self.sorted = False

    def __len__(self):
        return self.shape[0]
    
    def __getitem__(self, i):
        if self.sorted:
            if i < 0:
                i = i + len(self)
            for j, item in enumerate(self):
                if j == i:
                    return item
        else:
            j = reshape_toindex(i, self.n, order='C')
            return self.get(*j)
    
    def get(self, *index):
        return sum([self.arrs[k][index[k]] for k in range(self.N)])
        
    def __iter__(self):
        if self.sorted:
            pass
        else:
            for i in range(self.shape[0]):
                yield self[i]
    
    def sort(self):
        self.sorted = True

        self.arrs = tuple([np.sort(arri) for arri in self.arrs])

    def argsort(self):
        return np.argsort(self)
    
    def construct(self):
        res = kronsum(*self.arrs)
        if self.sorted:
            return np.sort(res)
        else:
            return res
    
    def __repr__(self):
        return str(self.construct())


class Template:

    __slots__ = '__dict__',

    def __init__(self, **kwargs):
        self.__dict__['_args'] = {key: kwargs[key] for key in kwargs}
    
    @property
    def _args(self)->dict:
        return self.__dict__['_args']
    
    def __getattr__(self, attr):
        return self._args[attr]
    
    def __setattr__(self, attr, value):
        raise ValueError("Class does not allow attribute setting")
    
    def _set(self, **kwargs):
        for attr in kwargs:
            if attr not in self._args:
                raise ValueError(f'{self.__class__} object has no attribute named "{attr}"')
            else:
                self._args[attr] = kwargs[attr]
    
    def _copy_data_from(self, other: Template):
        if ( type(other) is not type(self) ) or ( list(other._args.keys()) != list(self._args.keys()) ):
            raise ValueError(f'Cannot copy data from object of type "{other.__class__}" to object of type "{self.__class__}"')
        else:
            for key in self._args:
                value = other._args[key]
                if isinstance(value, (list, np.ndarray)):
                    self._args[key] = value.copy()
                else:
                    self._args[key] = value

    def clone(self):
        obj = object.__new__(type(self))
        for key in self.__dict__:
            obj.__dict__[key] = copy.deepcopy(self.__dict__[key])
        return obj