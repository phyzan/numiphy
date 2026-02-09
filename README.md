<div align="center">

# NumiPhy

**Symbolic expressions that compile to Python, NumPy, PyTorch, and C++**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

</div>

---

NumiPhy combines intuitive symbolic mathematics with fast compiled code. Define mathematical expressions symbolically and instantly compile them to optimized C++ with zero Python overhead, targeting numerical computing.

## About NumiPhy

NumiPhy is **heavily inspired by SymPy** but intentionally **more limited and focused**. While SymPy is a comprehensive computer algebra system with extensive symbolic capabilities, NumiPhy prioritizes:

| | Feature | Description |
|---|---------|-------------|
| :keyboard: | **Modern Type Hinting** | Built with full Python type hints from the ground up, providing better IDE support and type safety |
| :zap: | **Direct C++ Compilation** | Automatically compiles symbolic expressions to optimized C++ shared libraries and returns function pointers, not just code generation |
| :triangular_ruler: | **Symbolic Operators** | First-class support for differential operators within symbolic expressions, naturally integrated with the math system |

> :bulb: Be sure to use **SymPy** for comprehensive symbolic math (integration, series, limits, simplification, equation solving, etc.)

## Key Features

### 1. Symbolic Math Expressions

NumiPhy provides a **full symbolic algebra system** for building mathematical expressions with ease:

```python
from numiphy.symlib.symcore import *

# Define symbolic variables
x = Symbol('x')
y = Symbol('y')
z = Symbol('z')

# Build symbolic expressions naturally, and perform partial differentiations
expr = sin(x) * cos(y) + sqrt(x**2 + y**2)
complex_expr = exp(-z) * (x**2 + y**2) + log(abs(x))
trig_expr = sin(x)**2 + cos(x)**2
gauss = exp(-(x**2 + y**2))
print(gauss.diff(x))
```

**Symbolic Features:**
- **Expression Trees**: Full operator overloading for intuitive syntax
- **Mathematical Functions**: sin, cos, tan, exp, log, sqrt, Abs, etc.
- **Multi-variable Support**: Combine any number of symbols in complex expressions
- **Boolean Expressions**: Comparison operators for conditional logic
- **Flexible Evaluation**: Convert to executable code in multiple ways

### 2. Low Level Callables: Compile to C++

This is NumiPhy's superpower. Automatically compile symbolic expressions to optimized C++ and get function pointers:

```python
from numiphy.lowlevelsupport import *
from numiphy.toolkit.tools import import_lowlevel_module
# Define a symbolic expression
x = Symbol('x')
y = Symbol('y')
expr = (x**2 + y**2)**0.5  # Distance from origin

# Compile to C++ - returns a function pointer
callable_obj = ScalarLowLevelCallable(expr, x, y)
c_func_ptr = callable_obj.compile()  # Gets void* pointer to compiled function

# Use with external libraries that accept C++ function pointers
```

**How It Works:**
1. **C++ Code Generation**: Symbolic expression → type-safe C++ code
2. **Pybind11 Compilation**: Compiled to native shared library (.so/.dll)
3. **Function Pointers**: Returns void* pointers to compiled functions
4. **External Integration**: Pass directly to libraries accepting C function pointers
5. **Zero Python Overhead**: Runs at native C++ speed with std::sin, std::cos, etc.

**Generated Code Example:**
```cpp
double _lowlevel_func_0(const double& x, const double& y, const void*) {
    return std::sqrt(x*x + y*y);
}
```

**Callable Types:**
- `ScalarLowLevelCallable`: Returns double scalars - fastest for simple computations
- `TensorLowLevelCallable`: Returns arrays for multi-dimensional outputs
- `BooleanLowLevelCallable`: Returns boolean conditions
- Automatic type inference from symbolic expressions

**Save and Reuse Compiled Functions:**
```python
# Use with external libraries that accept C++ function pointers
# Save to disk for later use
callable_obj.compile(directory='compiled_funcs', module_name='func_name')

# Load without recompiling
c_funcs = import_lowlevel_module('compiled_funcs', 'func_name').pointers() #tuple of pointers in general
print(c_funcs)
```

### 3. Multiple Execution Backends

Similar to SymPy, convert symbolic expressions to executable code in various ways:

```python
from numiphy.symlib.symcore import *
import numpy as np
import torch

x = Symbol('x')
y = Symbol('y')
expr = sin(x) * exp(-y**2)

# NumPy lambdify (vectorized operations on arrays)
f_numpy = expr.lambdify(x, y, lib='numpy')
result = f_numpy(np.array([0.5]), np.array([1.0]))

# PyTorch lambdify (GPU-capable tensors)
f_torch = expr.lambdify(x, y, lib='torch')
x_gpu = torch.tensor([0.5], device='cpu')
y_gpu = torch.tensor([1.0], device='cpu')
result = f_torch(x_gpu, y_gpu)

# Standard library (pure Python)
f_math = expr.lambdify(x, y, lib='math')
result = f_math(0.5, 1.0)
```

**Backends:**
- **`'numpy'`**: Vectorized NumPy operations - great for arrays
- **`'torch'`**: PyTorch tensors - GPU acceleration support
- **`'math'`**: Python standard library - pure Python, no dependencies
- **`ScalarLowLevelCallable.compile()`**: C++ function pointer - pass to scientific libraries and ctypes

## Library Structure

```
numiphy/
├── symlib/                  # Symbolic computation core
│   ├── symcore.py           # Expression trees, atoms, derivatives, integrals
│   ├── mathfuncs.py         # Math functions (sin, cos, exp, log, etc.)
│   ├── boolean.py           # Boolean expression system (And, Or, Not)
│   ├── geom.py              # Geometric object abstractions for PDEs
│   ├── hashing.py           # Hashing utilities for comparable objects
│   └── pylambda.py          # Callable wrappers for symbolic expressions
│
├── findiffs/                # Grid & finite differences
│   ├── grids.py             # 1D and N-dimensional grid representations
│   └── finitedifferences.py # Finite difference operators and weights
│
├── pdes/                    # PDE solvers
│   ├── bounds.py            # Boundary conditions (Dirichlet, Neumann, Robin)
│   ├── bvps.py              # Boundary value problem solvers
│   ├── ivps.py              # Initial value problem / time evolution
│   ├── linalg.py            # Eigenvalue problems and linear algebra
│   ├── cached.py            # Operator caching system
│   └── qm.py                # 1D quantum mechanics (Schrödinger equation)
│
├── toolkit/                 # Utilities and visualization
│   ├── tools.py             # General utilities, sparse matrices, I/O
│   ├── plotting.py          # Matplotlib visualization and animation
│   └── compile_tools.py     # C++ compilation with pybind11
│
└── lowlevelsupport/         # C++ integration
    └── lowlevelcallables.py # Symbolic to C++ code generation
```

## Installation

In the root folder, run
```bash
pip install .
```


## Quick Example

```python
>>> from numiphy.symlib import *
>>> from numiphy.lowlevelsupport import *
>>> import torch
>>> t, x, y = symbols('t, x, y')
>>> F = x**2  + y**2 + Piecewise((t, t < 0), (x, True))
>>> out = Symbol("out")

>>> F.repr("math")
'x**2 + y**2 + (t if (t < 0) else (x))'

>>> F.repr("numpy")
'x**2 + y**2 + numpy.where((t < 0), t, x)'

>>> F.repr("torch", out=out)
'torch.add(torch.add(torch.pow(x, 2, out=out), torch.pow(y, 2, out=out), out=out), torch.where(torch.lt(t, 0, out=out), t, x, out=out), out=out)'

>>> F.lowlevel_repr("double")
'(x*x) + (y*y) + (((t < 0.)) ? t : x)'

>>> f = ScalarLowLevelCallable(F, x, y, t, scalar_type="double")
>>> print(f.code("my_func"))
double my_func(const double& x, const double& y, const double& t, const void*){
    return (x*x) + (y*y) + (((t < 0.)) ? t : x);
}

>>> print(f.to_python_callable().code("my_func", "numpy"))
def my_func(x, y, t)->float:
	return x**2 + y**2 + numpy.where((t < 0), t, x)
```

