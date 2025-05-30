import os
import subprocess
import random
import string
import keyword

def compile(cpp_path, so_dir, module_name, no_math_errno=True, no_math_trap=False, fast_math=False):

    if not os.path.exists(cpp_path):
        raise RuntimeError(f"CPP file path does not exist: {cpp_path}")
    
    if not os.path.exists(so_dir):
        raise RuntimeError(f"Cannot compile ode at {so_dir}: Path does not exist")
    
    errno_flag = ' -fno-math-errno' if no_math_errno else ''
    no_trap_flag = ' -fno-trapping-math' if no_math_trap else ''
    fm_flag = ' -ffast-math' if fast_math else ''
    compile_comm = f"g++ -O3{errno_flag}{no_trap_flag}{fm_flag} -Wall -shared -march=x86-64 -std=c++20 -fopenmp -I/usr/include/python3.12 -I/usr/include/pybind11 -fPIC $(python3 -m pybind11 --includes) {cpp_path} -o {os.path.join(so_dir, module_name)}$(python3-config --extension-suffix)  -lmpfr -lgmp"
    print('Compiling ODE...')

    subprocess.check_call(compile_comm, shell=True)
    print('Done')


def random_module_name(length=8):
    first_char = random.choice(string.ascii_letters + '_')
    other_chars = random.choices(string.ascii_letters + string.digits + '_', k=length - 1)
    name = first_char + ''.join(other_chars)

    while keyword.iskeyword(name):
        name = random_module_name(length)
    return name