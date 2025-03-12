import os
import subprocess

def compile(cpp_path, so_dir, module_name, no_math_errno=False):

    if not os.path.exists(cpp_path):
        raise RuntimeError(f"CPP file path does not exist")
    
    if not os.path.exists(so_dir):
        raise RuntimeError(f"Cannot compile ode at {so_dir}: Path does not exist")
    
    errno_flag = '-fno-math-errno ' if no_math_errno else ''
    compile_comm = f"g++ -O3 -Wall -shared -std=c++20 -fopenmp {errno_flag} -I/usr/include/python3.12 -I/usr/include/pybind11 -fPIC $(python3 -m pybind11 --includes) {cpp_path} -o {os.path.join(so_dir, module_name)}$(python3-config --extension-suffix)  -lmpfr -lgmp"
    print('Compiling ODE...')

    subprocess.check_call(compile_comm, shell=True)
    print('Done')