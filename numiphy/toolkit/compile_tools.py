import os
import subprocess
import random
import string
import keyword
import sysconfig
import sys


def compile(cpp_path, so_dir, module_name, no_math_errno=True, no_math_trap=False, fast_math=False):
    if not os.path.exists(cpp_path):
        raise RuntimeError(f"CPP file path does not exist: {cpp_path}")
    
    if not os.path.exists(so_dir):
        raise RuntimeError(f"Cannot compile at {so_dir}: Path does not exist")
    
    # Compiler flags
    errno_flag = ' -fno-math-errno' if no_math_errno else ''
    no_trap_flag = ' -fno-trapping-math' if no_math_trap else ''
    fm_flag = ' -ffast-math' if fast_math else ''
    
    # Python paths
    python_include = sysconfig.get_path("include")
    extension_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    python_exec = sys.executable
    
    # pybind11 includes
    try:
        pybind11_includes = subprocess.check_output(
            [python_exec, "-m", "pybind11", "--includes"],
            text=True
        ).strip()
    except subprocess.CalledProcessError as e:
        raise RuntimeError("Failed to get pybind11 include paths") from e

    # Output file
    output_file = os.path.join(so_dir, module_name + extension_suffix)
    
    # Compile command
    compile_comm = (
        f"g++ -O3{errno_flag}{no_trap_flag}{fm_flag} -Wall -shared -march=x86-64 -std=c++20 -fopenmp "
        f"-I{python_include} {pybind11_includes} -fPIC {cpp_path} "
        f"-o {output_file} -lmpfr -lgmp"
    )
    
    print("Compiling...")
    subprocess.check_call(compile_comm, shell=True)
    print("Done")


def random_module_name(length=8):
    first_char = random.choice(string.ascii_letters + '_')
    other_chars = random.choices(string.ascii_letters + string.digits + '_', k=length - 1)
    name = first_char + ''.join(other_chars)

    while keyword.iskeyword(name):
        name = random_module_name(length)
    return name    