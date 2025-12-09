import os
import random
import string
import keyword
import sysconfig
import subprocess
import pybind11


def compile(cpp_path, so_dir, module_name,
            no_math_errno=True, no_math_trap=True, fast_math=False):
    if not os.path.exists(cpp_path):
        raise RuntimeError(f"CPP file path does not exist: {cpp_path}")
    if not os.path.exists(so_dir):
        raise RuntimeError(f"Cannot compile at {so_dir}: Path does not exist")

    python_include = sysconfig.get_path("include")
    pybind11_include = pybind11.get_include()
    extension_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    output_file = os.path.join(so_dir, module_name + extension_suffix)

    flags = ['-O3', '-Wall', '-std=c++20', '-fopenmp', '-fPIC']
    if no_math_errno:
        flags.append('-fno-math-errno')
    if no_math_trap:
        flags.append('-fno-trapping-math')
    if fast_math:
        flags.append('-ffast-math')

    compile_cmd = [
        "clang++",
        *flags,
        "-shared",
        f"-I{python_include}",
        f"-I{pybind11_include}",
        cpp_path,
        "-o", output_file,
        "-lmpfr", "-lgmp"
    ]

    print("Compiling...")
    subprocess.check_call(compile_cmd)
    print(f"Compilation done: {output_file}")


def random_module_name(length=8):
    first_char = random.choice(string.ascii_letters + '_')
    other_chars = random.choices(string.ascii_letters + string.digits + '_', k=length - 1)
    name = first_char + ''.join(other_chars)

    while keyword.iskeyword(name):
        name = random_module_name(length)
    return name