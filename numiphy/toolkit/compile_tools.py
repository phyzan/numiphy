import os
import random
import string
import keyword
import sysconfig
import subprocess
import pybind11
import time
from typing import Iterable

def compile(cpp_path, so_dir, module_name, links: Iterable[tuple[str, str]] = (), extra_flags: Iterable[str] = ()):
    ## links[i] = (directory, name), so that -Ldirectory and -lname can be added to the compile command
    if not os.path.exists(cpp_path):
        raise RuntimeError(f"CPP file path does not exist: {cpp_path}")
    if not os.path.exists(so_dir):
        raise RuntimeError(f"Cannot compile at {so_dir}: Path does not exist")

    python_include = sysconfig.get_path("include")
    pybind11_include = pybind11.get_include()
    extension_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    output_file = os.path.join(so_dir, module_name + extension_suffix)

    flags = ['-std=c++20', '-fopenmp', '-fPIC']

    compile_cmd = [
        "g++",
        *flags,
        "-shared",
        f"-I{python_include}",
        f"-I{pybind11_include}",
        *[f"-{flag}" for flag in extra_flags],
        cpp_path,
        "-o", output_file,
    ]

    extra_links = []
    for directory, name in links:
        if directory is not None:
            extra_links.append(f"-L{directory}")
        extra_links.append(f"-l{name}")
    
    compile_cmd += extra_links
    print("Compiling...")
    t1 = time.perf_counter()
    subprocess.check_call(compile_cmd)
    t2 = time.perf_counter()
    print(f"Compiled binary constructed in {t2 - t1:.2f} seconds\n")


def random_module_name(length=8):
    first_char = random.choice(string.ascii_letters + '_')
    other_chars = random.choices(string.ascii_letters + string.digits + '_', k=length - 1)
    name = first_char + ''.join(other_chars)

    while keyword.iskeyword(name):
        name = random_module_name(length)
    return name