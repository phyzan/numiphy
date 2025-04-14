import os
import sys
from setuptools import setup, find_packages
from setuptools.command.install import install
import tempfile
import importlib
import subprocess

# Check if the C++ headers are installed

class CustomInstall(install):
    def run(self):
        print('Installing c++ headers...')
        subprocess.check_call('sudo apt install libmpfrc++-dev', shell=True)
        subprocess.check_call('sudo apt install libeigen3-dev', shell=True)
        subprocess.check_call('git clone https://github.com/phyzan/odepack && cd odepack && chmod +x install.sh && sudo ./install.sh && cd ..', shell=True)
        package_dir = self.build_lib
        target_dir = os.path.join(package_dir, "numiphy", "odesolvers")
        odepack_name = "odepack"
        code = f'#include <odepack/pyode.hpp>\n\nPYBIND11_MODULE({odepack_name}, m)'+'{\n\tdefine_ode_module<double, vec<double>>(m);\n}'

        tools_path = os.path.join(package_dir, "numiphy", "toolkit", "compile_tools.py")
        module_name = os.path.splitext(os.path.basename(tools_path))[0]  # e.g., "my_script"

        # Load the module dynamically
        spec = importlib.util.spec_from_file_location(module_name, tools_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        # Now import the specific function
        compile = module.compile

        with tempfile.TemporaryDirectory() as temp_dir:
            cpp_temp_path = os.path.join(temp_dir, f"{odepack_name}.cpp")
            with open(cpp_temp_path, "w") as f:
                f.write(code)
            compile(cpp_temp_path, target_dir, odepack_name, no_math_errno=True)
        super().run()

odepack_installed = os.path.exists("/usr/include/odepack")  # Change path if needed

if not odepack_installed:
    print("\nERROR: The 'odepack' C++ library is not installed!")
    print("Please install it manually:")
    print("https://github.com/phyzan/odepack\n")
    sys.exit(1)

setup(
    name="numiphy",
    version="1.0",
    python_requires=">=3.12",
    packages=find_packages(),
    package_data={
        "numiphy": ["odesolvers/*.so"],
    },
    include_package_data=True,
    install_requires=[
        "numpy==2.1.2",
        "scipy==1.14.1",
        "matplotlib==3.9.2",
        "pybind11==2.13.6",
        "joblib==1.4.2"
    ],
    cmdclass={"install": CustomInstall},
)
