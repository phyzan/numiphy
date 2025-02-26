import os
import sys
from setuptools import setup, find_packages

# Check if the C++ headers are installed
odepack_installed = os.path.exists("/usr/include/odepack")  # Change path if needed

if not odepack_installed:
    print("\nERROR: The 'odepack' C++ library is not installed!")
    print("Please install it manually by running:")
    print("\n    git clone https://github.com/phyzan/odepack && cd odepack && chmod +x install.sh\n")
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
)
