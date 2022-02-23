# -*- coding: utf-8 -*-
"""
    Setup file for copa_map.
    Use setup.cfg to configure your project.

    This file was generated with PyScaffold 3.3.1.
    PyScaffold helps you to put up the scaffold of your new Python project.
    Learn more under: https://pyscaffold.org/
"""
import sys
from pkg_resources import VersionConflict, require
from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension(
        "brescount",
        ["src/copa_map/util/brescount.pyx"],
        extra_compile_args=["-O3", "-ffast-math", "-march=native", "-fopenmp"],
        extra_link_args=['-fopenmp'],
        include_dirs=[numpy.get_include()]
    )
]

try:
    require("setuptools>=38.3")
except VersionConflict:
    print("Error: version of setuptools is too old (<38.3)!")
    sys.exit(1)


if __name__ == "__main__":
    setup(use_pyscaffold=True, ext_modules=cythonize(ext_modules))
