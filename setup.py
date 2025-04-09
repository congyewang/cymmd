import os
import sys

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup

# Automatically detect if it is in the sdist package
is_sdist = False
if os.path.exists("PKG-INFO"):
    is_sdist = True

# Determine the actual location of fast_mmd.pyx
possible_paths = [
    "src/cymmd/cymmd.pyx",
]

pyx_path = None
for path in possible_paths:
    if os.path.exists(path):
        pyx_path = path
        break

if pyx_path is None:
    # If installing through the UV source package, try to locate
    for root, _, files in os.walk("."):
        if "cymmd.pyx" in files:
            pyx_path = os.path.join(root, "cymmd.pyx")
            break

if pyx_path is None:
    # If the .pyx file cannot be found but there is a .c file, use the precompiled C file.
    for root, _, files in os.walk("."):
        if "cymmd.c" in files:
            c_path = os.path.join(root, "cymmd.c")
            extensions = [
                Extension(
                    "cymmd.cymmd",
                    [c_path],
                    include_dirs=[np.get_include()],
                    extra_compile_args=["-O3"],
                )
            ]
            break
    else:
        raise FileNotFoundError(
            "Unable to locate the cymmd.pyx or cymmd.c file, please confirm the file location."
        )
else:
    print(f"Find the Cython file: {pyx_path}")
    extensions = [
        Extension(
            "cymmd.cymmd",
            [pyx_path],
            include_dirs=[np.get_include()],
            extra_compile_args=["-O3"],
        )
    ]

setup(
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": 3,
            "boundscheck": False,
            "wraparound": False,
        },
    ),
    include_package_data=True,
)
