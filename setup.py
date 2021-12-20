import numpy
from setuptools import setup, find_packages
from setuptools import Extension
from setuptools.command.build_ext import build_ext as _build_ext
from codecs import open
from os import path
from distutils.extension import Extension
from Cython.Build import cythonize
import os

# see https://stackoverflow.com/a/21621689/1862861 for why this is here
class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        self.include_dirs.append(numpy.get_include())

with open("requirements.txt") as requires_file:
    requirements = requires_file.read().split("\n")

setup(
    name = 'starcluster',
    use_scm_version=True,
    description = 'Search for star clusters using DPGMM',
    author = 'Daniele Sanfratello, Stefano Rinaldi, Walter Del Pozzo',
    author_email = 'd.sanfratello@studenti.unipi.it, stefano.rinaldi@phd.unipi.it, walter.delpozzo@unipi.it',
    url = 'https://github.com/d-sanfratello/starcluster',
    python_requires = '>=3.7',
    packages = ['starcluster'],
    install_requires=requirements,
    include_dirs = [numpy.get_include()],
    entry_points = {},
    )
