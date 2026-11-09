"""Use the same explicit boundary for wheels and strict editable installs."""
from setuptools import setup
from _core_build import CoreBuildPy

setup(cmdclass={"build_py": CoreBuildPy})
