from setuptools import setup
from setuptools import find_packages, setup, Command

with open('requirements.txt', 'r') as f:
    requires = f.read().splitlines()

setup(
name='minimal_fsdp',
version = 0.0,
install_requirements=requires,
packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),


)