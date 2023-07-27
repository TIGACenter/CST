from setuptools import find_packages, setup
from pkg_resources import parse_requirements

REQUIREMENTS_FILE = 'requirements.txt'
install_requires = [str(r) for r in parse_requirements(open(REQUIREMENTS_FILE, 'rt'))]


setup(
    name='cp_toolbox',
    packages=find_packages(),
    version='0.1.0',
    description='CNN Stability Training',
    author='Felipe Miranda',
    licence='MIT',
    install_requires = install_requires
)