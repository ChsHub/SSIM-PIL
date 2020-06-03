import setuptools
from distutils.core import setup
from SSIM_PIL import __version__

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name='SSIM-PIL',
    version=__version__,
    description=long_description.split('\n')[1],
    author='ChsHub',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ChsHub/SSIM-PIL",
    packages=['SSIM_PIL'],
    license='MIT License',
    classifiers=['Programming Language :: Python :: 3']
)

# python setup.py sdist bdist_wheel
# python -m twine upload dist/*
# https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyopencl