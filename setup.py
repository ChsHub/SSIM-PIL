import setuptools
from distutils.core import setup
from SSIM_PIL import __version__

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name='SSIM-PIL',
    version=__version__,
    description='Structural similarity',
    author='ChsHub',
    author_email='christian1193@web.com',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ChsHub/SSIM-PIL",
    packages=['SSIM_PIL'],
    license='MIT License',
    classifiers=['Programming Language :: Python :: 3']
)
# C:\Python37\python.exe setup.py sdist bdist_wheel
# C:\Python37\python.exe -m twine upload dist/*
# https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyopencl