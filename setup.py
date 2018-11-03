from distutils.core import setup
import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name='SSIM-PIL',
    version='1.0.1',
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
#C:\Python37\python.exe -m setup.py sdist bdist_wheel
#C:\Python37\python.exe -m twine upload dist/*
