from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='cfdonnx',
    version='0.1.0',
    description='Converting ML-CFD models to ONNX',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Carlos Pena Monferrer',
    url='https://github.com/simzero/cfdonnx',
    packages=['cfdonnx', 'cfdonnx.models', 'cfdonnx.models.torch'],
    package_dir={'': 'src'},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
    ],
)
