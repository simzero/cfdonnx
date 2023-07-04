from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='cfdonnx',
    version='2.0.1',
    description='Converting ML-CFD models to ONNX',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Carlos Pena Monferrer',
    keywords='cfd ml onnx',
    url='https://github.com/simzero/cfdonnx',
    packages=['cfdonnx', 'cfdonnx.models', 'cfdonnx.models.torch'],
    package_dir={'': 'src'},
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "torch==2.0.0",
        "torchvision==0.15.1",
        "onnx==1.13.1",
        "onnxruntime==1.14.1",
    ],
)
