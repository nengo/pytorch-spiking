# Automatically generated by nengo-bones, do not edit this file directly

import io
import pathlib
import runpy

try:
    from setuptools import find_packages, setup
except ImportError:
    raise ImportError(
        "'setuptools' is required but not installed. To install it, "
        "follow the instructions at "
        "https://pip.pypa.io/en/stable/installing/#installing-with-get-pip-py"
    )


def read(*filenames, **kwargs):
    encoding = kwargs.get("encoding", "utf-8")
    sep = kwargs.get("sep", "\n")
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)


root = pathlib.Path(__file__).parent
version = runpy.run_path(str(root / "pytorch_spiking" / "version.py"))["version"]

install_req = [
    "numpy>=1.16.0",
    "torch>=1.0.0",
]
docs_req = [
    "jupyter>=1.0.0",
    "matplotlib>=2.0.0",
    "nbsphinx>=0.3.5",
    "nengo-sphinx-theme>=1.2.1",
    "numpydoc>=0.6.0",
    "sphinx>=3.0.0",
    "torchvision>=0.7.0",
]
optional_req = []
tests_req = [
    "pylint>=1.9.2",
    "pytest>=3.6.0",
    "pytest-allclose>=1.0.0",
    "pytest-cov>=2.6.0",
    "pytest-rng>=1.0.0",
    "pytest-xdist>=1.16.0",
]

setup(
    name="pytorch-spiking",
    version=version,
    author="Applied Brain Research",
    author_email="info@appliedbrainresearch.com",
    packages=find_packages(),
    url="https://www.nengo.ai/pytorch-spiking",
    include_package_data=True,
    license="Free for non-commercial use",
    description="Spiking neuron integration for PyTorch",
    long_description=read("README.rst", "CHANGES.rst"),
    zip_safe=False,
    install_requires=install_req,
    extras_require={
        "all": docs_req + optional_req + tests_req,
        "docs": docs_req,
        "optional": optional_req,
        "tests": tests_req,
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: Free for non-commercial use",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
