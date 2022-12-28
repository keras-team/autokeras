from distutils.core import setup
from pathlib import Path

from setuptools import find_packages

this_file = Path(__file__).resolve()
readme = this_file.parent / "README.md"

setup(
    name="autokeras",
    description="AutoML for deep learning",
    package_data={"": ["README.md"]},
    long_description=readme.read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="DATA Lab, Keras Team",
    author_email="jhfjhfj1@gmail.com",
    url="http://autokeras.com",
    keywords=["AutoML", "Keras"],
    install_requires=[
        "packaging",
        "tensorflow>=2.8.0",
        "keras-tuner>=1.1.0",
        "keras-nlp>=0.4.0",
        "pandas",
    ],
    extras_require={
        "tests": [
            "pytest>=4.4.0",
            "flake8",
            "black[jupyter]==22.12.0",
            "isort",
            "pytest-xdist",
            "pytest-cov",
            "coverage",
            "typedapi>=0.2,<0.3",
            "scikit-learn",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Libraries",
    ],
    license="Apache License 2.0",
    packages=find_packages(exclude=("*test*",)),
)
