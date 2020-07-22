from distutils.core import setup
from pathlib import Path

from setuptools import find_packages

this_file = Path(__file__).resolve()
readme = this_file.parent / 'README.md'

setup(
    name='autokeras',
    version='1.0.4',
    description='AutoML for deep learning',
    package_data={'': ['README.md']},
    long_description=readme.read_text(encoding='utf-8'),
    long_description_content_type='text/markdown',
    author='Data Analytics at Texas A&M (DATA) Lab, Keras Team',
    author_email='jhfjhfj1@gmail.com',
    url='http://autokeras.com',
    download_url='https://github.com/keras-team/autokeras/archive/1.0.4.tar.gz',
    keywords=['AutoML', 'Keras'],
    install_requires=[
        'packaging',
        'tensorflow>=2.2.0',
        'scikit-learn',
        'numpy',
        'pandas',
    ],
    extras_require={
        'tests': ['pytest>=4.4.0',
                  'flake8',
                  'isort',
                  'pytest-xdist',
                  'pytest-cov',
                  'coverage',
                  'typeguard>=2,<2.10.0',
                  'typedapi>=0.2,<0.3'
                  ],
    },
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        'License :: OSI Approved :: MIT License',
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Libraries",
    ],
    license="MIT",
    packages=find_packages(exclude=('tests',)),
)
