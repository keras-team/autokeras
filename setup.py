from distutils.core import setup
from pathlib import Path

from setuptools import find_packages

this_file = Path(__file__).resolve()
readme = this_file.parent / 'README.md'

setup(
    name='autokeras',
    version='1.0.0b0',
    description='AutoML for deep learning',
    package_data={'': ['README.md']},
    long_description=readme.read_text(encoding='utf-8'),
    long_description_content_type='text/markdown',
    author='Data Analytics at Texas A&M (DATA) Lab, Keras Team',
    author_email='jhfjhfj1@gmail.com',
    url='http://autokeras.com',
    download_url='https://github.com/keras-team/autokeras/archive/1.0.0b0.tar.gz',
    keywords=['AutoML', 'keras'],
    # TODO: Do not install tensorflow if tensorflow-gpu is installed.
    install_requires=[
        'packaging',
        'keras-tuner>=1.0.0',
        'scikit-learn',
        'numpy',
        'pandas',
        'lightgbm',
    ],
    extras_require={
        'tests': ['pytest>=4.4.0',
                  'flake8',
                  'pytest-xdist',
                  'pytest-cov',
                  # can be removed once coveralls is compatible with
                  # coverage 5.0
                  'coverage==5.0.2'
                  ],
    },
    packages=find_packages(exclude=('tests',)),
)
