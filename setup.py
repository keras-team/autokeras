from distutils.core import setup

from setuptools import find_packages

setup(
    name='autokeras',
    version='1.0.0a0',
    description='AutoML for deep learning',
    author='Data Analytics at Texas A&M (DATA) Lab, Keras Team',
    author_email='jhfjhfj1@gmail.com',
    url='http://autokeras.com',
    download_url='https://github.com/keras-team/autokeras/archive/1.0.0a0.tar.gz',
    keywords=['AutoML', 'keras'],
    install_requires=[
        'tensorflow>=2.0.0b1',
        'keras-tuner',
        'scikit-learn',
        'numpy',
        'lightgbm',
    ],
    extras_require={
        'tests': ['pytest>=4.4.0',
                  'flake8',
                  'pytest-xdist',
                  'pytest-cov',
                  ],
    },
    packages=find_packages(exclude=('tests',)),
)
