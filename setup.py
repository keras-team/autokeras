from distutils.core import setup
from setuptools import find_packages
from pathlib import Path

this_file = Path(__file__).resolve()
readme = this_file.parent / 'README.md'

setup(
    name='autokeras',
    version='1.0.0a0',
    description='AutoML for deep learning',
    package_data={'': ['README.md']},
    long_description=readme.read_text(encoding='utf-8'),
    long_description_content_type='text/markdown',
    author='Data Analytics at Texas A&M (DATA) Lab, Keras Team',
    author_email='jhfjhfj1@gmail.com',
    url='http://autokeras.com',
    download_url='https://github.com/keras-team/autokeras/archive/1.0.0a0.tar.gz',
    keywords=['AutoML', 'keras'],
    install_requires=[
        'tensorflow',
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
                  ],
    },
    packages=find_packages(exclude=('tests',)),
)
