from distutils.core import setup
from setuptools import find_packages

setup(
    name='autokeras',
    packages=find_packages(exclude=('tests',)),
    install_requires=[
        'tensorflow==2.0.0b0',
        'scikit-learn==0.20.2',
    ],
    extras_require={
        'tests': [
            'pytest',
            'pytest-pep8',
            'pytest-xdist',
            'pytest-cov',
        ],
    },
    version='1.0.0',
    description='AutoML for deep learning',
    author='Data Analytics at Texas A&M (DATA) Lab, Keras Team',
    author_email='jhfjhfj1@gmail.com',
    url='http://autokeras.com',
    download_url='https://github.com/keras-team/autokeras/archive/1.0.0.tar.gz',
    keywords=['AutoML', 'keras'],
    classifiers=[]
)
