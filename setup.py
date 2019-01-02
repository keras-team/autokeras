from distutils.core import setup
from setuptools import find_packages

setup(
    name='autokeras',
    packages=find_packages(exclude=('tests',)),
    install_requires=['scipy==1.1.0',
                      'torch==0.4.1',
                      'torchvision==0.2.1',
                      'numpy==1.14.5',
                      'keras==2.2.2',
                      'scikit-learn==0.20.1',
                      'scikit-image==0.13.1',
                      'tqdm==4.25.0',
                      'tensorflow==1.10.0',
                      'imageio==2.4.1',
                      'requests==2.20.1',
                      'GPUtil==1.3.0',
                      'lightgbm==2.2.2',
                      'pandas==0.23.4',
                      'opencv-python==3.4.4.19'],
    version='0.3.5',
    description='AutoML for deep learning',
    author='DATA Lab at Texas A&M University',
    author_email='jhfjhfj1@gmail.com',
    url='http://autokeras.com',
    download_url='https://github.com/jhfjhfj1/autokeras/archive/0.3.5.tar.gz',
    keywords=['AutoML', 'keras'],
    classifiers=[]
)
