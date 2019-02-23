from distutils.core import setup
from setuptools import find_packages

setup(
    name='autokeras',
    packages=find_packages(exclude=('tests',)),
    install_requires=['scipy==1.2.0',
                      'torch==1.0.1.post2',
                      'torchvision==0.2.1',
                      'numpy==1.15.4',
                      'scikit-learn==0.20.2',
                      'scikit-image==0.14.2',
                      'tqdm==4.31.0',
                      'imageio==2.5.0',
                      'requests==2.21.0',
                      'librosa==0.6.2',
                      'numba',
                      'inflect',
                      'unidecode',
                      'nltk==3.3',
                      'lws==1.2',
                      'opencv-python==4.0.0.21',
                      'boto3'],
    version='0.3.7',
    description='AutoML for deep learning',
    author='DATA Lab at Texas A&M University',
    author_email='jhfjhfj1@gmail.com',
    url='http://autokeras.com',
    download_url='https://github.com/jhfjhfj1/autokeras/archive/0.3.7.tar.gz',
    keywords=['AutoML', 'keras'],
    classifiers=[]
)
