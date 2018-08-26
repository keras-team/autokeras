from distutils.core import setup

setup(
    name='autokeras',
    packages=['autokeras'],  # this must be the same as the name above
    install_requires=['torch==0.4.1', 'torchvision==0.2.1', 'numpy>=1.15.1', 'keras==2.2.2', 'scikit-learn==0.19.1',
                      'tensorflow>=1.10.1'],
    version='0.2.7',
    description='AutoML for deep learning',
    author='Haifeng Jin',
    author_email='jhfjhfj1@gmail.com',
    url='http://autokeras.com',
    download_url='https://github.com/jhfjhfj1/autokeras/archive/0.2.7.tar.gz',
    keywords=['automl'],  # arbitrary keywords
    classifiers=[]
)
