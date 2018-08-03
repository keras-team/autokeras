from distutils.core import setup

setup(
    name='autokeras',
    packages=['autokeras'],  # this must be the same as the name above
    install_requires=['torch==0.4.0', 'torchvision==0.2.1', 'numpy==1.14.5', 'keras', 'scikit-learn==0.19.1', 'tensorflow'],
    version='0.2.0',
    description='Automated Machine Learning with Keras',
    author='Haifeng Jin',
    author_email='jhfjhfj1@gmail.com',
    url='http://autokeras.com',
    download_url='https://github.com/jhfjhfj1/autokeras/archive/0.1.1.tar.gz',
    keywords=['automl'],  # arbitrary keywords
    classifiers=[]
)
