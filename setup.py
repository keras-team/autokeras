from distutils.core import setup

setup(
    name='autokeras',
    packages=['autokeras'],  # this must be the same as the name above
    install_requires=['pytest', 'numpy', 'keras', 'scikit-learn', 'tensorflow'],
    version='0.0.2',
    description='Automated Machine Learning with Keras',
    author='Haifeng Jin',
    author_email='jhfjhfj1@gmail.com',
    url='https://github.com/jhfjhfj1/autokeras',
    download_url='https://github.com/jhfjhfj1/autokeras/archive/0.0.2.tar.gz',
    keywords=['automl'],  # arbitrary keywords
    classifiers=[]
)
