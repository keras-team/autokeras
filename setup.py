from distutils.core import setup

setup(
    name='autokeras',
    packages=['autokeras'],  # this must be the same as the name above
    requires=['pytest', 'numpy'],
    version='0.0.1',
    description='Automated Machine Learning with Keras',
    author='Haifeng Jin',
    author_email='jhfjhfj1@gmail.com',
    url='https://github.com/jhfjhfj1/auto-keras',
    download_url='https://github.com/jhfjhfj1/autokeras/archive/0.0.1.tar.gz',
    keywords=['automl'],  # arbitrary keywords
    classifiers=[]
)
