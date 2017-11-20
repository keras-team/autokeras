from distutils.core import setup

setup(
    name='autokeras',
    packages=['autokeras'],  # this must be the same as the name above
    requires=['pytest', 'numpy'],
    version='alpha v0.1',
    description='Automated Machine Learning with Keras',
    author='Haifeng Jin',
    author_email='jhfjhfj1@gmail.com',
    url='https://github.com/jhfjhfj1/auto-keras',  # use the URL to the github repo
    download_url='https://github.com/jhfjhfj1/auto-keras/archive/0.1.tar.gz',  # I'll explain this in a second
    keywords=['automl'],  # arbitrary keywords
    classifiers=[]
)
