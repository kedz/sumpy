from setuptools import setup

setup(
    name = 'sumpy',
    packages = ['sumpy'],
    version = '0.0.1',
    description = 'SUMPY: an automatic text summarization library',
    author='Chris Kedzie',
    author_email='kedzie@cs.columbia.edu',
    url='https://github.com/kedz/sumpy',
    install_requires=[
        'nltk', 'numpy', 'scipy', 'scikit-learn',
    ],

)
                    
