from setuptools import setup
import os
import sys

data_dir = os.path.join(sys.prefix, "data")
setup(
    name = 'sumpy',
    packages = ['sumpy', 'sumpy.system', 'sumpy.annotators'],
    version = '0.0.1',
    description = 'SUMPY: an automatic text summarization library',
    author='Chris Kedzie',
    author_email='kedzie@cs.columbia.edu',
    url='https://github.com/kedz/sumpy',
    install_requires=[
        'nltk', 'numpy', 'scipy', 'scikit-learn', 'pandas',
        'networkx',
    ],
    include_package_data=True,
    package_data={
        'sumpy': [os.path.join(data_dir, 'smart_common_words.txt.gz'),
                  os.path.join(data_dir, 'mead_example_docs', '41.docsent'),   
                  os.path.join(data_dir, 'mead_example_docs', '81.docsent'),
                  os.path.join(data_dir, 'mead_example_docs', '87.docsent'),
                 ]},

)
                    
