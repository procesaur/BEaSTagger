from setuptools import setup

setup(
    name='BEaSTagger',
    version='0.99',
    packages=[],
    install_requires=[
        'classla~=1.0.1',
        'torch~=1.8.1',
        'spacy~=3.0.6',
        'tqdm~=4.44.1',
        'numpy~=1.18.2',
        'pandas~=1.0.3',
        'sklearn~=0.0',
        'scikit-learn~=0.22.2.post1',
        'setuptools~=49.2.1'
    ],
    url='https://github.com/procesaur/BEaSTagger',
    license='GPL',
    author='Mihailo',
    author_email='procesaur@gmail.com',
    description='POS tagging tool based on Bidirectional, Expandable and Stacked architecture.'
)
