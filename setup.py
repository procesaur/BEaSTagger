from setuptools import setup
import os


def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths


setup(
    name='BEaSTagger',
    version='1.00',
    packages=['beast'],
    install_requires=[
        'torch>=1.8.1',
        'spacy~=3.0.6',
        'tqdm>=4.44.1',
        'numpy>=1.19.5',
        'pandas>=1.0.3',
        'sklearn>=0.0',
        'scikit-learn~=0.22.2.post1',
        'setuptools>=49.2.1'
    ],
    package_data={'beast': ['data\\default\\pos.pt']},
    include_package_data=True,
    url='https://github.com/procesaur/BEaSTagger',
    license='GPL',
    author='Mihailo Škorić',
    author_email='procesaur@gmail.com',
    description='POS tagging tool based on Bidirectional, Expandable and Stacked architecture.'
)
