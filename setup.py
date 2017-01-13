from setuptools import setup, find_packages

install_requires = [
    'dill==0.2.5',
    'easydict==1.6',
    'h5py==2.6.0',
    'jsonpickle==0.9.3',
    'Keras==1.2.0',
    'nflgame==1.2.20',
    'numpy==1.11.2',
    'pandas==0.19.1',
    'scikit-learn==0.18.1',
    'scipy==0.18.1',
    'tensorflow==0.12.0rc1',
    'Theano==0.8.2',
]

with open('README.md', 'r', 'utf-8') as f:
    readme = f.read()

setup(
    name="wincast",
    version=__version__,
    url='https://github.com/kahnjw/wincast',
    author_email='thomas.welfley+djproxy@gmail.com',
    long_description=readme,
    license='MIT',
    packages=find_packages(exclude=['tests', 'tests.*']),
    install_requires=install_requires,
)
