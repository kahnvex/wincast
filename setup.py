from setuptools import setup, find_packages


install_requires = [
    'dill==0.2.5',
    'easydict==1.6',
    'jsonpickle==0.9.3',
    'numpy==1.11.2',
    'pandas==0.19.1',
    'scikit-learn==0.18.1',
    'scipy==0.18.1',
    'tabulate==0.7.7',
]

setup(
    name="wincast",
    version='0.0.10',
    url='https://github.com/kahnjw/wincast',
    author_email='jarrod.kahn+wincast@gmail.com',
    license='MIT',
    packages=find_packages(exclude=['tests', 'tests.*']),
    install_requires=install_requires,
    package_data={
        'wincast': ['models/wincast.clf.pkl', 'models/wincast.scaler.pkl']
    }
)
