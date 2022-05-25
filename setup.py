from distutils.core import setup
from setuptools import find_packages

setup(
    name='rambo',
    packages=find_packages(),
    version='0.0.1',
    description='Robust Adversarial Model-Based Offline Reinforcement Learning',
    long_description=open('./README.md').read(),
    author='Marc Rigter',
    author_email='mrigter@robots.ox.ac.uk',
    entry_points={
        'console_scripts': (
            'rambo=rambo.scripts.console_scripts:main'
        )
    },
    requires=(),
    zip_safe=True,
    license='MIT'
)
