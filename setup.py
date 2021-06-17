# Copyright (c) Facebook, Inc. and its affiliates.

from setuptools import find_packages, setup

setup(
    name='svg',
    version='0.0.1',
    description="Stochastic value gradient for continuous control",
    platforms=['any'],
    url="https://github.com/facebookresearch/svg",
    packages=find_packages(),
    install_requires=[
        'numpy>=1<2',
        'hydra-core==1.0',
    ]
)
