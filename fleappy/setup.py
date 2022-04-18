#!/usr/bin/env python

from setuptools import setup, find_packages

version = '0.2.1'

required = open('requirements.txt').read().split('\n')

setup(
    name='fleappy',
    version=version,
    description='A modular framework for the analysis of experiment data neuroscience experiments.',
    author='Jeremy T. Chang',
    author_email='jeremy.chang@mpfi.org',
    url='https://github.com/jtchang/fleappy',
    packages=['fleappy.notifications', 'fleappy.tiffread', 'fleappy.imgregistration'],
    install_requires=required,
    long_description='See ' + 'https://github.com/jtchang/fleappy',
    license='MIT',
    python_requires='~=3.7',
    dependency_links=['https://download.pytorch.org/whl/torch_stable.html']
)
