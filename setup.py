# -*- coding: utf-8 -*-
# python3 setup.py install

from setuptools import setup, find_packages


AUTHOR = 'Frank Longford'
AUTHOR_EMAIL = 'f.longford@soton.ac.uk'
URL = 'https://github.com/franklongford/ColECM'
PLATFORMS = ['Linux', 'Unix', 'Mac OS X']
PACKAGE_DIR = {'ColECM/src': '.'}

with open('README.rst') as f:
	readme = f.read()

with open('LICENSE') as f:
	LICENSE = f.read()

with open('requirements.txt') as f:
	requirements = f.read()

setup(
	name='ColECM',
	description='Collagen Extracellular Matrix Simulation',
	long_description=readme,
	author=AUTHOR,
	author_email=AUTHOR_EMAIL,
	url=URL,
	license=LICENSE,
	platforms=PLATFORMS,
	package_dir=PACKAGE_DIR,
	packages=find_packages(exclude=('tests', 'docs')),
	python_requires='>=3.0',
	install_requires=requirements,
)
