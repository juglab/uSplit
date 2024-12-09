from __future__ import absolute_import

from os import path

from setuptools import find_packages, setup

_dir = path.abspath(path.dirname(__file__))

with open(path.join(_dir, 'usplit', 'version.py')) as f:
    exec(f.read())

with open(path.join(_dir, 'README.md')) as f:
    long_description = f.read()

setup(
    name='uSplit',
    version=__version__,
    description='uSplit',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/juglab/uSplit/',
    author='Ashesh, Alexander Krull, Moises Di Sante, Francesco Silvio Pasqualini, Florian Jug',
    author_email=
    'ashesh276@gmail.com, a.f.f.krull@bham.ac.uk,moises.disante@unipv.it,francesco.pasqualini@unipv.it, florian.jug@fht.org',
    license='BSD 3-Clause License',
    packages=["usplit"],
    project_urls={
        'Repository': 'https://github.com/juglab/uSplit/',
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3.9',
    ],
    install_requires=[],
)
