# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import, unicode_literals

import setuptools # required to allow for use of python setup.py develop, may also be important for cython/compiling if it is used

from distutils.core import setup


setup(
    name = 'eeg-hdfstorage',
    version='0.0.3',
    description="""eeg storage in hdf5 + related functions""",
    author="""Chris Lee-Messer""",
    url="https://github.com/cleemesser/eeg-hdfstorage",
    # download_url="",
    classifiers=['Topic :: Science :: EEG'],
    packages=['eeghdf'],
    # package_data={}
    # data_files=[],
    # scripts = [],
    )
