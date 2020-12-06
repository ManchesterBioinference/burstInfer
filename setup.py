# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 03:22:19 2020

@author: Jon
"""
from distutils.core import setup

DESCRIPTION = "burstInfer: inferring single-cell transcriptional parameters"
LONG_DESCRIPTION = DESCRIPTION
NAME = "burstInfer"
AUTHOR = "Jonathan R. Bowles"
AUTHOR_EMAIL = "email placeholder"
MAINTAINER = "Jonathan R. Bowles"
MAINTAINER_EMAIL = "email placeholder"
DOWNLOAD_URL = 'https://github.com/ManchesterBioinference/burstInfer'
LICENSE = 'MIT'

VERSION = '0.1'

requirements = [
]

setup(name=NAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      url=DOWNLOAD_URL,
      download_url=DOWNLOAD_URL,
      license=LICENSE,
      packages=['burstInfer'],
      package_data={},
	install_requires=requirements
      )
