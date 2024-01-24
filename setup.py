#!/usr/bin/env python
from setuptools import setup, find_packages, Extension
from os.path import join

setup(
    name="tiktoken-offline",
    version="1.0.0",
    packages=find_packages(),
    package_data={'tiktoken_offline': ['data/*.tiktoken','data/*.json','data/*.bpe']},  
    ext_modules=[Extension('tiktoken_offline.regex_3._regex', [join('tiktoken_offline/regex_3', '_regex.c'),
      join('tiktoken_offline/regex_3', '_regex_unicode.c')])],
    python_requires = '>=3.8',
)
