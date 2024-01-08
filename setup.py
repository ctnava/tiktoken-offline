from setuptools import setup, find_packages

setup(
    name="tiktoken-offline",
    version="1.0.0",
    packages=find_packages(),
    package_data={'tiktoken_offline': ['data/*.tiktoken','data/*.json','data/*.bpe']},  
    python_requires = '>=3.8',
)
