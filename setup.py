import os
from setuptools import setup, find_packages


def read_file(filename):
    with open(os.path.join(os.path.dirname(__file__), filename), 'r', encoding='utf-8') as file:
        return file.read()

setup(
    name='chatgpt_block',
    version='0.1.4.2',
    packages=find_packages(),
    install_requires=[
        'openai>=0.27.2',
        'requests>=2.28.2',
        'tiktoken>=0.3.2'
    ],
    long_description=read_file('README.md'),
    long_description_content_type='text/markdown'
)
