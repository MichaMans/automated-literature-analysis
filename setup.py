# coding=utf-8
import setuptools

setuptools.setup(
    name="litstudy",
    version="0.1",
    description="Test",
    url="",
    author="",
    author_email="",
    license="MIT License",
    packages=setuptools.find_packages(),
    install_requires=[
        "jupyter",
        "gensim",
        "pybliometrics==2.2",
        "numpy",
        "pandas",
        "matplotlib",
        "scikit-learn==0.20.0",
        "seaborn",
        "wordcloud",
        "tqdm",
        "requests",
        "bibtexparser",
        "networkx>=2.3",
        "python-Levenshtein==0.12.0",
        "PyYAML==5.4",
        "iso-639>=0.4",
        "sphinx",
    ],
    classifiers=("Programming Language :: Python :: 3",),
)
