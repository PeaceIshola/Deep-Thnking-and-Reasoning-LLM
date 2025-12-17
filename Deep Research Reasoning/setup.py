"""
Setup configuration for Deep Research Reasoning System.
Install as a package: pip install -e .
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="deep-research-reasoning",
    version="1.0.0",
    author="Deep Research Team",
    description="Advanced reasoning techniques for LLMs including CoT, Self-Consistency, and Tree-of-Thoughts",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
    install_requires=[
        "openai>=1.0.0",
        "transformers>=4.30.0",
        "torch>=2.0.0",
        "langchain>=0.1.0",
        "langchain-community>=0.0.20",
        "duckduckgo-search>=4.0.0",
        "numpy>=1.24.0",
        "regex>=2023.0.0",
    ],
    extras_require={
        "jupyter": ["ipykernel>=6.20.0", "jupyter>=1.0.0"],
    },
    entry_points={
        "console_scripts": [
            "deep-research=main:main",
        ],
    },
)
