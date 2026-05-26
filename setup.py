#!/usr/bin/env python3
"""Setup configuration for normalization_pipeline package."""

from setuptools import setup, find_packages

setup(
    name="normalization_pipeline",
    version="0.1.0",
    description="Cross-species gene expression normalization benchmarking pipeline",
    author="Your Name",
    packages=find_packages(where="."),
    package_dir={"": "."},
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "scikit-bio>=0.5.6",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "pydeseq2>=0.4.0"
    ],
    entry_points={
        "console_scripts": [
            "normalize=bin.normalize:main",
        ],
    },
)