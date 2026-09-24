import os
import sys
# pyrefly: ignore [missing-source-for-stubs]
from setuptools import setup, find_packages

setup(
    name="omnisight",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "opencv-python-headless>=4.8.0",
        "matplotlib>=3.7.0",
        "plotly>=5.15.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "pillow>=9.5.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
        "rich>=13.0.0",
    ],
    entry_points={
        "console_scripts": [
            "omnisight=run_experiment:main",
        ],
    },
)
