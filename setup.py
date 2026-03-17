"""Package setup for BioMassters AGB estimation benchmark."""

from setuptools import setup, find_packages

setup(
    name="biomassters",
    version="0.1.0",
    description="Multi-architecture deep learning benchmark for Above Ground Biomass estimation",
    author="Reza Asiyabi",
    license="MIT",
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "pytorch-lightning>=2.0.0",
        "torchmetrics>=1.0.0",
        "omegaconf>=2.3.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "full": [
            "wandb>=0.15.0",
            "timm>=0.9.0",
            "huggingface_hub>=0.16.0",
            "rasterio>=1.3.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "scipy>=1.10.0",
            "scikit-learn>=1.2.0",
            "jupyter>=1.0.0",
            "ipywidgets>=8.0.0",
        ],
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.0.0",
            "ruff>=0.1.0",
            "black>=23.0.0",
            "mypy>=1.4.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: GIS",
    ],
)
