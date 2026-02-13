"""Setup script for the LLM Fine-Tuning Project."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="llm-fine-tuning",
    version="0.1.0",
    author="Sona Mansuryan",
    author_email="mansuryansona04@gmail.com",
    description="Language Model Fine-Tuning for Misinformation Detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sonamansuryan/FakeNewsDetectionSystems",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "mypy>=1.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "llm-explore=data_processing.data_explorer:main",
            "llm-combine=data_processing.data_combiner:main",
            "llm-train=training.trainer:main",
            "llm-evaluate=evaluation.evaluator:main",
        ],
    },
)