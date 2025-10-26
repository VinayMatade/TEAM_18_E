"""
Setup script for UAV Log Processor package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
with open(requirements_path) as f:
    requirements = [line.strip() for line in f if line.strip()
                    and not line.startswith("#")]

# Read README if it exists
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    with open(readme_path, encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="uav-log-processor",
    version="1.0.0",
    description="Process UAV flight logs into ML-ready datasets for GPS error correction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Vinay Matade",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="uav drone gps mavlink machine-learning tcn",
    entry_points={
        "console_scripts": [
            "uav-log-processor=uav_log_processor.cli:main",
        ],
    },
)
