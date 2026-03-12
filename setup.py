# SPDX-License-Identifier: MIT
from pathlib import Path
from setuptools import setup, find_packages

HERE = Path(__file__).parent
README_PATH = HERE / "README.md"
long_description = README_PATH.read_text(encoding="utf-8") if README_PATH.exists() else (
    "A Gymnasium environment for mobile robot navigation with lidar, collision checking, and A* planning."
)

about = {}
version_file = HERE / "mobile_robot_nav" / "__version__.py"
if version_file.exists():
    exec(version_file.read_text(encoding="utf-8"), about)
else:
    about["__version__"] = "0.1.0"

setup(
    name="mobile-robot-nav",
    version=about["__version__"],
    description="Gymnasium environment for mobile robot navigation with lidar and A* planning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Md Shahriar Forhad",
    author_email="shahriar.forhad.eee@gmail.com",
    url="https://github.com/Shahriar88/mobile-robot-nav",
    license="MIT",
    license_files=["LICENSE"],
    packages=find_packages(exclude=("tests", "docs")),
    include_package_data=True,
    install_requires=[
        "gymnasium>=0.29.0",
        "numpy>=1.24.0",
        "pygame>=2.5.0",
    ],
    extras_require={
        "dev": [
            "build>=1.0.0",
            "twine>=5.0.0",
        ]
    },
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords=[
        "reinforcement learning",
        "gymnasium",
        "mobile robot",
        "navigation",
        "lidar",
        "astar",
        "path planning",
    ],
    project_urls={
        "Source": "https://github.com/Shahriar88/mobile-robot-nav",
        "Issues": "https://github.com/Shahriar88/mobile-robot-nav/issues",
    },
)