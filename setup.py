import pathlib

from setuptools import find_packages, setup

# directory holding this file
HERE = pathlib.Path(__file__).parent

# text of the README file
README = (HERE / "README.md").read_text()

setup(
    name="mdpfuzz",
    version="0.0.1",
    author="Mazouni Quentin",
    author_email="quentin@simula.no",
    description="Re-implementation of MDPFuzz.",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/QuentinMaz/MDPFuzz_Replication/",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords=["mdpfuzz"],
    packages=find_packages(),
    python_requires=">=3.5",
    install_requires=[
        "numpy",
        "pandas",
        "tqdm",
        "scipy",
        "matplotlib",
        "pillow",
    ],
    extras_require={
        "test": [
            "pytest",
        ],
    },
    project_urls={
        "Homepage": "https://github.com/QuentinMaz/MDPFuzz_Replication/",
        "Issues": "https://github.com/QuentinMaz/MDPFuzz_Replication/issues",
    },
)
