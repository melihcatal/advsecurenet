import os

from setuptools import find_packages, setup

with open(os.path.join(os.path.dirname(__file__), 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


def read_version():
    with open("cli/__init__.py") as f:
        for line in f:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip("\"'")


setup(
    name="advsecurenet",
    version=read_version(),
    description="AdvSecureNet | Adversarial Secure Networks | Machine Learning Security",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="Melih Catal",
    author_email="melih.catal@uzh.ch",
    url="https://github.com/melihcatal/advsecurenet",
    packages=find_packages(),
    include_package_data=True,
    package_data={'': ['*.yml']},
    python_requires='>=3.10',
    install_requires=[
        "click",
        "torch",
        "torchvision",
        "colored",
        "tqdm",
        "PyYAML",
        "opencv-python",
        "ruamel.yaml",
        "matplotlib",
        "scikit-image",
        "einops",
        "filetype"
    ],
    entry_points={
        'console_scripts': [
            'advsecurenet=cli.cli:main',
        ],
    },
)
