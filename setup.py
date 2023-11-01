from setuptools import setup, find_packages

setup(
    name="AdvSecureNet",
    version="0.1",
    description="Adversarial Secure Networks",
    author="Melih Catal",
    author_email="melih.catal@uzh.ch",
    url="https://github.com/melihcatal/advsecurenet",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "click",
        "torch",
        "torchvision",
        "colored",
        "pytest",
        "pytest-mock",
        "tqdm", 
        "PyYAML",
        "opencv-python"
    ],
    entry_points={
        'console_scripts': [
            'advsecurenet=cli.cli:main',
        ],
    },
)
