from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = "-e ."

def get_requirements(file_path: str) -> List[str]:
    """
    This function reads the requirements.txt, and returns a list of required packages.

    :param file_path:
    :return:
    """

    # Initialize an empty list for requirements
    requirements = []

    # Read and save each package
    with open(file_path, 'r') as file:
        packages = file.readlines()
        requirements = [package.replace('\n', '') for package in packages]

    # Exclude `-e .`
    if HYPHEN_E_DOT in requirements:
        requirements.remove(HYPHEN_E_DOT)

    return requirements

setup(
    name='ML SIEM',
    version='0.0.1',
    author='Emre OTU',
    author_email='dev.emre17@gmail.com',
    packages=find_packages(),
    install_requirements=get_requirements('requirements.txt')
)