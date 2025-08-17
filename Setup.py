from setuptools import setup, find_packages
from typing import List


Hyphen_E_Dot='-e .'

def get_requirements(file_path: str) -> List[str]:
    """
    This function reads a requirements file and returns a list of packages.
    """
    with open(file_path, 'r') as file:
        requirements = file.readlines()
    if Hyphen_E_Dot in requirements:
        requirements.remove(Hyphen_E_Dot)
    # Strip whitespace and filter out comments
    return [req.strip() for req in requirements if req.strip() and not req.startswith('#')]

setup( 
    name='ml_project',
    author='Suhas',
    version='0.0.1',
    author_email='suhas.chandrashekaran@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)