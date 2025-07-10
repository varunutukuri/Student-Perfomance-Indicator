from setuptools import setup, find_packages
from typing import List

HYPHEN_E_DOT = '-e .'
def get_requirements(file_path:str)->List[str]:
    '''
    this function will return the list of requirements
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n', '') for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements
    
       
    

setup(
    name = 'ml_project',
    version = '0.0.1',
    author = 'varun',
    author_email = 'varun.utukuri@gmail.com',
    pacjkages = find_packages(),
    install_requires = get_requirements('requirements.txt')

)