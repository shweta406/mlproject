from setuptools import find_packages,setup
from typing import List

HYPHEN_E_DOT='-e .'
def get_requirements(file_path:str)->List[str]:
    '''
    this func will return list of requirements'''

    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        #to remove /n in requirements.txt
        requirements=[req.replace("\n","") for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)#kyuki vo automatically ise trigger kr deta toh yha se add krne ki jrurat ni

    return requirements
setup(
   name='mlproject',
   version='0.0.1',
   author='Shweta yadav',
   author_email='23cs2042@rgipt.ac.in',
   packages=find_packages(),
   install_requires=get_requirements('requirements.txt'),#func will return a list
   )