
from setuptools import find_packages,setup 
from typing import List 

## find_packages(Module) will automatically find out all the packages that are available in the entire ML
## application in the directory that we have actually created  

## This Basically is the Meta Data information about the Project

HYPEN_E_DOT = '-e .'
def get_requirements(file_path:str)-> List[str]:    ## List[str] --> is a type hint not a data type

    '''
    this function will return the list of requirements
    '''

    requirements =[]
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements


setup(
    name='mlproject',
    version='0.0.1',
    author='Revanth',
    author_email='revanthdoppalapudi97@gmail.com',
    packages=find_packages(),                                 ## find_packages is a module that we have imported
    install_requires=get_requirements('requirements.txt')
)

## mlproject.egg-info ---> This indicates that the Package is getting installed, now we can use this package anywhere if you probably 
## deploy in py py

## Whenever we are trying to install this requirements.txt, At this point of time, the setup.py file should also run to build the package
##  for enabling that we write "-e . " --> This will automatically trigger setup.py file

## ['pandas,'numpy','seaborn'] ---> We might require hundereds of packages like these, so it is not feasible to right like this
## 

## How it will know that how many packages are created or not is by using "src" folder

## If we want the source to be found as a package, we will create a __init__.py file, 

## whenever in setup.py, this find_packages in running, it will just go and see, In how many folders we have __init__.py, it will
## directly consider this source as package itself and then it will try to build this. Once "src" builds, we can import anywhere 
## we want like we import seaborn, pandas, numpy

## For "src" to built at the package itself, we will be using the __init__.py file

## Whenever we create any new folder also, we will be using this file, that internal folder also behaves like a package once we 
## build it.