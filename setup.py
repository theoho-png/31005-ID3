from setuptools import setup

setup(
   name='ID3',
   version='1.0',
   description='ID3 Algorithm Implementation',
   author='Theodora Ho',
   packages=['ID3'],  #same as name
   install_requires=['numpy', 'pandas', 'seaborn', 'matplotlib'], #external packages as dependencies
)