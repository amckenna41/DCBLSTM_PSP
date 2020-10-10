#Setup.py installs all the required packages and dependancies when packaging
#the GCP application for training

from setuptools import setup, find_packages

setup(name='training',
      version='0.1',
      description='Running Deep CNN on Google Cloud Ai-platform',
      author='Adam McKenna',
      author_email='amckenna41@qub.ac.uk',
      license='',
      install_requires=[
          'tensorflow>=2.1.2',
          # 'tensorflow-gpu==1.15',
          'h5py',
          'keras',
          'google.cloud',
          'matplotlib',
          'seaborn',
          # 'pydot',
          # 'graphviz',
          'cloudml-hypertune'
      ],
     packages=find_packages(),
     include_package_data=True,
     zip_safe=False)
