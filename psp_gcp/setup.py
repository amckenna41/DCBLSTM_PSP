#Setup.py installs all the required packages and dependancies when packaging
#the GCP application for training

from setuptools import setup, find_packages
# import os, sys
# sys.path.append(os.path.abspath(os.path.join('..', 'models')))

# currentdir = os.path.dirname(os.path.realpath(__file__))
# parentdir = os.path.dirname(currentdir)
# sys.path.append(parentdir)

setup(name='training',#psp_gcp/training
      version='0.1',
      description='Running Deep CNN on Google Cloud Ai-platform',
      author='Adam McKenna',
      author_email='amckenna41@qub.ac.uk',
      license='',
      install_requires=[
          'tensorflow==2.1',
          # 'tensorflow-gpu==1.15',
          'h5py',
          'keras',
          'google.cloud',
          'matplotlib',
          'seaborn'
      ],
     packages=find_packages(),
     include_package_data=True,
     zip_safe=False)
