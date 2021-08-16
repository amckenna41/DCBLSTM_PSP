################################################################################
######## Setup.py - installs all the required packages and dependancies ########
########        when packaging the GCP application for training         ########
################################################################################

from setuptools import setup, find_packages
import sys

if sys.version_info[0] < 3:
    sys.exit('Python 3 is the minimum version requirement')

setup(name='training',
      version='1.0.0',
      description='Running Protein Structure Prediction, on Google Cloud Ai-platform',
      author='Adam McKenna',
      author_email='amckenna41@qub.ac.uk',
      license='MIT',
      install_requires=[
          'tensorflow>=2.2.0',
          'numpy==1.16.6',
          'h5py',
          'pandas',
          'keras',
          'google.cloud',
          'google-cloud-core==1.3.0',
          'google-api-core==1.16.0',
          'requests',
          'matplotlib',
          'seaborn',
          'cloudml-hypertune'
      ],
     packages=find_packages(),
     include_package_data=True,
     zip_safe=False)

       # no longer needed modules/libraries
       # 'google-cloud-logging',
       # 'google-cloud-pubsub==2.1.0",
       # 'tensorflow-gpu==1.15',
       # 'google-cloud-core==1.7.0',
       # 'google-cloud-core==1.4.3',
       # 'grpcio',
       # 'pydot',
       # 'graphviz',
