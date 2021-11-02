################################################################################
######## Setup.py - installs all the required packages and dependancies ########
########        when packaging the GCP application for training         ########
################################################################################

from setuptools import setup, find_packages
import sys, pathlib
import psp_gcp

#check current Python version is 3 or above
if sys.version_info[0] < 3:
    sys.exit('Python 3 is the minimum version requirement')

HERE = pathlib.Path(__file__).parent

README = (HERE / 'README.md').read_text()

setup(name='DCBLSTM_PSP on GCP',
      version=psp_gcp.__version__,
      description='Running Protein Structure Prediction, on Google Cloud Ai-platform',
      long_description=README,
      long_description_content_type="text/markdown",
      author=psp_gcp.__license__,
      author_email=psp_gcp.__authorEmail__,
      license=psp_gcp.__license__,
      url=psp_gcp.__url__,
      classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Prog9amming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: Implementation :: PyPy',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
      ],
      install_requires=[
          'Cython',
          'pyparsing==2.4.2',
          'grpcio==1.34.0',
          'tensorflow>=2.2.0',
          'tensorboard==2.5.0',
          'numpy==1.19.3',
          'h5py',
          'pandas',
          'keras',
          'google-cloud==0.34.0',
          'google-cloud-core==1.4.1',
          'google-api-core==1.26.0',
          'google-cloud-storage==1.31.0',
          'google-api-python-client==1.12.1',
          'google-cloud-logging',
          'oauth2client==4.1.3',
          'gcsfs',
          'requests',
          'matplotlib',
          'seaborn',
          'cloudml-hypertune',
          'pydot',
          'graphviz'
      ],
     packages=find_packages(),
     include_package_data=True,
     zip_safe=True)

       # old modules/libraries
       # 'numpy >= 1.19.3; python_version < "3.9.0"',
       # 'google-cloud-logging',
       # 'google-cloud-pubsub==2.1.0",
       # 'tensorflow-gpu==1.15',
       # 'google-cloud-core==1.7.0',
       # 'google-cloud-core==1.4.3',
       # 'google-cloud-pubsub',
