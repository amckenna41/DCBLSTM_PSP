#Setup.py installs all the required packages and dependancies when packaging
#the GCP application for training

from setuptools import setup, find_packages

setup(name='training',
      version='0.1',
      description='Running Deep NN on Google Cloud Ai-platform',
      author='Adam McKenna',
      author_email='amckenna41@qub.ac.uk',
      license='',
      install_requires=[
          'tensorflow>=2.1.2',
          # 'tensorflow-gpu==1.15',
          'h5py',
          'pandas',
          'keras',
          'google.cloud',
          'google-cloud-core==1.3.0',
          'google-api-core==1.16.0',
          # 'google-cloud-logging',
          # 'google-cloud-pubsub==2.1.0",
          'requests',
          'matplotlib',
          'seaborn',
          'grpcio',
          # 'pydot',
          # 'graphviz',
          'cloudml-hypertune'
      ],
     packages=find_packages(),
     include_package_data=True,
     zip_safe=False)
          # 'google-cloud-core==1.7.0',
          # 'google-cloud-core==1.4.3',
