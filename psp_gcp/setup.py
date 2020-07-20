from setuptools import setup, find_packages

setup(name='training',
      version='0.1',
      description='example to run keras on gcloud ml-engine',
      author='Adam McKenna',
      author_email='amckenna41@qub.ac.uk',
      license='MIT',
      install_requires=[
          'tensorflow>=2.2',
          'h5py',
          'keras'
         # 'keras==2.4.2',
         # 'tensorflow==2.1.0',
          #'h5py==2.10.0'
          #'tensorflow'
          #'tensorflow>=2.2.0',
          #'keras==2.4.3',

          #'h5py==2.10.0'
      ],
     packages=find_packages(),
     include_package_data=True,


     zip_safe=False)

# REQUIRED_PACKAGES = ['some_PyPI_package>=1.0']
