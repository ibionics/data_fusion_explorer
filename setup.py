from setuptools import setup

setup(name='v2',
      version='2.0',
      description='Applying data fusion for data pipeline platform',
      url='http://github.ncsu.edu/dmarti22/data_fusion',
      license='MIT',
      packages=['DFE_object'],
      zip_safe=False,
      install_requires=['numpy', 'pandas', 'matplotlib', 'seaborn', 'scikit-learn']
      )
