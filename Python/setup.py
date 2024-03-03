from distutils.core import setup

setup(name='VQPCASuite',
  version= '1.0.0',
  description='Python libraries for model order reduction, clustering and data analysis.',
  author='Matteo Savarese',
  author_email= 'matteo.savarese@ulb.be',
  packages =['VQPCASuite'],
  keywords = ['Principal Component Analysis', 'clustering', 'dimensionality reduction', 'model order reduction'])

'''
$ python setup.py install
'''
