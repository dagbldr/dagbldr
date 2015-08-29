import os
import setuptools

setuptools.setup(
    name='dagbldr',
    version='0.0.1',
    packages=setuptools.find_packages(),
    author='Kyle Kastner',
    author_email='kastnerkyle@gmail.com',
    description='Deep DAG in Theano',
    long_description=open(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), 'README.rst')).read(),
    license='BSD 3-clause',
    url='http://github.com/dagbldr/dagbldr/',
    package_data={
       'dagbldr': ['utils/js_plot_dependencies/*.html',
                   'utils/js_plot_dependencies/js/*']
    },
    install_requires=['numpy',
                      'scipy',
                      'theano',
                      'tables'],
    classifiers=['Development Status :: 3 - Alpha',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: BSD License',
                 'Operating System :: OS Independent',
                 'Topic :: Scientific/Engineering'],
)
