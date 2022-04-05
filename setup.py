from setuptools import setup

setup(
    name='pyMFI',
    version='0.1.0',    
    description='A package to compute FES via Mean Force Integration',
    url='https://github.com/shuds13/pyexample',
    author='Matteo Salvalaglio, Antoniu Bjola',
    author_email='m.salvalaglio@ucl.ac.uk',
    license='MIT',
    packages=['pyMFI'],
    install_requires=['scipy',
                      'numpy',                     
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)
