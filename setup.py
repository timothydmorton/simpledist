from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name = "simpledist",
    version = "0.1.5-4",
    description = "Defines objects useful for describing simple probability distributions.",
    long_description = readme(),
    author = "Timothy D. Morton",
    author_email = "tim.morton@gmail.com",
    url = "https://github.com/timothydmorton/simpledist",
    packages = ['simpledist'],
    classifiers=[
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Science/Research',
      'Operating System :: OS Independent',
      'Programming Language :: Python',
      'Topic :: Scientific/Engineering'
      ],
    install_requires=['plotutils','pandas>=0.13'],
    zip_safe=False
) 
