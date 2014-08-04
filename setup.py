from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name = "simpledist",
    version = "0.1",
    description = "Defines objects useful for describing simple probability distributions.",
    long_description = readme(),
    author = "Timothy D. Morton",
    author_email = "tim.morton@gmail.com",
    url = "https://github.com/timothydmorton/distributions",
    packages = ['distributions'],
    classifiers=[
      'Development Status :: 4 - Beta',
      'Intended Audience :: Science/Research',
      'Operating System :: OS Independent',
      'Programming Language :: Python',
      'Topic :: Scientific/Engineering'
      ]
    install_requires=['plotutils'],
    zip_safe=False)
) 
