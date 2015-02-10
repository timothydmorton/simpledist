from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

# Hackishly inject a constant into builtins to enable importing of the
# package before the library is built.
import sys
if sys.version_info[0] < 3:
    import __builtin__ as builtins
else:
    import builtins
builtins.__SIMPLEDIST_SETUP__ = True
import simpledist

setup(name = "simpledist",
    version = simpledist.__version__,
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
    install_requires=['plotutils','pandas>=0.15'],
    zip_safe=False
) 
