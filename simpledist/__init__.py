__version__ = '0.1.6'

try:
    __SIMPLEDIST_SETUP__
except NameError:
    __SIMPLEDIST_SETUP__ = False

if not __SIMPLEDIST_SETUP__:
    __all__ = ['distributions']

