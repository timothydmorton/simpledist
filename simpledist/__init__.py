__version__ = '0.1.13'

try:
    __SIMPLEDIST_SETUP__
except NameError:
    __SIMPLEDIST_SETUP__ = False

if not __SIMPLEDIST_SETUP__:
    __all__ = ['distributions']

