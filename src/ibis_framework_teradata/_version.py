from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version('ibis_framework_teradata')
except PackageNotFoundError:
    __version__ = '0+unknown'
