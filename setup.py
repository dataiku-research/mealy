# *- encoding: utf-8 -*-
import sys
import os
from collections import defaultdict

from setuptools import setup, find_packages

descr = """Model Error Analysis python package"""

def load_version():
    """Executes mealy/version.py in a globals dictionary and return it.
    """
    globals_dict = {}
    with open(os.path.join('mealy', 'version.py')) as fp:
        exec(fp.read(), globals_dict)
    return globals_dict


def is_installing():
    # Allow command-lines such as "python setup.py build install"
    install_commands = set(['install', 'develop'])
    return install_commands.intersection(set(sys.argv))


# Make sources available using relative paths from this file's directory.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

_VERSION_GLOBALS = load_version()
DISTNAME = 'mealy'
DESCRIPTION = 'Model Error Analysis python package'
with open('README.md') as fp:
    LONG_DESCRIPTION = fp.read()
MAINTAINER = 'Du Phan'
MAINTAINER_EMAIL = 'du.phan@dataiku.com'
URL = 'https://github.com/dataiku/mealy'
LICENSE = 'Apache 2.0'
DOWNLOAD_URL = 'https://github.com/dataiku/mealy'
VERSION = _VERSION_GLOBALS['__version__']

if __name__ == "__main__":
    if is_installing():
        module_check_fn = _VERSION_GLOBALS['check_modules']
        module_check_fn()

    install_requires = []
    extras_require = defaultdict(list)

    for mod, meta in _VERSION_GLOBALS['DEPENDENCIES_METADATA']:
        if 'min_version' in meta:
            dep_str = '%s>=%s' % (mod, meta['min_version'])
        elif 'exact_version' in meta:
            dep_str = '%s==%s' % (mod, meta['exact_version'])
        if 'extra_options' in meta:
            for extra_option in meta['extra_options']:
                extras_require[extra_option].append(dep_str)
            extras_require['all'].append(dep_str)
        else:
            install_requires.append(dep_str)

    setup(name=DISTNAME,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          license=LICENSE,
          url=URL,
          version=VERSION,
          download_url=DOWNLOAD_URL,
          long_description=LONG_DESCRIPTION,
          long_description_content_type='text/markdown',
          zip_safe=False,  # the package can run out of an .egg file
          classifiers=[
              'Intended Audience :: Science/Research',
              'Intended Audience :: Developers',
              'Programming Language :: Python',
              'Topic :: Scientific/Engineering',
              'Operating System :: Microsoft :: Windows',
              'Operating System :: POSIX',
              'Operating System :: Unix',
              'Operating System :: MacOS',
              'Programming Language :: Python :: 3.6',
              'Programming Language :: Python :: 3.7',
          ],
          packages=find_packages(),
          package_data={},
          python_requires='>=3.6',
          install_requires=install_requires,
          extras_require=extras_require)
