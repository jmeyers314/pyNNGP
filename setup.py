import os
import re
import sys
import subprocess

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion

VERSIONFILE="pyNNGP/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    version = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
        build_args += ['--', '-j4']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        if 'CMAKE_VERBOSE_MAKEFILE' in env:
            cmake_args += ['-DCMAKE_VERBOSE_MAKEFILE=1']
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

with open("README.md", 'r') as fh:
    long_description = fh.read()

setup(
    name='pyNNGP',
    version=version,
    author='Josh Meyers',
    author_email='jmeyers314@gmail.com',
    license='BSD',
    url='https://github.com/jmeyers314/pyNNGP',
    description="Nearest Neighbor Gaussian Process",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=['pyNNGP'],
    package_dir={'pyNNGP': 'pyNNGP'},
    package_data={'pyNNGP' : ['data/**/*']},
    ext_modules=[CMakeExtension('pyNNGP._pyNNGP')],
    install_requires=['numpy', 'pyyaml'],
    python_requires='>=3.4',
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
    include_package_data=True,
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3.4'
        'Programming Language :: Python :: 3.5'
        'Programming Language :: Python :: 3.6'
        'Programming Language :: Python :: 3.7'
    ],
    project_urls={
        'Source': "https://github.com/jmeyers314/pyNNGP",
        'Tracker': "https://github.com/jmeyers314/pyNNGP/issues"
    }
)
