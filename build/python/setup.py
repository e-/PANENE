from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy as np

extra_link_args = ["-lpanene"] #FIXME Is there a more portable way to do it?
# tried adding "-L.." but does not work

if "-std=c++11":
    extra_link_args.append("-std=c++11")

if "":
    extra_link_args.append("")
    
if "-fopenmp":
    extra_link_args.append("-fopenmp")

extension = [Extension(
        "pynene",
        ["/home/hkko/vbox/PANENE/python/pynene.pyx"],
        include_dirs= ["/home/hkko/vbox/PANENE/include", np.get_include(),],
        extra_compile_args=["-std=c++11", "-fopenmp"],
        extra_link_args=extra_link_args,
        language="c++",
    )]

setup(name='pynene',
      version='0.0.1',
      description='Progressive Library for Approximate Nearest Neighbors',
      author='Jaemin Jo',
      author_email='jmjo@hcil.snu.ac.kr',
      license='BSD',
      url='https://github.com/e-/PANENE',
      package_dir={ '': '/home/hkko/vbox/PANENE/python' },
#      packages=['pynene'],
      classifiers=[
          "Development Status :: 2 - PRe-Alpha",
          "Topic :: Scientific/Engineering :: Visualization",
          "Topic :: Scientific/Engineering :: Information Analysis",
          "License :: OSI Approved :: BSD License",
      ],
    platforms='any',
    install_requires = [
        "numpy>=1.11.3",
        "scipy>=0.18.1",
        "cython>=0.25.1",
    ],
    test_suite='tests',
    ext_modules = cythonize(extension),
    zip_safe = False
)
