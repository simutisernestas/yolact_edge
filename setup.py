#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

cmdclass = {}
cmdclass.update({'build_ext': build_ext})
ext_modules = [Extension("cython_nms", ["yolact_edge/utils/cython_nms.pyx"])]

with open('README.md') as readme_file:
    readme = readme_file.read()

setup(
    name='yolact_edge',
    author="Ernestas Šimutis",
    author_email='simutisernestas@gmail.com',
    description="YOLACT Edge is real time instance segmentation package.",
    license="Apache Software License 2.0 TODO",
    long_description=readme,
    include_package_data=True,
    keywords='yolact_edge',
    packages=find_packages(include=['yolact_edge', 'yolact_edge.*']),
    url='https://github.com/simutisernestas/yolact_edge',
    version='0.1.0',
    zip_safe=False,
    cmdclass=cmdclass,
    ext_modules=ext_modules,
)
