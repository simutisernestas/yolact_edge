#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

cmdclass = {}
cmdclass.update({'build_ext': build_ext})
ext_modules = [Extension("yolact_edge", ["utils/cython_nms.pyx"])]
    
with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = []

setup_requirements = []

test_requirements = []

setup(
    author="Ernestas Šimutis",
    author_email='simutisernestas@gmail.com',
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="YOLACT Edge is real time instance segmentation package.",
    install_requires=requirements,
    license="Apache Software License 2.0",
    long_description=readme,
    include_package_data=True,
    keywords='yolact_edge',
    name='yolact_edge',
    packages=find_packages(include=['yolact_edge', 'yolact_edge.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/simutisernestas/yolact_edge',
    version='0.1.0',
    zip_safe=False,
    cmdclass=cmdclass,
    ext_modules=ext_modules,
)
