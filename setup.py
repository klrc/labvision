#!/usr/bin/env python
# -*- coding:utf-8 -*-

#############################################
# File Name: setup.py
# Author: klrc
# Mail: sh@mail.ecust.edu.cn
#############################################

from setuptools import setup, find_packages  # 这个包没有的可以pip一下

setup(
    name="labvision",  # 这里是pip项目发布的名称
    version="0.3.6",  # 版本号，数值大的会优先被pip
    keywords=("pip", "labvision", "ecust"),
    description="ecust vision lab platform",
    long_description="experimental platform for CV research.",
    license="MIT Licence",

    url="https://github.com/klrc/labvision",  # 项目相关文件地址，一般是github
    author="klrc",
    author_email="sh@mail.ecust.edu.com",

    packages=find_packages(),
    include_package_data=True,
    platforms=["all"],
    install_requires=["torch", "torchvision", "paramiko", "opencv-python", "numpy", "scipy",
                      'matplotlib', 'imageio', 'Pillow']  # 这个项目需要的第三方库
)


'''
    >>> pipreqs labvision
    >>> python setup.py sdist
    >>> twine upload dist/labvision-0.3.x.tar.gz
    >>> pip install labvision -U -i https://pypi.org/simple
'''
