"""Setup."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2023(18588220928@163.com), All Rights Reserved.
# ***
# ***    File Author: Dell, Sat 02 Sep 2023 07:27:04 PM CST
# ***
# ************************************************************************************/
#

from setuptools import setup

with open("README.md", "r") as file:
    long_description = file.read()

setup(
    name="Derender",
    version="1.0.0",
    author="Dell Du",
    author_email="18588220928@163.com",
    description="Derender Model Package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/delldu/Derender.git",
    packages=["Derender"],
    package_data={"Derender": ["models/co3d_netD.pth","models/co3d_netA.pth","models/co3d_netL.pth",
        "models/face_netD.pth","models/face_netA.pth","models/face_netL.pth",
        ]
    },
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "torch >= 1.9.0",
        "torchvision >= 0.10.0",
        "Pillow >= 7.2.0",
        "numpy >= 1.19.5",
        "todos >= 1.0.0",
    ],
)
