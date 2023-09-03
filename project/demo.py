# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2023(18588220928@163.com), All Rights Reserved.
# ***
# ***    File Author: Dell, Sat 02 Sep 2023 07:27:04 PM CST
# ***
# ************************************************************************************/
#

import Derender

Derender.predict("images/co3d/*.png", "output/co3d", version="co3d")
Derender.predict("images/face/*.png", "output/co3d", version="co3d")

Derender.predict("images/co3d/*.png", "output/face", version="face")
Derender.predict("images/face/*.png", "output/face", version="face")
