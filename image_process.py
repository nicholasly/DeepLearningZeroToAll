# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 10:34:07 2018

@author: nicholasly
"""

import tensorflow as tf
import cv2

# 这里定义一个tensorflow读取的图片格式转换为opencv读取的图片格式的函数
# 请注意：
# 在tensorflow中，一个像素点的颜色顺序是R，G，B。
# 在opencv中，一个像素点的颜色顺序是B，G，R。
# 因此，我们循环遍历每一个像素点，将第0位的颜色和第2位的颜色数值换一下即可。
# 第一个参数name：将要显示的窗口名称。
# 第二个参数image：储存图片信息的一个tensor。
def cv2Show(name="", image=None):
    # 获取矩阵信息
    np = image.eval()
    # 获取行数列数
    row, col = len(np),len(np[1])

    # 两重循环遍历
    for i in range(row):
        for j in range(col):
            # 交换数值
            tmp = np[i][j][0]
            np[i][j][0] = np[i][j][2]
            np[i][j][2] = tmp

    # 显示图片
    cv2.imshow(name,np)
    pass

# tensorflow会话
with tf.Session() as sess:
    # 以二进制的方式读取图片。
    image_raw_data = tf.gfile.FastGFile("bus.jpg", "rb").read()

    # 按照jpeg的格式解码图片。
    image_data = tf.image.decode_jpeg(image_raw_data)

    # 显示原图片。
    cv2Show("Read by Tensorflow+Dispalyed by Opencv",image_data)

    # opencv读取同一张图片。
    img = cv2.imread("bus.jpg")

    # opencv显示图片。
    cv2.imshow("Read by Opencv+Displayed by Opencv",img)

    # 重新调整大小。
    # resize_images(images,size,method=ResizeMethod.BILINEAR,align_corners=False)'
    # 参数如下：
    # images:需要调整的图片，通常是一个tensor
    # size：调整之后的大小，一般是一个长度为2的list
    # method：调整大小使用的方法，这里我们使用最近邻居法。
    # align_corner：是否对齐四角。
    resized = tf.image.resize_images(image_data,[300,300],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    # 显示图片。
    cv2Show("Resized:Nearest Neighbor",resized)

    # 调整大小，采用切割或者填充的方式。
    # 如果原始图像的尺寸大于目标图像，那么这个函数会自动切割原始图像中的居中的部分。
    # 如果原始图像的尺寸小于目标图像，那么这个函数会自动在原始图像的周围采用全0填充。
    # resize_image_with_crop_or_pad(image, target_height, target_width)
    # image：待调整的图像。
    # target_height：目标图像的高度。
    # target_width：目标图像的宽度。
    cropped = tf.image.resize_image_with_crop_or_pad(image_data,300,300)
    cv2Show("After being cropped",cropped)

    padded = tf.image.resize_image_with_crop_or_pad(image_data,600,900)
    cv2Show("After being padded",padded)

    # 通过比例调整图像大小。
    # central_crop(image, central_fraction)
    # image：待调整的图像。
    # central_fraction：比例，是一个(0,1]之间的数字，表示需要调整的比例。这里我们选择的比例是0.5，即50%。
    central_cropped = tf.image.central_crop(image_data,0.5)
    cv2Show("After being central-cropped",central_cropped)

    cv2.waitKey()