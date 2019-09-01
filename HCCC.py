# Python 3.7.4
# Handwritten Chinese Character Classification

from gnt import GNT
import os
import zipfile
from matplotlib import pyplot as plt

step = 1000  # 全局变量，每次处理step个图片，防止占用太多内存，可以根据实际情况更改

def Getstep(gnt, imgs, labels):
    i = 0
    for img, label in gnt:
        if i == step:
            break
        imgs[i] = img
        labels[i] = label
        i = i + 1

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# path为数据集目录
root = 'D:\data\课程\人工智能\手写文本数据库'
# file为文件名
file = 'HWDB1.0trn_gnt.zip'

Z = zipfile.ZipFile(f'{root}\{file}')  # 数据集为压缩包形式
set_name = Z.namelist()[0]  # 取压缩包中的第一个数据集
gnt = GNT(Z, set_name)  # gnt即包含了目标数据集中的所有数据，形式为：(img, label)

imgs = [0 for x in range(0, step)]
labels = [0 for x in range(0, step)]
Getstep(gnt, imgs, labels)  # 获取数据集中的1000个数据
for i in range(0, step):  # 显示图片，实际训练中不需要
    plt.imshow(imgs[i])
    plt.title(labels[i])
    plt.show()