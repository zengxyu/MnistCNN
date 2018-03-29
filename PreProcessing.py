import os
import struct
import numpy as np
import csv

# 读入数据，加载下载的MNIST数据是，需要这样读出
def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels.idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images.idx3-ubyte'
                               % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 28, 28)

    return images, labels  # image的大小是60000*28*28


# 将数据从原来的数据中读出，并写入CSV，分为6个CSV写入
# 把一个数据集拆分为6个
# 数据集的格式为：一张图片：28*28  + 1
# 返回数据集名称张量
def makeCSVFiles(path, kind="train"):
    data_train_path = os.path.join(path, "data_train")
    labels_path = os.path.join(path,
                               '%s-labels.idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images.idx3-ubyte'
                               % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)
    #打开第一个CSV
    file = open(os.path.join(data_train_path, ('mnist_batch0' +  ".csv")),'w',newline='')
    for i in range(len(labels)):
        example = []
        example.extend(images[i])
        example.append(labels[i])
        # with file as f:
        writer = csv.writer(file)
        writer.writerow(example)
        # print("example:"+str(example))
        if i % 10000 == 0:
            print("i:"+str(i))
            index = int(i / 10000)
            file.close()
            # 存入文件中，第10000-20000个存入另一个CSV
            file = open(os.path.join(data_train_path, ('mnist_batch' + str(index) + ".csv")),'w',newline='')
    file.close()

# makeCSVFiles(path="data")