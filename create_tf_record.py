# -*-coding: utf-8 -*-
"""
    @Project: create_tfrecord
    @File   : create_tfrecord.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2018-07-27 17:19:54
    @desc   : 将图片数据保存为单个tfrecord文件
"""

##########################################################################

import tensorflow as tf
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import random
from PIL import Image


##########################################################################
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
# 生成字符串型的属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
# 生成实数型的属性
def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def get_example_nums(tf_records_filenames):
    '''
    统计tf_records图像的个数(example)个数
    :param tf_records_filenames: tf_records文件路径
    :return:
    '''
    nums= 0
    for record in tf.python_io.tf_record_iterator(tf_records_filenames):
        nums += 1
    return nums

def show_image(title,image):
    '''
    显示图片
    :param title: 图像标题
    :param image: 图像的数据
    :return:
    '''
    # plt.figure("show_image")
    # print(image.dtype)
    plt.imshow(image)
    plt.axis('on')    # 关掉坐标轴为 off
    plt.title(title)  # 图像题目
    plt.show()

def load_labels_file(filename,labels_num=1,shuffle=False):
    '''
    载图txt文件，文件中每行为一个图片信息，且以空格隔开：图像路径 标签1 标签2，如：test_image/1.jpg 0 2
    :param filename:
    :param labels_num :labels个数
    :param shuffle :是否打乱顺序
    :return:images type->list
    :return:labels type->list
    '''
    images=[]
    labels=[]
    with open(filename) as f:
        lines_list=f.readlines()
        if shuffle:
            random.shuffle(lines_list)

        for lines in lines_list:
            line=lines.rstrip().split(' ')
            label=[]
            for i in range(labels_num):
                label.append(int(line[i+1]))
            images.append(line[0])
            labels.append(label)
    return images,labels

def read_image(filename, resize_height, resize_width,normalization=False):
    '''
    读取图片数据,默认返回的是uint8,[0,255]
    :param filename:
    :param resize_height:
    :param resize_width:
    :param normalization:是否归一化到[0.,1.0]
    :return: 返回的图片数据
    '''

    bgr_image = cv2.imread(filename)
    if len(bgr_image.shape)==2:#若是灰度图则转为三通道
        print("Warning:gray image",filename)
        bgr_image = cv2.cvtColor(bgr_image, cv2.COLOR_GRAY2BGR)

    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)#将BGR转为RGB
    # show_image(filename,rgb_image)
    # rgb_image=Image.open(filename)
    if resize_height>0 and resize_width>0:
        rgb_image=cv2.resize(rgb_image,(resize_width,resize_height))
    rgb_image=np.asanyarray(rgb_image)
    if normalization:
        # 不能写成:rgb_image=rgb_image/255
        rgb_image=rgb_image/255.0
    # show_image("src resize image",image)
    return rgb_image


def get_batch_images(images,labels,batch_size,labels_nums,one_hot=False,shuffle=False,num_threads=1):
    '''
    :param images:图像
    :param labels:标签
    :param batch_size:
    :param labels_nums:标签个数
    :param one_hot:是否将labels转为one_hot的形式
    :param shuffle:是否打乱顺序,一般train时shuffle=True,验证时shuffle=False
    :return:返回batch的images和labels
    '''
    min_after_dequeue = 200
    capacity = min_after_dequeue + 3 * batch_size  # 保证capacity必须大于min_after_dequeue参数值
    if shuffle:
        images_batch, labels_batch = tf.train.shuffle_batch([images,labels],
                                                                    batch_size=batch_size,
                                                                    capacity=capacity,
                                                                    min_after_dequeue=min_after_dequeue,
                                                                    num_threads=num_threads)
    else:
        images_batch, labels_batch = tf.train.batch([images,labels],
                                                        batch_size=batch_size,
                                                        capacity=capacity,
                                                        num_threads=num_threads)
    if one_hot:
        labels_batch = tf.one_hot(labels_batch, labels_nums, 1, 0)
    return images_batch,labels_batch

def read_records(filename,resize_height, resize_width,type=None):
    '''
    解析record文件:源文件的图像数据是RGB,uint8,[0,255],一般作为训练数据时,需要归一化到[0,1]
    :param filename:
    :param resize_height:
    :param resize_width:
    :param type:选择图像数据的返回类型
         None:默认将uint8-[0,255]转为float32-[0,255]
         normalization:归一化float32-[0,1]
         centralization:归一化float32-[0,1],再减均值中心化
    :return:
    '''
    # 创建文件队列,不限读取的数量
    filename_queue = tf.train.string_input_producer([filename])
    # create a reader from file queue
    reader = tf.TFRecordReader()
    # reader从文件队列中读入一个序列化的样本
    _, serialized_example = reader.read(filename_queue)
    # get feature from serialized example
    # 解析符号化的样本
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'depth': tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([], tf.int64)
        }
    )
    tf_image = tf.decode_raw(features['image_raw'], tf.uint8)#获得图像原始的数据

    tf_height = features['height']
    tf_width = features['width']
    tf_depth = features['depth']
    tf_label = tf.cast(features['label'], tf.int32)
    # PS:恢复原始图像数据,reshape的大小必须与保存之前的图像shape一致,否则出错
    # tf_image=tf.reshape(tf_image, [-1])    # 转换为行向量
    tf_image=tf.reshape(tf_image, [resize_height, resize_width, 3]) # 设置图像的维度

    # 恢复数据后,才可以对图像进行resize_images:输入uint->输出float32
    # tf_image=tf.image.resize_images(tf_image,[224, 224])

    # 存储的图像类型为uint8,tensorflow训练时数据必须是tf.float32
    if type is None:
        tf_image = tf.cast(tf_image, tf.float32)
    elif type=='normalization':# [1]若需要归一化请使用:
        # 仅当输入数据是uint8,才会归一化[0,255]
        # tf_image = tf.image.convert_image_dtype(tf_image, tf.float32)
        tf_image = tf.cast(tf_image, tf.float32) * (1. / 255.0)  # 归一化
    elif type=='centralization':
        # 若需要归一化,且中心化,假设均值为0.5,请使用:
        tf_image = tf.cast(tf_image, tf.float32) * (1. / 255) - 0.5 #中心化

    # 这里仅仅返回图像和标签
    # return tf_image, tf_height,tf_width,tf_depth,tf_label
    return tf_image,tf_label


def create_records(image_dir,file, output_record_dir, resize_height, resize_width,shuffle,log=5):
    '''
    实现将图像原始数据,label,长,宽等信息保存为record文件
    注意:读取的图像数据默认是uint8,再转为tf的字符串型BytesList保存,解析请需要根据需要转换类型
    :param image_dir:原始图像的目录
    :param file:输入保存图片信息的txt文件(image_dir+file构成图片的路径)
    :param output_record_dir:保存record文件的路径
    :param resize_height:
    :param resize_width:
    PS:当resize_height或者resize_width=0是,不执行resize
    :param shuffle:是否打乱顺序
    :param log:log信息打印间隔
    '''
    # 加载文件,仅获取一个label
    images_list, labels_list=load_labels_file(file,1,shuffle)

    writer = tf.python_io.TFRecordWriter(output_record_dir)
    for i, [image_name, labels] in enumerate(zip(images_list, labels_list)):
        image_path=os.path.join(image_dir,images_list[i])
        if not os.path.exists(image_path):
            print('Err:no image',image_path)
            continue
        image = read_image(image_path, resize_height, resize_width)
        image_raw = image.tostring()
        if i%log==0 or i==len(images_list)-1:
            print('------------processing:%d-th------------' % (i))
            print('current image_path=%s' % (image_path),'shape:{}'.format(image.shape),'labels:{}'.format(labels))
        # 这里仅保存一个label,多label适当增加"'label': _int64_feature(label)"项
        label=labels[0]
        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': _bytes_feature(image_raw),
            'height': _int64_feature(image.shape[0]),
            'width': _int64_feature(image.shape[1]),
            'depth': _int64_feature(image.shape[2]),
            'label': _int64_feature(label)
        }))
        writer.write(example.SerializeToString())
    writer.close()

def disp_records(record_file,resize_height, resize_width,show_nums=4):
    '''
    解析record文件，并显示show_nums张图片，主要用于验证生成record文件是否成功
    :param tfrecord_file: record文件路径
    :return:
    '''
    # 读取record函数
    tf_image, tf_label = read_records(record_file,resize_height,resize_width,type='normalization')
    # 显示前4个图片
    init_op = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(show_nums):
            image,label = sess.run([tf_image,tf_label])  # 在会话中取出image和label
            # image = tf_image.eval()
            # 直接从record解析的image是一个向量,需要reshape显示
            # image = image.reshape([height,width,depth])
            print('shape:{},tpye:{},labels:{}'.format(image.shape,image.dtype,label))
            # pilimg = Image.fromarray(np.asarray(image_eval_reshape))
            # pilimg.show()
            show_image("image:%d"%(label),image)
        coord.request_stop()
        coord.join(threads)


def batch_test(record_file,resize_height, resize_width):
    '''
    :param record_file: record文件路径
    :param resize_height:
    :param resize_width:
    :return:
    :PS:image_batch, label_batch一般作为网络的输入
    '''
    # 读取record函数
    tf_image,tf_label = read_records(record_file,resize_height,resize_width,type='normalization')
    image_batch, label_batch= get_batch_images(tf_image,tf_label,batch_size=4,labels_nums=5,one_hot=False,shuffle=False)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:  # 开始一个会话
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(4):
            # 在会话中取出images和labels
            images, labels = sess.run([image_batch, label_batch])
            # 这里仅显示每个batch里第一张图片
            show_image("image", images[0, :, :, :])
            print('shape:{},tpye:{},labels:{}'.format(images.shape,images.dtype,labels))

        # 停止所有线程
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    # 参数设置

    resize_height = 224  # 指定存储图片高度
    resize_width = 224  # 指定存储图片宽度
    shuffle=True
    log=5
    # 产生train.record文件
    image_dir='dataset/train'
    train_labels = 'dataset/train.txt'  # 图片路径
    train_record_output = 'dataset/record/train{}.tfrecords'.format(resize_height)
    create_records(image_dir,train_labels, train_record_output, resize_height, resize_width,shuffle,log)
    train_nums=get_example_nums(train_record_output)
    print("save train example nums={}".format(train_nums))

    # 产生val.record文件
    image_dir='dataset/val'
    val_labels = 'dataset/val.txt'  # 图片路径
    val_record_output = 'dataset/record/val{}.tfrecords'.format(resize_height)
    create_records(image_dir,val_labels, val_record_output, resize_height, resize_width,shuffle,log)
    val_nums=get_example_nums(val_record_output)
    print("save val example nums={}".format(val_nums))

    # 测试显示函数
    # disp_records(train_record_output,resize_height, resize_width)
    batch_test(train_record_output,resize_height, resize_width)
