#coding=utf-8

import tensorflow as tf 
import numpy as np 
import pdb
import cv2
import os
import glob
import slim.nets.inception_v3 as inception_v3

from create_tf_record import *
import tensorflow.contrib.slim as slim


def  predict(models_path,image_dir,labels_filename,labels_nums, data_format):
    [batch_size, resize_height, resize_width, depths] = data_format

    labels = np.loadtxt(labels_filename, str, delimiter='\t')
    input_images = tf.placeholder(dtype=tf.float32, shape=[None, resize_height, resize_width, depths], name='input')

    #其他模型预测请修改这里
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        out, end_points = inception_v3.inception_v3(inputs=input_images, num_classes=labels_nums, dropout_keep_prob=1.0, is_training=False)

    # 将输出结果进行softmax分布,再求最大概率所属类别
    score = tf.nn.softmax(out,name='pre')
    class_id = tf.argmax(score, 1)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, models_path)
    images_list=glob.glob(os.path.join(image_dir,'*.jpg'))
    for image_path in images_list:
        im=read_image(image_path,resize_height,resize_width,normalization=True)
        im=im[np.newaxis,:]
        #pred = sess.run(f_cls, feed_dict={x:im, keep_prob:1.0})
        pre_score,pre_label = sess.run([score,class_id], feed_dict={input_images:im})
        max_score=pre_score[0,pre_label]
        print("{} is: pre labels:{},name:{} score: {}".format(image_path,pre_label,labels[pre_label], max_score))
    sess.close()


if __name__ == '__main__':

    class_nums=5
    image_dir='test_image'
    labels_filename='dataset/label.txt'
    models_path='models/model.ckpt-10000'

    batch_size = 1  #
    resize_height = 299  # 指定存储图片高度
    resize_width = 299  # 指定存储图片宽度
    depths=3
    data_format=[batch_size,resize_height,resize_width,depths]
    predict(models_path,image_dir, labels_filename, class_nums, data_format)
