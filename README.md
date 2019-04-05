# tensorflow_models_learning
> 老铁要是觉得不错，给个“star”
## 1.生成record训练数据
dataset已经包含了训练和测试的图片，请直接运行create_tf_record.py</br>
> 对于InceptionNet V1:设置resize_height和resize_width = 224 </br>
> 对于InceptionNet V3:设置resize_height和resize_width = 299 </br>
> 其他模型，请根据输入需要设置resize_height和resize_width的大小</br>

```
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

```
## 2.训练过程
目前提供VGG、inception_v1、inception_v3、mobilenet_v以及resnet_v1的训练文件，只需要生成tfrecord数据，即可开始训练
> 训练VGG请直接运行：vgg_train_val.py </br>
> 训练inception_v1请直接运行：inception_v1_train_val.py </br>
> 训练inception_v3请直接运行：inception_v3_train_val.py </br>
> 训练mobilenet_v1请直接运行：mobilenet_train_val.py </br>
> 其他模型，请参考训练文件进行修改</br>

## 3.资源下载
- 本项目详细说明，请参考鄙人博客资料：
> 《使用自己的数据集训练GoogLenet InceptionNet V1 V2 V3模型》: https://panjinquan.blog.csdn.net/article/details/81560537 </br>
> 《tensorflow实现将ckpt转pb文件》: https://panjinquan.blog.csdn.net/article/details/82218092 </br>
> 《使用自己的数据集训练MobileNet、ResNet实现图像分类（TensorFlow）》https://panjinquan.blog.csdn.net/article/details/88252699
> 预训练模型下载地址: https://download.csdn.net/download/guyuealian/10610847  </br>
- 老铁要是觉得不错，给个“star”
- tensorflow-gpu==1.4.0
