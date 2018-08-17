#-*-coding:utf-8-*-
"""
    @Project: tensorflow_models_nets
    @File   : export_inference_graph.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2018-08-17 14:04:04
"""

import argparse

parser = argparse.ArgumentParser()
# 命令行解析，help是提示符，type是输入的类型，
parser.add_argument("file", type=str, help="input ckpt model dir")
aggs = parser.parse_args()
print(aggs)