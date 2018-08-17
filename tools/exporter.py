#-*-coding:utf-8-*-
"""
    @Project: tensorflow_models_nets
    @File   : exporter.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2018-08-17 14:07:47
"""

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 15:13:27 2018

@author: shirhe-lyh
"""

"""Functions to export inference graph.

Modified from: TensorFlow models/research/object_detection/export.py
"""

import logging
import os
import tempfile
import tensorflow as tf

from tensorflow.core.protobuf import saver_pb2
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.client import session
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.training import saver as saver_lib

slim = tf.contrib.slim


# TODO: Replace with freeze_graph.freeze_graph_with_def_protos when
# newer version of Tensorflow becomes more common.
def freeze_graph_with_def_protos(
        input_graph_def,
        input_saver_def,
        input_checkpoint,
        output_node_names,
        restore_op_name,
        filename_tensor_name,
        clear_devices,
        initializer_nodes,
        variable_names_blacklist=''):
    """Converts all variables in a graph and checkpoint into constants."""
    del restore_op_name, filename_tensor_name  # Unused by updated loading code.

    # 'input_checkpoint' may be a prefix if we're using Saver V2 format
    if not saver_lib.checkpoint_exists(input_checkpoint):
        raise ValueError(
            "Input checkpoint ' + input_checkpoint + ' does not exist!")

    if not output_node_names:
        raise ValueError(
            'You must supply the name of a node to --output_node_names.')

    # Remove all the explicit device specifications for this node. This helps
    # to make the graph more portable.
    if clear_devices:
        for node in input_graph_def.node:
            node.device = ''

    with tf.Graph().as_default():
        tf.import_graph_def(input_graph_def, name='')
        config = tf.ConfigProto(graph_options=tf.GraphOptions())
        with session.Session(config=config) as sess:
            if input_saver_def:
                saver = saver_lib.Saver(saver_def=input_saver_def)
                saver.restore(sess, input_checkpoint)
            else:
                var_list = {}
                reader = pywrap_tensorflow.NewCheckpointReader(
                    input_checkpoint)
                var_to_shape_map = reader.get_variable_to_shape_map()
                for key in var_to_shape_map:
                    try:
                        tensor = sess.graph.get_tensor_by_name(key + ':0')
                    except KeyError:
                        # This tensor doesn't exist in the graph (for example
                        # it's 'global_step' or a similar housekeeping element)
                        # so skip it.
                        continue
                    var_list[key] = tensor
                saver = saver_lib.Saver(var_list=var_list)
                saver.restore(sess, input_checkpoint)
                if initializer_nodes:
                    sess.run(initializer_nodes)

            variable_names_blacklist = (variable_names_blacklist.split(',') if
                                        variable_names_blacklist else None)
            output_graph_def = graph_util.convert_variables_to_constants(
                sess,
                input_graph_def,
                output_node_names.split(','),
                variable_names_blacklist=variable_names_blacklist)
    return output_graph_def


def replace_variable_values_with_moving_averages(graph,
                                                 current_checkpoint_file,
                                                 new_checkpoint_file):
    """Replaces variable values in the checkpoint with their moving averages.

    If the current checkpoint has shadow variables maintaining moving averages
    of the variables defined in the graph, this function generates a new
    checkpoint where the variables contain the values of their moving averages.

    Args:
        graph: A tf.Graph object.
        current_checkpoint_file: A checkpoint both original variables and
            their moving averages.
        new_checkpoint_file: File path to write a new checkpoint.
    """
    with graph.as_default():
        variable_averages = tf.train.ExponentialMovingAverage(0.0)
        ema_variables_to_restore = variable_averages.variables_to_restore()
        with tf.Session() as sess:
            read_saver = tf.train.Saver(ema_variables_to_restore)
            read_saver.restore(sess, current_checkpoint_file)
            write_saver = tf.train.Saver()
            write_saver.save(sess, new_checkpoint_file)


def _image_tensor_input_placeholder(input_shape=None):
    """Returns input placeholder and a 4-D uint8 image tensor."""
    if input_shape is None:
        input_shape = (None, None, None, 3)
    input_tensor = tf.placeholder(
        dtype=tf.uint8, shape=input_shape, name='image_tensor')
    return input_tensor, input_tensor


def _encoded_image_string_tensor_input_placeholder():
    """Returns input that accepts a batch of PNG or JPEG strings.

    Returns:
        A tuple of input placeholder and the output decoded images.
    """
    batch_image_str_placeholder = tf.placeholder(
        dtype=tf.string,
        shape=[None],
        name='encoded_image_string_tensor')

    def decode(encoded_image_string_tensor):
        image_tensor = tf.image.decode_image(encoded_image_string_tensor,
                                             channels=3)
        image_tensor.set_shape((None, None, 3))
        return image_tensor

    return (batch_image_str_placeholder,
            tf.map_fn(
                decode,
                elems=batch_image_str_placeholder,
                dtype=tf.uint8,
                parallel_iterations=32,
                back_prop=False))


input_placeholder_fn_map = {
    'image_tensor': _image_tensor_input_placeholder,
    'encoded_image_string_tensor':
        _encoded_image_string_tensor_input_placeholder,
    #    'tf_example': _tf_example_input_placeholder,
}


def _add_output_tensor_nodes(postprocessed_tensors,
                             output_collection_name='inference_op'):
    """Adds output nodes.

    Adjust according to specified implementations.

    Adds the following nodes for output tensors:
        * classes: A float32 tensor of shape [batch_size] containing class
            predictions.

    Args:
        postprocessed_tensors: A dictionary containing the following fields:
            'classes': [batch_size].
        output_collection_name: Name of collection to add output tensors to.

    Returns:
        A tensor dict containing the added output tensor nodes.
    """
    outputs = {}
    classes = postprocessed_tensors.get('classes')  # Assume containing 'classes'
    outputs['classes'] = tf.identity(classes, name='classes')
    for output_key in outputs:
        tf.add_to_collection(output_collection_name, outputs[output_key])
    return outputs


def write_frozen_graph(frozen_graph_path, frozen_graph_def):
    """Writes frozen graph to disk.

    Args:
        frozen_graph_path: Path to write inference graph.
        frozen_graph_def: tf.GraphDef holding frozen graph.
    """
    with gfile.GFile(frozen_graph_path, 'wb') as f:
        f.write(frozen_graph_def.SerializeToString())
    logging.info('%d ops in the final graph.', len(frozen_graph_def.node))


def write_saved_model(saved_model_path,
                      frozen_graph_def,
                      inputs,
                      outputs):
    """Writes SavedModel to disk.

    If checkpoint_path is not None bakes the weights into the graph thereby
    eliminating the need of checkpoint files during inference. If the model
    was trained with moving averages, setting use_moving_averages to True
    restores the moving averages, otherwise the original set of variables
    is restored.

    Args:
        saved_model_path: Path to write SavedModel.
        frozen_graph_def: tf.GraphDef holding frozen graph.
        inputs: The input image tensor.
        outputs: A tensor dictionary containing the outputs of a slim model.
    """
    with tf.Graph().as_default():
        with session.Session() as sess:
            tf.import_graph_def(frozen_graph_def, name='')

            builder = tf.saved_model.builder.SavedModelBuilder(
                saved_model_path)

            tensor_info_inputs = {
                'inputs': tf.saved_model.utils.build_tensor_info(inputs)}
            tensor_info_outputs = {}
            for k, v in outputs.items():
                tensor_info_outputs[k] = tf.saved_model.utils.build_tensor_info(
                    v)

            detection_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs=tensor_info_inputs,
                    outputs=tensor_info_outputs,
                    method_name=signature_constants.PREDICT_METHOD_NAME))

            builder.add_meta_graph_and_variables(
                sess, [tf.saved_model.tag_constants.SERVING],
                signature_def_map={
                    signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                        detection_signature,
                },
            )
            builder.save()


def write_graph_and_checkpoint(inference_graph_def,
                               model_path,
                               input_saver_def,
                               trained_checkpoint_prefix):
    """Writes the graph and the checkpoint into disk."""
    for node in inference_graph_def.node:
        node.device = ''
    with tf.Graph().as_default():
        tf.import_graph_def(inference_graph_def, name='')
        with session.Session() as sess:
            saver = saver_lib.Saver(saver_def=input_saver_def,
                                    save_relative_paths=True)
            saver.restore(sess, trained_checkpoint_prefix)
            saver.save(sess, model_path)


def _get_outputs_from_inputs(input_tensors, model,
                             output_collection_name):
    inputs = tf.to_float(input_tensors)
    preprocessed_inputs = model.preprocess(inputs)
    output_tensors = model.predict(preprocessed_inputs)
    postprocessed_tensors = model.postprocess(output_tensors)
    return _add_output_tensor_nodes(postprocessed_tensors,
                                    output_collection_name)


def _build_model_graph(input_type, model, input_shape,
                       output_collection_name, graph_hook_fn):
    """Build the desired graph."""
    if input_type not in input_placeholder_fn_map:
        raise ValueError('Unknown input type: {}'.format(input_type))
    placeholder_args = {}
    if input_shape is not None:
        if input_type != 'image_tensor':
            raise ValueError("Can only specify input shape for 'image_tensor' "
                             'inputs.')
        placeholder_args['input_shape'] = input_shape
    placeholder_tensor, input_tensors = input_placeholder_fn_map[input_type](
        **placeholder_args)
    outputs = _get_outputs_from_inputs(
        input_tensors=input_tensors,
        model=model,
        output_collection_name=output_collection_name)

    # Add global step to the graph
    slim.get_or_create_global_step()

    if graph_hook_fn: graph_hook_fn()

    return outputs, placeholder_tensor


def export_inference_graph(input_type,
                           model,
                           trained_checkpoint_prefix,
                           output_directory,
                           input_shape=None,
                           use_moving_averages=None,
                           output_collection_name='inference_op',
                           additional_output_tensor_names=None,
                           graph_hook_fn=None):
    """Exports inference graph for the desired graph.

    Args:
        input_type: Type of input for the graph. Can be one of ['image_tensor',
            'encoded_image_string_tensor', 'tf_example']. In this file,
            input_type must be 'image_tensor'.
        model: A model defined by model.py.
        trained_checkpoint_prefix: Path to the trained checkpoint file.
        output_directory: Path to write outputs.
        input_shape: Sets a fixed shape for an 'image_tensor' input. If not
            specified, will default to [None, None, None, 3].
        use_moving_averages: A boolean indicating whether the
            tf.train.ExponentialMovingAverage should be used or not.
        output_collection_name: Name of collection to add output tensors to.
            If None, does not add output tensors to a collection.
        additional_output_tensor_names: List of additional output tensors to
            include in the frozen graph.
    """
    tf.gfile.MakeDirs(output_directory)
    frozen_graph_path = os.path.join(output_directory,
                                     'frozen_inference_graph.pb')
    saved_model_path = os.path.join(output_directory, 'saved_model')
    model_path = os.path.join(output_directory, 'model.ckpt')

    outputs, placeholder_tensor = _build_model_graph(
        input_type=input_type,
        model=model,
        input_shape=input_shape,
        output_collection_name=output_collection_name,
        graph_hook_fn=graph_hook_fn)

    saver_kwargs = {}
    if use_moving_averages:
        # This check is to be compatible with both version of SaverDef.
        if os.path.isfile(trained_checkpoint_prefix):
            saver_kwargs['write_version'] = saver_pb2.SaverDef.V1
            temp_checkpoint_prefix = tempfile.NamedTemporaryFile().name
        else:
            temp_checkpoint_prefix = tempfile.mkdtemp()
        replace_variable_values_with_moving_averages(
            tf.get_default_graph(), trained_checkpoint_prefix,
            temp_checkpoint_prefix)
        checkpoint_to_use = temp_checkpoint_prefix
    else:
        checkpoint_to_use = trained_checkpoint_prefix

    saver = tf.train.Saver(**saver_kwargs)
    input_saver_def = saver.as_saver_def()

    write_graph_and_checkpoint(
        inference_graph_def=tf.get_default_graph().as_graph_def(),
        model_path=model_path,
        input_saver_def=input_saver_def,
        trained_checkpoint_prefix=checkpoint_to_use)

    if additional_output_tensor_names is not None:
        output_node_names = ','.join(outputs.keys() +
                                     additional_output_tensor_names)
    else:
        output_node_names = ','.join(outputs.keys())

    frozen_graph_def = freeze_graph_with_def_protos(
        input_graph_def=tf.get_default_graph().as_graph_def(),
        input_saver_def=input_saver_def,
        input_checkpoint=checkpoint_to_use,
        output_node_names=output_node_names,
        restore_op_name='save/restore_all',
        filename_tensor_name='save/Const:0',
        clear_devices=True,
        initializer_nodes='')
    write_frozen_graph(frozen_graph_path, frozen_graph_def)
    write_saved_model(saved_model_path, frozen_graph_def,
                      placeholder_tensor, outputs)

