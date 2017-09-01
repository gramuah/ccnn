import six

from functools import wraps

from keras.layers import (ZeroPadding2D, Dropout, Conv2D, Flatten, Dense,
                          BatchNormalization, Activation, MaxPooling2D,
                          AveragePooling2D, Input, Multiply, Add, Maximum,
                          Concatenate)
from keras.models import Model

from caffe2keras import caffe_pb2 as caffe
import google.protobuf
import google.protobuf.text_format
from caffe2keras.caffe_utils import (layer_type, normalize_layers,
                                     get_output_names, is_data_input)
from caffe2keras.extra_layers import Select

import numpy as np

# manipulated by caffe2keras
debug = False
# maps layer type names --> converter functions
_converters = {}


def construct(type_name, num_bottoms=1, num_tops=1):
    """Decorator used to register a constructor for some type of Caffe layer.
    The constructor is expected to take a layer spec and list of bottoms (Keras
    tensors), then return the appropriate tops (also Keras tensors)."""
    # numbers can be constants or '+' (at-least-one)
    assert isinstance(num_bottoms, int) or num_bottoms == '+'
    assert isinstance(num_tops, int) or num_tops == '+'

    def take_func(converter_func):
        @wraps(converter_func)
        def wrapper(spec, bottoms):
            name = spec.name

            # verify bottom coutn
            nbot = len(bottoms)
            if num_bottoms == '+':
                assert nbot >= 1, "Expected one or more bottoms at %s," \
                    " got %d" % (name, nbot)
            else:
                assert nbot == num_bottoms, "Expected %d bottoms at %s," \
                    " got %d" % (num_bottoms, name, nbot)

            # don't pass in list when there's always one bottom
            if num_bottoms == 1:
                bot_arg = bottoms[0]
            else:
                bot_arg = bottoms
            rv = converter_func(spec, bot_arg)
            if num_tops == 1 and not isinstance(rv, (list, tuple)):
                # can return only one result here
                rv = [rv]

            # verify top count
            ntop = len(rv)
            if num_tops == '+':
                assert ntop >= 1, "Expected one or more tops at %s," \
                    " got %d from converter" % (name, ntop)
            else:
                assert ntop == num_tops, "Expected %d tops at %s," \
                    " got %d from converter" % (num_tops, name, ntop)

            return rv

        # register in global dict
        if isinstance(type_name, six.string_types):
            type_names = [type_name]
        else:
            type_names = type_name
        for tn in type_names:
            assert tn not in _converters, \
                "double-registered handler for %s" % tn
            _converters[tn] = wrapper

        return wrapper

    return take_func


@construct('concat', num_bottoms='+')
def handle_concat(spec, bottoms):
    axis = spec.concat_param.axis
    return Concatenate(axis=axis, name=spec.name)(bottoms)


@construct('convolution')
def handle_conv(spec, bottom):
    has_bias = spec.convolution_param.bias_term
    nb_filter = spec.convolution_param.num_output
    nb_col = (spec.convolution_param.kernel_size or
              [spec.convolution_param.kernel_h])[0]
    nb_row = (spec.convolution_param.kernel_size or
              [spec.convolution_param.kernel_w])[0]
    stride_h = (spec.convolution_param.stride or
                [spec.convolution_param.stride_h])[0] or 1
    stride_w = (spec.convolution_param.stride or
                [spec.convolution_param.stride_w])[0] or 1
    pad_h = (spec.convolution_param.pad or [spec.convolution_param.pad_h])[0]
    pad_w = (spec.convolution_param.pad or [spec.convolution_param.pad_w])[0]
    dilation = spec.convolution_param.dilation or (1, 1)

    if debug:
        print("kernel")
        print(str(nb_filter) + 'x' + str(nb_col) + 'x' + str(nb_row))
        print("stride")
        print(stride_h)
        print("pad")
        print(pad_h)

    if pad_h + pad_w > 0:
        bottom = ZeroPadding2D(
            padding=(pad_h, pad_w),
            name=spec.name + '_zeropadding',
            data_format='channels_first')(bottom)

    # XXX: I remember this sometimes had an off-by-one error on the output
    # shape. After going through the output computation formulae for both Caffe
    # & Keras, I can't see where the problem would lie (see NOTES.md). However,
    # if there's an off-by-one error in output size, then it's probable that
    # this code (and possibly my analysis) is incorrect.
    return Conv2D(
        nb_filter,
        kernel_size=(nb_col, nb_row),
        strides=(stride_h, stride_w),
        use_bias=has_bias,
        name=spec.name,
        dilation_rate=dilation,
        data_format='channels_first')(bottom)


@construct('dropout')
def handle_dropout(spec, bottom):
    prob = spec.dropout_param.dropout_ratio
    return Dropout(prob, name=spec.name)(bottom)


@construct('flatten')
def handle_flatten(spec, bottom):
    return Flatten(name=spec.name)(bottom)


@construct('innerproduct')
def handle_dense(spec, bottom):
    name = spec.name
    output_dim = spec.inner_product_param.num_output

    if len(bottom._keras_shape[1:]) > 1:
        bottom = Flatten(name=name + '_flatten')(bottom)

    return Dense(output_dim, name=name)(bottom)


@construct('pooling')
def handle_pooling(spec, bottom):
    kernel_h = spec.pooling_param.kernel_size or spec.pooling_param.kernel_h
    kernel_w = spec.pooling_param.kernel_size or spec.pooling_param.kernel_w

    # caffe defaults to 1, hence both of the params can be zero. 'or 1'
    stride_h = spec.pooling_param.stride or spec.pooling_param.stride_h or 1
    stride_w = spec.pooling_param.stride or spec.pooling_param.stride_w or 1

    pad_h = spec.pooling_param.pad or spec.pooling_param.pad_h
    pad_w = spec.pooling_param.pad or spec.pooling_param.pad_w

    if debug:
        print("kernel")
        print(str(kernel_h) + 'x' + str(kernel_w))
        print("stride")
        print(stride_h)
        print("pad")
        print(pad_h)
        print(pad_w)

    # XXX: This sometimes produces outputs which are too small by ~1px. IIRC
    # Caffe uses a different method to Keras for computing output sizes. I've
    # been using (fake) padding in my protoxtxts to get around the problem, but
    # this should be fixed properly at some point.
    if pad_h + pad_w > 0:
        bottom = ZeroPadding2D(
            padding=(pad_h, pad_w),
            name=spec.name + '_zeropadding',
            data_format='channels_first')(bottom)
    if spec.pooling_param.pool == 0:  # MAX pooling
        # border_mode = 'same'
        border_mode = 'valid'
        if debug:
            print("MAX pooling")
        return MaxPooling2D(
            padding=border_mode,
            pool_size=(kernel_h, kernel_w),
            strides=(stride_h, stride_w),
            name=spec.name,
            data_format='channels_first')(bottom)
    elif (spec.pooling_param.pool == 1):  # AVE pooling
        if debug:
            print("AVE pooling")
        return AveragePooling2D(
            pool_size=(kernel_h, kernel_w),
            strides=(stride_h, stride_w),
            name=spec.name,
            data_format='channels_first')(bottom)

    # Stochastic pooling still needs to be implemented
    raise NotImplementedError(
        "Only MAX and AVE pooling are implemented in keras!")


@construct('relu')
def handle_relu(spec, bottom):
    return Activation('relu', name=spec.name)(bottom)


@construct('sigmoid')
def handle_sigmoid(spec, bottom):
    return Activation('sigmoid', name=spec.name)(bottom)


@construct(['softmax', 'softmaxwithloss'])
def handle_softmax(spec, bottom):
    return Activation('softmax', name=spec.name)(bottom)


@construct('tanh')
def handle_tanh(spec, bottom):
    return Activation('tanh', name=spec.name)(bottom)


@construct('eltwise', num_bottoms='+')
def handle_eltwise(spec, bottoms):
    # axis = spec.scale_param.axis
    op = spec.eltwise_param.operation  # PROD=0, SUM=1, MAX=2
    if op == 0:
        Merger = Multiply
    elif op == 1:
        Merger = Add
    elif op == 2:
        Merger = Maximum
    else:
        raise NotImplementedError(
            'Operation with id=%d of eltwise layer is not implemented' % op)
    return Merger(name=spec.name)(bottoms)


def _make_slicer(slices):
    # this function exists because Python makes it hard to construct closures
    # in loops
    return lambda l: l[slices]


@construct('slice', num_tops='+')
def handle_slice(spec, bottom):
    sp = spec.slice_param
    slice_points = sp.slice_point
    assert len(slice_points) + 1 == len(spec.top), \
        "slice points must be one less than top count at %s" % spec.name
    axis = sp.axis or sp.slice_dim

    if debug:
        print('-- Slice (%s)' % spec.name)
        top_j = ', '.join(spec.top)
        slice_j = ', '.join(map(str, slice_points))
        print('Axis %d' % axis)
        print('Feeding to %s at slice points %s' % (top_j, slice_j))

    rv = []
    for top_idx in range(len(spec.top)):
        if top_idx > 0:
            slice_begin = slice_points[top_idx - 1]
        else:
            slice_begin = 0
        if top_idx < len(slice_points):
            slice_end = slice_points[top_idx]
        else:
            slice_end = None
        top_name = spec.top[top_idx]
        out = Select(slice_begin, slice_end, name=top_name)(bottom)
        rv.append(out)

        if debug:
            print('Slices for top %s: %s:%s' %
                  (top_name, str(slice_begin), str(slice_end)))

    return rv


@construct('batchnorm')
def handle_batch_norm(spec, bottom):
    axis = spec.scale_param.axis
    epsilon = spec.batch_norm_param.eps
    decay = spec.batch_norm_param.moving_average_fraction or 0.999

    if debug:
        print('-- BatchNormalization')
        print(spec.name)
        print('axis')
        print(axis)

    return BatchNormalization(
        epsilon=epsilon,
        momentum=decay,
        axis=axis,
        name=spec.name,
        data_format='channels_first')(bottom)


@construct('input', num_bottoms=0, num_tops='+')
def handle_input(spec, bottoms):
    all_shapes = spec.input_param.shape
    rv = []
    # XXX: shape handling here isn't very intelligent. It's okay when you
    # actually have an Input layer, but in real nets you need to figure out the
    # input shape from training data or the sizes of things downstream.
    if len(all_shapes) == len(spec.top):
        # 1:1 mapping between shapes and tops
        for shape, top_name in zip(all_shapes, spec.top):
            new_in = Input(batch_shape=shape.dim, name=top_name)
            rv.append(new_in)
    elif len(all_shapes) == 1:
        # copy same input (with same shape) to all tops
        shape, = all_shapes
        for top_name in spec.top:
            new_in = Input(batch_shape=shape.dim, name=spec.name)
            rv.append(new_in)
    else:
        raise ValueError("Number of output shapes (%d) should be 1, or "
                         "should match number of tops (%d)!" %
                         (len(all_shapes), len(spec.tops)))
    return rv


def _set_debug(is_debug):
    # function to get around naming of caffe_to_keras params :)
    global debug
    debug = is_debug


def caffe_to_keras(prototext, caffemodel, phase='train', debug=False):
    '''
        Converts a Caffe Graph into a Keras Graph
        prototext: model description file in caffe
        caffemodel: stored weights file
        phase: train or test

        Usage:
            model = caffe_to_keras('VGG16.prototxt', 'VGG16_700iter.caffemodel')
    '''
    _set_debug(debug)
    config = caffe.NetParameter()
    prototext = preprocess_prototxt(prototext)
    google.protobuf.text_format.Merge(prototext, config)

    print("CREATING MODEL")
    model = create_model(config, 0 if phase == 'train' else 1,
                         tuple(config.input_dim[1:]))

    print('Reading caffemodel')
    params = caffe.NetParameter()
    with open(caffemodel, 'rb') as fp:
        model_binary = fp.read()
    params.MergeFromString(model_binary)

    if len(params.layers) != 0:
        param_layers = params.layers[:]  # V1
        v = 'V1'
    elif len(params.layer) != 0:
        param_layers = params.layer[:]  # V2
        v = 'V2'
    else:
        raise Exception('could not load any layers from caffemodel')

    print("Printing the converted model:")
    model.summary()

    print('')
    print("LOADING WEIGHTS")
    weights = convert_weights(param_layers, v)

    load_weights(model, weights)

    return model


def preprocess_prototxt(prototxt):
    with open(prototxt) as fp:
        p = fp.readlines()

    for i, line in enumerate(p):
        l = line.strip().replace(" ", "").split('#')[0]
        # Change "layers {" to "layer {"
        # if len(l) > 6 and l[:7] == 'layers{':
        #     p[i] = 'layer {'
        # Write all layer types as strings
        if len(l) > 6 and l[:5] == 'type:' and l[5] != "\'" and l[5] != '\"':
            type_ = l[5:]
            p[i] = '  type: "' + type_ + '"'
        # blobs_lr
        # elif len(l) > 9 and l[:9] == 'blobs_lr:':
        #     print("The prototxt parameter 'blobs_lr' found in line "+str(i+1)+" is outdated and will be removed. Consider using param { lr_mult: X } instead.")
        #    p[i] = ''
        #
        # elif len(l) > 13 and l[:13] == 'weight_decay:':
        #     print("The prototxt parameter 'weight_decay' found in line "+str(i+1)+" is outdated and will be removed. Consider using param { decay_mult: X } instead.")
        #     p[i] = ''

    p = '\n'.join(p)
    if debug:
        print('Writing preprocessed prototxt to debug.prototxt')
        with open('debug.prototxt', 'w') as f:
            f.write(p)
    return p


def create_model(config, phase, input_dim):
    '''
        layers:
            a list of all the layers in the model
        phase:
            parameter to specify which network to extract: training or test
        input_dim:
            `input dimensions of the configuration (if in model is in deploy mode)
    '''
    layers = normalize_layers(config, phase)
    output_names = get_output_names(layers)

    # May as well have a Caffe terminology reference here (took me a while to
    # grok):
    #
    # *blob:* memory location which can store activations and gradients.
    # *layer:* module which takes some "bottom" blobs and writes out new
    #   activations to some "top" blobs (or does the reverse on the backwards
    #   pass).
    # *net:* collection of blobs, joined up by layers.
    #
    # Note that you can have situations where a layer has the *same* bottom and
    # top (i.e. it's operating in-place). Hence, a network's implicit blob
    # "graph" can have (multiple) self-loops.

    # stores computed outputs for each layer; blobs may be overwritten when
    # blob graph is not acyclic
    blobs = {}
    model_inputs = []
    model_outputs = []

    for layer in layers:
        if debug:
            print('\nBuilding layer %s' % layer.name)

        name = layer.name
        type_of_layer = layer_type(layer)
        tops = layer.top
        bottoms = layer.bottom

        layer_bottom_blobs = []
        for bottom in bottoms:
            # this assertion might get raised if the layer in the .prototxt
            # aren't toposorted
            assert bottom in blobs, \
                "Must have computed bottom %s before reaching layer %s" \
                % (bottom, name)
            layer_bottom_blobs.append(blobs[bottom])
        if debug and layer_bottom_blobs:
            print('input shapes: ' + ', '.join(
                str(b._keras_shape) for b in layer_bottom_blobs))

        if type_of_layer in _converters:
            converter = _converters[type_of_layer]
            out_blobs = converter(layer, layer_bottom_blobs)
            assert len(out_blobs) == len(tops)
            for blob, blob_name in zip(out_blobs, tops):
                blobs[blob_name] = blob

                if blob_name in output_names:
                    model_outputs.append(blob)

                if is_data_input(layer):
                    model_inputs.append(blob)
        else:
            raise RuntimeError('layer type', type_of_layer,
                               'used in this model is not currently supported')

    assert len(model_inputs) >= 1, "No inputs detected"
    assert len(model_outputs) >= 1, "No outputs detected"

    model = Model(inputs=model_inputs, outputs=model_outputs)

    return model


def rot90(W):
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W[i, j] = np.rot90(W[i, j], 2)
    return W


def convert_weights(param_layers, v='V1'):
    weights = {}

    for layer in param_layers:
        typ = layer_type(layer)
        if typ == 'innerproduct':
            blobs = layer.blobs

            if (v == 'V1'):
                nb_filter = blobs[0].num
                stack_size = blobs[0].channels
                nb_col = blobs[0].height
                nb_row = blobs[0].width
            elif (v == 'V2'):
                if (len(blobs[0].shape.dim) == 4):
                    nb_filter = int(blobs[0].shape.dim[0])
                    stack_size = int(blobs[0].shape.dim[1])
                    nb_col = int(blobs[0].shape.dim[2])
                    nb_row = int(blobs[0].shape.dim[3])
                else:
                    nb_filter = 1
                    stack_size = 1
                    nb_col = int(blobs[0].shape.dim[0])
                    nb_row = int(blobs[0].shape.dim[1])
            else:
                raise RuntimeError('incorrect caffemodel version "' + v + '"')

            weights_p = np.array(blobs[0].data).reshape(
                nb_filter, stack_size, nb_col, nb_row)[0, 0, :, :]
            weights_p = weights_p.T  # need to swapaxes here, hence transpose. See comment in conv
            weights_b = np.array(blobs[1].data)
            layer_weights = [
                weights_p.astype(dtype=np.float32),
                weights_b.astype(dtype=np.float32)
            ]

            weights[layer.name] = layer_weights

        elif typ == 'batchnorm':
            blobs = layer.blobs
            if (v == 'V2'):
                nb_kernels = int(blobs[0].shape.dim[0])
            else:
                raise NotImplementedError(
                    'Conversion on layer type "' + typ +
                    '"not implemented forcaffemodel version "' + v + '"')

            weights_mean = np.array(blobs[0].data)
            weights_std_dev = np.array(blobs[1].data)

            weights[layer.name] = [
                np.ones(nb_kernels), np.zeros(nb_kernels),
                weights_mean.astype(dtype=np.float32),
                weights_std_dev.astype(dtype=np.float32)
            ]

        elif typ == 'scale':
            blobs = layer.blobs
            if (v == 'V2'):
                nb_gamma = int(blobs[0].shape.dim[0])
                nb_beta = int(blobs[1].shape.dim[0])
                assert nb_gamma == nb_beta
            else:
                raise NotImplementedError(
                    'Conversion on layer type "' + typ +
                    '"not implemented forcaffemodel version "' + v + '"')

            weights_gamma = np.array(blobs[0].data)
            weights_beta = np.array(blobs[1].data)

            weights[layer.name] = [
                weights_gamma.astype(dtype=np.float32),
                weights_beta.astype(dtype=np.float32)
            ]

        elif typ == 'convolution':
            blobs = layer.blobs

            if (v == 'V1'):
                nb_filter = blobs[0].num
                temp_stack_size = blobs[0].channels
                nb_col = blobs[0].height
                nb_row = blobs[0].width
            elif (v == 'V2'):
                nb_filter = int(blobs[0].shape.dim[0])
                temp_stack_size = int(blobs[0].shape.dim[1])
                nb_col = int(blobs[0].shape.dim[2])
                nb_row = int(blobs[0].shape.dim[3])
            else:
                raise RuntimeError('incorrect caffemodel version "' + v + '"')

            # NOTE: on model parallel networks, if group is > 1, that means the
            # conv filters are split up into a number of 'groups' and each
            # group lies on a seperate GPU. Each group only acts on the select
            # group of outputs from previous layer that was in the same GPU
            # (not the entire stack). Here, we add zeros to simulate the same
            # effect. This was famously used in AlexNet and few other models
            # from 2012-14.

            group = layer.convolution_param.group
            stack_size = temp_stack_size * group

            weights_p = np.zeros((nb_filter, stack_size, nb_col, nb_row))

            if layer.convolution_param.bias_term:
                weights_b = np.array(blobs[1].data)
            else:
                weights_b = None

            group_data_size = len(blobs[0].data) // group
            stacks_size_per_group = stack_size // group
            nb_filter_per_group = nb_filter // group

            if debug:
                print(layer.name)
                print("nb_filter")
                print(nb_filter)
                print("(channels x height x width)")
                print("(" + str(temp_stack_size) + " x " + str(nb_col) + " x "
                      + str(nb_row) + ")")
                print("groups")
                print(group)

            for i in range(group):
                group_weights = weights_p[
                    i*nb_filter_per_group:(i+1)*nb_filter_per_group,
                    i*stacks_size_per_group:(i+1)*stacks_size_per_group,
                    :, :]
                blob_d = blobs[0].data
                blob_gw = blob_d[i*group_data_size:(i+1)*group_data_size]
                group_weights[:] \
                    = np.array(blob_gw).reshape(group_weights.shape)

            # caffe, unlike theano, does correlation not convolution. We need
            # to flip the weights 180 deg
            weights_p = rot90(weights_p)

            # Keras needs h*w*i*o filters (where d is input, o is output), so
            # we transpose
            weights_p = weights_p.transpose((3, 2, 1, 0))

            if weights_b is not None:
                layer_weights = [
                    weights_p.astype(dtype=np.float32),
                    weights_b.astype(dtype=np.float32)
                ]
            else:
                layer_weights = [weights_p.astype(dtype=np.float32)]

            weights[layer.name] = layer_weights

    return weights


def load_weights(model, weights):
    for layer in model.layers:
        # TODO: add a check to make sure we're not jumping over any layers with
        # trainable weights
        if layer.name in weights:
            print('Copying weights for %s' % layer.name)
            model.get_layer(layer.name).set_weights(weights[layer.name])
        elif not layer.trainable_weights:
            # this is fine; we don't expect weights
            print('No weights for untrainable layer %s' % layer.name)
        else:
            # this isn't fine; if there are trainable weights, they should
            # probably be in the param file
            print('(!!) No weights for trainable layer %s, but it should have '
                  'weights (?!). Does the parameter file match the .prototxt?'
                  % layer.name)
