from caffe2keras import caffe_pb2 as caffe

layer_num_to_name = {
    0: 'NONE',
    1: 'ACCURACY',
    2: 'BNLL',
    3: 'CONCAT',
    4: 'CONVOLUTION',
    5: 'DATA',
    6: 'DROPOUT',
    7: 'EUCLIDEANLOSS',
    8: 'FLATTEN',
    9: 'HDF5DATA',
    10: 'HDF5OUTPUT',
    11: 'IM2COL',
    12: 'IMAGEDATA',
    13: 'INFOGAINLOSS',
    14: 'INNERPRODUCT',
    15: 'LRN',
    16: 'MULTINOMIALLOGISTICLOSS',
    17: 'POOLING',
    18: 'RELU',
    19: 'SIGMOID',
    20: 'SOFTMAX',
    21: 'SOFTMAXWITHLOSS',
    22: 'SPLIT',
    23: 'TANH',
    24: 'WINDOWDATA',
    25: 'ELTWISE',
    26: 'POWER',
    27: 'SIGMOIDCROSSENTROPYLOSS',
    28: 'HINGELOSS',
    29: 'MEMORYDATA',
    30: 'ARGMAX',
    31: 'THRESHOLD',
    32: 'DUMMY_DATA',
    33: 'SLICE',
    34: 'MVN',
    35: 'ABSVAL',
    36: 'SILENCE',
    37: 'CONTRASTIVELOSS',
    38: 'EXP',
    39: 'DECONVOLUTION'
}


def layer_type(layer):
    if type(layer.type) == int:
        typ = layer_num_to_name[layer.type]
    else:
        typ = str(layer.type)
    return typ.lower()


def extra_input_layer(config):
    if config.input:
        top_names = []
        top_shapes = []
        for input_idx in range(len(config.input)):
            in_name = config.input[input_idx]

            if config.input_shape:
                in_shape = config.input_shape
            elif config.input_dim:
                in_shape = caffe.BlobShape(dim=config.input_dim)
            else:
                raise ValueError("if input: occurs at top-level of network "
                                 "spec, it must be matched by input_shape or "
                                 "input_dim")

            top_names.append(in_name)
            top_shapes.append(in_shape)

        input_param = caffe.InputParameter(shape=top_shapes)

        return caffe.LayerParameter(
            name='dummy_input',
            top=top_names,
            type='Input',
            input_param=input_param)

    return None


def normalize_layers(config, phase):
    """Make layers well-behaved by removing inapplicable layers, adding in
    missing 'type: Input' layers, etc."""
    if len(config.layers) != 0:
        # layers = config.layers[:]  # prototext V1
        raise Exception("Prototxt files V1 are not supported.")
    elif len(config.layer) != 0:
        layers = config.layer[:]  # prototext V2
    else:
        raise Exception('could not load any layers from prototext')

    extra_input = extra_input_layer(config)
    if extra_input is not None:
        layers.insert(0, extra_input)

    rv = []
    for layer in layers:
        # skip things that may not be included in this phase
        include_phases = {
            i.phase
            for i in layer.include if i.phase is not None
        }
        if len(include_phases) > 0 and phase not in include_phases:
            continue
        exclude_phases = {
            i.phase
            for i in layer.exclude if i.phase is not None
        }
        if phase in exclude_phases:
            continue

        rv.append(layer)

    return rv


def get_output_names(layers):
    tops = set()
    bottoms = set()
    for layer in layers:
        tops.update(layer.top)
        bottoms.update(layer.bottom)
    # What appears as a top, but is never consumed by anything?
    return tops - bottoms


def is_data_input(layer):
    return layer_type(layer) in [
        'data', 'imagedata', 'memorydata', 'hdf5data', 'windowdata', 'input'
    ]


def is_caffe_layer(node):
    '''The node an actual layer'''
    if node.startswith('caffe_layer_'):
        return True
    return False


def sanitize(string):
    '''removes the added identification prefix 'caffe_layer' '''
    return int(string[12:])


def get_data_dim(layer):
    '''Finds the input dimension by parsing all data layers for image and image
    transformation details'''
    if layer_type(layer) == 'data' or layer_type(layer) == 'imagedata':
        # DATA or IMAGEDATA layers
        try:
            scale = layer.transform_param.scale
            if scale <= 0:
                scale = 1
        except AttributeError:
            pass

        try:
            side = layer.transform_param.crop_size * scale
            return [3, side, side]
        except AttributeError:
            pass

        try:
            height = layer.image_param.new_height * scale
            width = layer.image_param.new_width * scale
            return [3, height, width]
        except AttributeError:
            pass
    return []
