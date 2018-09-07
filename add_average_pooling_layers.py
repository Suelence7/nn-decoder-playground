from keras.layers import AveragePooling2D, Flatten
from keras.models import Model, Sequential


def add_average_pooling_layers(pretrained_model, tensor_idxs):
    """ Adds an avearge pooling layer to each of the specified layers. """
    layers = [add_average_pooling_node(pretrained_model, idx)
              for idx in tensor_idxs]

    graph_input = pretrained_model.input
    outputs = [node.output for node in layers]

    multi_output_model = Model(inputs=graph_input, outputs=outputs)

    return multi_output_model


def add_average_pooling_node(pretrained_model, layer_idx):
    """ Adds an average pooling layer to a computational graph.
    """
    # Get layer of interest and the output tensor shape.
    layer = pretrained_model.layers[layer_idx].output
    data_shape = get_layer_shape(layer)

    # Build wrapper to average pool features of that layer.
    avg_pool = Sequential()
    avg_pool.add(
        AveragePooling2D(
            input_shape=data_shape,
            pool_size=(data_shape[0], data_shape[1]),
            padding='valid',
            strides=(1, 1),
            data_format='channels_last',
            name='avg_pool'
        )
    )
    avg_pool.add(
        Flatten(name='flatten')
    )

    # Build average pooling "model".
    pooling_inputs = pretrained_model.input
    pooling_outputs = avg_pool(layer)

    pooling_model = Model(inputs=pooling_inputs,
                          outputs=pooling_outputs)

    return pooling_model


def get_layer_shape(layer):
    return tuple(layer.shape.as_list()[1:])
