from keras.layers.core import Layer


class Select(Layer):
    """Selects a configurable part of an input tensor. Useful for dividing
    tensors up to make them go different places."""

    def __init__(self, start_or_stop, stop=None, step=None, axis=1, **kwargs):
        super(Select, self).__init__(**kwargs)
        self.slice = slice(start_or_stop, stop, step)
        self.axis = axis

    def call(self, x, **kwargs):
        pad_axes = [slice(None, None, None)] * self.axis
        axes = pad_axes + [self.slice]
        return x[axes]

    def compute_output_shape(self, input_shape):
        start, stop, stride = self.slice.indices(input_shape[self.axis])
        out_ax_size = (stop - start) // stride
        prefix = input_shape[:self.axis]
        suffix = input_shape[self.axis + 1:]
        return prefix + (out_ax_size, ) + suffix

    def get_config(self):
        return {
            'start': self.slice.start,
            'stop': self.slice.stop,
            'step': self.slice.step,
            'axis': self.axis
        }
