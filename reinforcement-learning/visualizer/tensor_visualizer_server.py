from flask import Flask, render_template, request
from base64 import b64encode
from io import BytesIO

import numpy as np
from PIL import Image

from models.tensor_visualizer import TensorVisualizer, TensorNode
from utils.list_utils import distinct

IMG_SIZE = (200, None)
IMG_SCALE = 0.5
NORMALIZE_BOUND = 255.0
INPUT_IMAGE_COUNT = 1

app = Flask(__name__)
app.jinja_env.add_extension('jinja2.ext.loopcontrols')
all_images_dict = {}


@app.route('/<dir_name>', defaults={'snapshot_number': 1})
@app.route('/<dir_name>/<snapshot_number>')
def index(dir_name, snapshot_number):
    all_images, snapshot_numbers, layer_info = load_images(dir_name)
    snapshot_idx, arg_img_idx, layer_name, image_indices, layer_indices, layer_names =\
        parse_arguments(all_images, layer_info, snapshot_numbers, snapshot_number)

    return render_template('index.html', all_images=all_images, dir_name=dir_name, snapshot_idx=snapshot_idx,
                           img_idx=arg_img_idx, layer_names=layer_names, layer_name=layer_name,
                           snapshot_numbers=snapshot_numbers, image_indices=image_indices, layer_indices=layer_indices)


@app.route('/<dir_name>/<snapshot_number>/images')
def images(dir_name, snapshot_number):
    all_images, snapshot_numbers, layer_info = load_images(dir_name)
    snapshot_idx, arg_img_idx, layer_name, image_indices, layer_indices, layer_names =\
        parse_arguments(all_images, layer_info, snapshot_numbers, snapshot_number)

    return render_template('images.html', all_images=all_images, dir_name=dir_name, snapshot_idx=snapshot_idx,
                           img_idx=arg_img_idx, layer_names=layer_names, layer_name=layer_name,
                           snapshot_numbers=snapshot_numbers, image_indices=image_indices, layer_indices=layer_indices)


def parse_arguments(all_images, layer_info, snapshot_numbers, snapshot_number):
    try:
        snapshot_number = int(snapshot_number)
    except ValueError as e:
        # TODO: Handle properly
        raise e

    snapshot_number = snapshot_number if snapshot_number in snapshot_numbers else snapshot_numbers[0]

    layer_name = request.args.get('layer') or TensorNode.Layer.CONV.value
    arg_img_idx = int(request.args.get('img_idx') or 0)

    layer_names = distinct(layer_info, lambda layer: layer.value)
    layer_indices = [idx for idx, layer in enumerate(layer_info) if layer.value == layer_name]
    snapshot_idx = snapshot_numbers.index(snapshot_number)
    image_count = len(all_images[snapshot_idx][0])

    image_indices = list(range(image_count))

    return snapshot_idx, arg_img_idx, layer_name, image_indices, layer_indices, layer_names


def parse_snapshot_range(snapshot_numbers):
    arg_snapshot_number_from = request.args.get('snapshot_from') or snapshot_numbers[0]
    arg_snapshot_number_to = request.args.get('snapshot_to') or snapshot_numbers[-1]

    # noinspection PyBroadException
    def _index(arr, element, default=None):
        try:
            return arr.index(element)
        except:
            return default

    def index_or_first(arr, element):
        return _index(arr, element, 0)

    def index_or_last(arr, element):
        return _index(arr, element, len(arr))

    arg_snapshot_idx_from = index_or_first(snapshot_numbers, int(arg_snapshot_number_from))
    arg_snapshot_idx_to = index_or_last(snapshot_numbers, int(arg_snapshot_number_to))

    return arg_snapshot_idx_from, arg_snapshot_idx_to + 1


def load_images(dir_name):
    global all_images_dict
    if dir_name in all_images_dict:
        return all_images_dict[dir_name]

    TensorVisualizer.instance.id = dir_name
    snapshot_numbers = TensorVisualizer.instance.snapshot_numbers

    arg_snapshot_idx_from, arg_snapshot_idx_to = parse_snapshot_range(snapshot_numbers)
    histories = TensorVisualizer.instance.load_histories(from_index=arg_snapshot_idx_from, to_index=arg_snapshot_idx_to)
    snapshot_numbers = TensorVisualizer.instance.snapshot_numbers[arg_snapshot_idx_from:arg_snapshot_idx_to]
    layer_info = TensorVisualizer.instance.layer_info

    all_images = []

    for episode in histories:
        inputs, input_gradients, layers = episode
        inputs = convert_images_into_b64_images(inputs)
        input_gradients = convert_images_into_b64_images(input_gradients)

        all_layer_images = []

        for idx in range(len(layers)):
            layer = layers[idx]
            layer_mode = layer_info[idx]
            gradient, kernel, activation = layer

            kernel_group_size = kernel.shape[2] if layer_mode == TensorNode.Layer.CONV else 1
            # activation_group_size = activation.shape[-1] if is_conv_layer else 1
            activation_group_size = 1

            layer_converter, activation_converter, activation_means_converter, gradient_means_converter =\
                get_converters(layer_mode)

            conv_kernels = layer_converter(kernel, resize_to=IMG_SIZE, normalize_bound=NORMALIZE_BOUND)
            conv_means = gradient_means_converter(kernel)
            activation_images = [activation_converter(activation[i:i+1], resize_to=IMG_SIZE,
                                                      normalize_bound=NORMALIZE_BOUND) for i in range(len(activation))]
            activation_means = [activation_means_converter(activation[i:i+1]) for i in range(len(activation))]
            gradient_images = [layer_converter(g, resize_to=IMG_SIZE, normalize_bound=NORMALIZE_BOUND)
                               for g in gradient]
            gradient_means = [gradient_means_converter(g) for g in gradient]

            conv_kernel_info = (conv_kernels, conv_means, kernel_group_size)
            activation_info = (activation_images, activation_means, activation_group_size)
            gradient_info = (gradient_images, gradient_means, kernel_group_size)

            layer_images = [conv_kernel_info, activation_info, gradient_info]

            all_layer_images.append(layer_images)
        all_images.append([inputs, input_gradients, all_layer_images])

    all_images_dict[dir_name] = (all_images, snapshot_numbers, layer_info)

    return all_images_dict[dir_name]


def get_converters(layer_mode):
    if layer_mode == TensorNode.Layer.CONV:
        layer_converter = convert_conv_layer_into_b64_images
        activation_converter = convert_conv_activation_layer_into_b64_images
        activation_means_converter = calc_conv_activation_layer_means
        gradient_means_converter = calc_conv_gradient_layer_means
    elif layer_mode == TensorNode.Layer.DENSE:
        layer_converter = convert_dense_layer_into_b64_image
        activation_converter = convert_dense_activation_layer_into_b64_images
        activation_means_converter = calc_dense_activation_layer_means
        gradient_means_converter = calc_dense_gradient_layer_means
    else:
        raise NotImplementedError('Layer mode \'{}\' not supported yet.'.format(layer_mode))

    return layer_converter, activation_converter, activation_means_converter, gradient_means_converter


def convert_images_into_b64_images(images):
    return [convert_array_into_b64_image(np.squeeze(image, axis=-1), resize_to=IMG_SIZE,
                                         normalize_bound=NORMALIZE_BOUND, ptp=np.ptp(images)) for image in images]


def calc_max_ptp_for_gradient(layer):
    max_ptp = 0
    max_ptp_idx = None
    for i in range(layer.shape[2]):
        for j in range(layer.shape[3]):
            ptp = np.ptp(layer[:, :, i, j])
            if ptp > max_ptp:
                max_ptp = ptp
                max_ptp_idx = (i, j)

    return max_ptp, max_ptp_idx


def calc_max_ptp_for_conv_activation(layer):
    max_ptp = 0
    max_ptp_idx = None
    for i in range(layer.shape[0]):
        for j in range(layer.shape[3]):
            ptp = np.ptp(layer[i, :, :, j])
            if ptp > max_ptp:
                max_ptp = ptp
                max_ptp_idx = (i, j)

    return max_ptp, max_ptp_idx


def calc_max_ptp_for_dense_activation(layer):
    max_ptp = 0
    max_ptp_idx = None
    for i, act in enumerate(layer):
        ptp = np.ptp(act)
        if ptp > max_ptp:
            max_ptp = ptp
            max_ptp_idx = i

    return max_ptp, max_ptp_idx


def convert_conv_layer_into_b64_images(layer, resize_to=None, normalize_bound=None):
    b64_images = []
    channel_cnt, filter_cnt = layer.shape[2], layer.shape[3]
    ptp, _ = calc_max_ptp_for_gradient(layer)

    for channel_idx in range(channel_cnt):
        for filter_idx in range(filter_cnt):
            image = layer[:, :, channel_idx, filter_idx]
            b64_image = convert_array_into_b64_image(image, resize_to=resize_to,
                                                     normalize_bound=normalize_bound, ptp=ptp)
            b64_images.append(b64_image)

    return b64_images


def convert_dense_layer_into_b64_image(layer, resize_to=None, normalize_bound=None):
    ptp = np.ptp(layer)
    return [convert_array_into_b64_image(layer, resize_to=resize_to, normalize_bound=normalize_bound, ptp=ptp)]


def calc_conv_activation_layer_means(activation_layer):
    means = []
    for activation_per_batch in activation_layer:
        channel_cnt = activation_per_batch.shape[-1]
        for channel in range(channel_cnt):
            channel_mean = activation_per_batch[:, :, channel].mean()
            means.append(channel_mean)

    return means


def calc_dense_activation_layer_means(activation_layer):
    means = []
    for activation_per_batch in activation_layer:
        channel_mean = activation_per_batch.mean()
        means.append(channel_mean)

    return means


def calc_conv_gradient_layer_means(gradient_layer):
    means = []
    channel_cnt, filter_cnt = gradient_layer.shape[2], gradient_layer.shape[3]
    
    for filter_idx in range(filter_cnt):
        mean = gradient_layer[:, :, :, filter_idx].mean()
        means.append(mean)

    return means


def calc_dense_gradient_layer_means(gradient_layer):
    return [gradient_layer.mean()]


def convert_conv_activation_layer_into_b64_images(activation_layer, resize_to=None, normalize_bound=None):
    b64_images = []
    ptp, _ = calc_max_ptp_for_conv_activation(activation_layer)

    for i, activation_per_batch in enumerate(activation_layer):
        sub_b64_images = convert_conv_activation_into_b64_images(activation_per_batch, resize_to=resize_to,
                                                                 normalize_bound=normalize_bound, ptp=ptp)
        b64_images.extend(sub_b64_images)

    return b64_images


def convert_dense_activation_layer_into_b64_images(activation_layer, resize_to=None, normalize_bound=None):
    b64_images = []
    ptp, _ = calc_max_ptp_for_dense_activation(activation_layer)

    for i, activation_per_batch in enumerate(activation_layer):
        sub_b64_images = convert_dense_activation_into_b64_images(activation_per_batch, resize_to=resize_to,
                                                                  normalize_bound=normalize_bound, ptp=ptp)
        b64_images.extend(sub_b64_images)

    return b64_images


def convert_conv_activation_into_b64_images(activation, resize_to=None, normalize_bound=None, ptp=None):
    b64_images = []
    channel_cnt = activation.shape[-1]
    for channel in range(channel_cnt):
        activation_channel = activation[:, :, channel]
        b64_image = convert_array_into_b64_image(activation_channel, resize_to=resize_to,
                                                 normalize_bound=normalize_bound, ptp=ptp)
        b64_images.append(b64_image)

    return b64_images


def convert_dense_activation_into_b64_images(activation, resize_to=None, normalize_bound=None, ptp=None):
    activation = np.expand_dims(activation, axis=-1)
    return [convert_array_into_b64_image(activation, resize_to=resize_to, normalize_bound=normalize_bound, ptp=ptp)]


def convert_array_into_b64_image(numpy_image, resize_to=None, normalize_bound=None, ptp=None):
    if normalize_bound and ptp:
        if ptp != 0:
            numpy_image = normalize_bound * (numpy_image - numpy_image.min()) / ptp
    numpy_image = numpy_image.astype(np.uint8)

    image = Image.fromarray(numpy_image)

    # TODO: Remove if unnecessary
    # if IMG_SCALE:
    #     width, height = image.size
    #     width = int(width * IMG_SCALE)
    #     height = int(height * IMG_SCALE)
    #     if width > 1 and height > 1:
    #         image = image.resize((width, height))
    if resize_to:
        width, height = image.size
        width_over_height = width / height
        if resize_to[0] is not None and resize_to[1] is None:
            resize_to = (resize_to[0], int(resize_to[0] * (1.0 / width_over_height)))
        elif resize_to[0] is None and resize_to[1] is not None:
            resize_to = (int(resize_to[1] * width_over_height), resize_to[1])
        image = image.resize(resize_to)

    image_buffer = BytesIO()
    image.save(image_buffer, format="PNG")
    b64_image = b64encode(image_buffer.getvalue()).decode('utf8')

    return b64_image
