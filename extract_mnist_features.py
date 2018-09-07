import numpy as np
import os

from keras.applications import resnet50
from keras.datasets import mnist
from skimage.transform import resize

from add_average_pooling_layers import add_average_pooling_layers


def build_save_directories(root_directory, tensor_idxs):
    # Create each of the following subdirectories for each tensor_idx.
    directories = [str(idx) for idx in tensor_idxs] + ['labels']
    subdirectories = ['training', 'testing']

    directories_created = dict()

    for directory in directories:
        absolute_directory = os.path.join(root_directory, directory)
        os.mkdir(absolute_directory)
        directories_created[directory] = dict()

        for subdirectory in subdirectories:
            dataset_directory = os.path.join(absolute_directory,
                                             subdirectory)
            os.mkdir(dataset_directory)
            directories_created[directory][subdirectory] = dataset_directory

    return directories_created


def calc_n_batches(data, batch_size):
    assert type(batch_size) == int

    return int(np.ceil(data.shape[0] / batch_size))


def extract_labels(labels, batch_size, dataset, save_directories):
    batches = generate_batches(labels, batch_size)
    n_batches = calc_n_batches(labels, batch_size)

    batch_fname_template = 'batch_{}.npy'
    length = get_number_length(n_batches)

    for i, batch in enumerate(batches):
        fname = batch_fname_template.format(str(i).zfill(length))
        save_labels(batch, dataset, fname, save_directories)


def extract_features(data, model, tensor_idxs,
                     batch_sz, dataset, save_dirs):
    assert type(batch_sz) == int
    assert dataset == 'training' or dataset == 'testing'

    batches = generate_batches(data, batch_sz)
    n_batches = calc_n_batches(data, batch_sz)

    batch_fname_template = 'batch_{}.npy'
    length = get_number_length(n_batches)

    for i, batch in enumerate(batches):
        preprocessed_batch = mnist_to_resnet_input_shape(batch)
        features = model.predict(preprocessed_batch)
        fname = batch_fname_template.format(str(i).zfill(length))

        args = (features, tensor_idxs, fname, save_dirs, dataset)
        save_features(*args)


def generate_batches(data, batch_size):
    n_batches = calc_n_batches(data, batch_size)

    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size

        yield data[start_idx: end_idx]

        if i == n_batches - 1:
            break


def get_number_length(number):
    return len(str(number))


def load_mnist():
    training = dict()
    testing = dict()

    (training['data'], training['labels']), \
        (testing['data'], testing['labels']) = mnist.load_data()

    return training, testing


def mnist_to_resnet_input_shape(data,
                                mnist_shape=(28, 28),
                                resnet_shape=(224, 224, 3)):
    assert len(data.shape) == 3
    assert data.shape[1:] == mnist_shape

    if mnist_shape != (28, 28):
        raise NotImplementedError

    resized_data = []
    for datum in data:
        reshaped_datum = resize(datum, resnet_shape)
        resized_data.append(reshaped_datum)

    resized_data = np.asarray(resized_data)

    return resized_data


def save_features(features, tensor_idxs, fname, save_directories, dataset):
    assert len(features) == len(tensor_idxs)

    for layer_features, idx in zip(features, tensor_idxs):
        save_directory = save_directories[str(idx)][dataset]
        fpath = os.path.join(save_directory, fname)

        np.save(fpath, layer_features)


def save_labels(labels, dataset, fname, save_directories):
    save_directory = save_directories['labels'][dataset]
    fpath = os.path.join(save_directory, fname)

    np.save(fpath, labels)


def extract_mnist_features():
    np.random.seed(1234)

    tensor_idxs = [17, 49, 91, 153, 173]  # Layers to average pool.
    root_directory = '/home/codas/Documents/xai/mnist_features/resnet_features'
    save_directories = build_save_directories(root_directory, tensor_idxs)

    batch_size = 1000
    batch_fname_template = 'batch_{}.npy'

    # Load MNIST data.
    training, testing = load_mnist()

    # Load ResNet model and append the computational graph with layers.
    resnet = resnet50.ResNet50(weights='imagenet')
    multi_output_model = add_average_pooling_layers(resnet, tensor_idxs)

    # Save training features.
    args = [training['data'],
            multi_output_model,
            tensor_idxs,
            batch_size,
            'training',
            save_directories]
    extract_features(*args)

    # Save testing features.
    args[0] = testing['data']
    args[-2] = 'testing'
    extract_features(*args)

    # Save training labels.
    args = [training['labels'], batch_size, 'training', save_directories]
    extract_labels(*args)

    # Save testing labels.
    args = [testing['labels'], batch_size, 'testing', save_directories]
    extract_labels(*args)


def main():
    import time
    start_time = time.time()

    np.random.seed(1234)
    extract_mnist_features()

    print(time.time() - start_time)


if __name__ == '__main__':
    main()
