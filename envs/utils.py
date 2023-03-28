import numpy as np
import torch


def load_data(
    training_percentage=0.8,
    shuffle_seed=0,
    device="cpu",
):
    """Returns two tensors with training and testing data"""
    # Loading the data.
    # This data is structured [b, c, i, j], where c corresponds to the class.
    data = np.load("assets/mario_slided.npz")['arr_0']
    data = data.transpose([0, 3, 1, 2])
    np.random.seed(shuffle_seed)
    np.random.shuffle(data)

    # Separating into training and test.
    n_data, _, _, _ = data.shape
    training_index = int(n_data * training_percentage)
    training_data = data[:training_index, :, :, :]
    testing_data = data[training_index:, :, :, :]
    training_tensors = torch.from_numpy(training_data).type(torch.float)
    test_tensors = torch.from_numpy(testing_data).type(torch.float)

    return training_tensors.to(device), test_tensors.to(device)


def load_data_int(
    training_percentage=1.,
    shuffle_seed=0,
    device="cpu",
):
    """Returns two tensors with training and testing data"""
    # Loading the data.
    # This data is structured [b, c, i, j], where c corresponds to the class.
    data_onehot = np.load("assets/mario_slided.npz")['arr_0']
    data = np.argmax(data_onehot, axis=-1)
    data = np.expand_dims(data, axis=-1)
    data = data.transpose([0, 3, 1, 2])
    np.random.seed(shuffle_seed)
    np.random.shuffle(data)

    # Separating into training and test.
    n_data, _, _, _ = data.shape
    training_index = int(n_data * training_percentage)
    training_data = data[:training_index, :, :, :]
    testing_data = data[training_index:, :, :, :]
    training_tensors = torch.from_numpy(training_data).type(torch.float)
    test_tensors = torch.from_numpy(testing_data).type(torch.float)

    return training_tensors.to(device), test_tensors.to(device)


def load_data_float(
    training_percentage=.9,
    shuffle_seed=0,
    device="cpu",
):
    """Returns two tensors with training and testing data"""
    # Loading the data.
    # This data is structured [b, c, i, j], where c corresponds to the class.
    data = np.load("assets/mario_slided_float.npz")['arr_0']
    data = np.expand_dims(data, axis=-1)
    data = data.transpose([0, 3, 1, 2])
    np.random.seed(shuffle_seed)
    np.random.shuffle(data)

    # Separating into training and test.
    n_data, _, _, _ = data.shape
    training_index = int(n_data * training_percentage)
    training_data = data[:training_index, :, :, :]
    testing_data = data[training_index:, :, :, :]
    training_tensors = torch.from_numpy(training_data).type(torch.float)/6
    test_tensors = torch.from_numpy(testing_data).type(torch.float)/6

    return training_tensors.to(device), test_tensors.to(device)
