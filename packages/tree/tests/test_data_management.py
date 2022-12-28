import numpy as np
from tree.processing.data_management import load_image, load_model


def test_load_image(file_name):
    _image = load_image(file_name=file_name)

    assert type(_image) == np.ndarray

def test_load_model(model_file_name):
    clf, columns, scaler = load_model(file_name=model_file_name)

    assert type(columns) == list