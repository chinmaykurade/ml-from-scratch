import pytest
from tree.config import config

@pytest.fixture
def file_name():
    return config.TESTING_FILE

@pytest.fixture
def model_file_name():
    return config.MODEL_FILE