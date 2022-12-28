import numpy as np
import pandas as pd

from tree.processing.data_management import load_pipeline
from tree.config import config, logging_config
from tree.processing.validation import validate_inputs
from tree import __version__ as _version

import logging
import typing as t


_logger = logging.getLogger(__name__)
file_handler = logging.FileHandler(config.LOGS_DIR / f"{__name__}.txt")
formatter = logging_config.FORMATTER
file_handler.setFormatter(formatter)
_logger.addHandler(file_handler)

pipeline_file_name = f"{config.PIPELINE_SAVE_FILE}{_version}.pkl"
_pipe = load_pipeline(file_name=pipeline_file_name)


def make_prediction(*, input_data: t.Union[pd.DataFrame, dict],
                    ) -> dict:
    """Make a prediction using a saved model pipeline.

    Args:
        input_data: Array of model prediction inputs.

    Returns:
        Predictions for each input row, as well as the model version.
    """

    data = pd.DataFrame(input_data)
    validated_data = validate_inputs(input_data=data)

    prediction = _pipe.predict(validated_data[config.FEATURES])

    output = np.exp(prediction)

    results = {"predictions": output, "version": _version}

    _logger.info(
        f"Making predictions with model version: {_version} "
        f"Inputs: {validated_data} "
        f"Predictions: {results}"
    )

    return results
