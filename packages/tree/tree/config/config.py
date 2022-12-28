import pathlib
import pandas as pd
import tree

pd.options.display.max_rows = 10
pd.options.display.max_columns = 10


PACKAGE_ROOT = pathlib.Path(tree.__file__).resolve().parent
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"
EXTERNAL_MODEL_DIR = PACKAGE_ROOT / "external_models"

DATASET_DIR = PACKAGE_ROOT / "datasets"
TESTING_DATA_FILE = "test.json"
TRAINING_DATA_FILE = "train.csv"

# variables
FEATURES = [
	'spl',
	'lucrative_degree',
	'lucrative_ratio',
	'in_degree',
	'out_degree',
	'avg_time_spent',
	'exit_rate',
	'number_visits',
	'est_t1',
	'est_t2',
	'est_t3',
	'number_exits',
	'prob_sp',
	'prob_ksp',
	'prob_next_page_abandon',
	'prob_curr_and_next_abandon',
	'avg_next_spl',
	'prob_curr_path',
	'avg_er',
	'avg_time_curr_path',
	'current_time_spent'
 ]

DROP_FEATURES = "YrSold"

PIPELINE_NAME = "recommender"
PIPELINE_SAVE_FILE = f"{PIPELINE_NAME}_output_v"

# used for differential testing
ACCEPTABLE_MODEL_DIFFERENCE = 0.05

LOGS_DIR = PACKAGE_ROOT / "logs"