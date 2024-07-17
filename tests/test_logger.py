import numpy as np
import pandas as pd

from mdpfuzz.logger import FuzzerLogger, Logger

LOG_FILE_EXAMPLE_PATH = "tests/log_file_example.txt"


def _check_df_example(df: pd.DataFrame):
    assert len(df) == 10
    assert df.columns.tolist() == [
        "input",
        "oracle",
        "reward",
        "episode_length",
        "sensitivity",
        "coverage",
        "test_exec_time",
        "coverage_time",
        "run_time",
    ]
    inputs = np.vstack(df["input"])
    assert inputs.shape == (10, 4)
    assert np.issubdtype(inputs.dtype, np.integer)


def test_fuzzer_logger_load():
    logger = FuzzerLogger(LOG_FILE_EXAMPLE_PATH)
    df = logger.load_logs()
    _check_df_example(df)


def test_logger_load():
    logger = Logger(LOG_FILE_EXAMPLE_PATH)
    df = logger.load_logs()
    _check_df_example(df)


def test_same_dataframe_load():
    fuzzer_logger = FuzzerLogger(LOG_FILE_EXAMPLE_PATH)
    logger = Logger(LOG_FILE_EXAMPLE_PATH)
    pd.testing.assert_frame_equal(fuzzer_logger.load_logs(), logger.load_logs())
