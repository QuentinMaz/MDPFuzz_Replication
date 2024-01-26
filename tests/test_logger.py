import numpy as np

from src.logger import FuzzerLogger


def test_load():
    log_file_example_path = 'tests/log_file_example.txt'
    logger = FuzzerLogger(log_file_example_path)
    df = logger.load_logs()
    assert len(df) == 301
    assert df.columns.tolist() == ['input', 'oracle', 'reward', 'sensitivity', 'coverage']
    inputs = np.vstack(df['input'])
    assert inputs.shape == (301, 4)
    assert np.issubdtype(inputs.dtype, np.integer)