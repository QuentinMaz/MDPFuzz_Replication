import numpy as np
import pandas as pd

from typing import Optional


class FuzzerLogger:
    '''
    A class for logging data values to a .txt file in a specific format.

    Args:
    - filepath (str): The path to the log file.

    Usage:
    - Initialize the logger with a file path.
    - Use the log method to add a new log entry.
    - Use load_logs_as_dataframe to load logs into a Pandas DataFrame.
    '''
    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        self.columns = ['input', 'oracle', 'reward', 'sensitivity', 'coverage']
        self.delimiter = '; '

    def log(self, input: Optional[np.ndarray] = None, oracle: Optional[bool] = None,
            reward: Optional[float] = None, sensitivity: Optional[float] = None,
            coverage: Optional[float] = None) -> None:
        '''
        Log values to the file.

        Args:
        - input (Optional[np.ndarray]): Input value as a NumPy array.
        - oracle (Optional[bool]): Oracle value as a boolean.
        - reward (Optional[float]): Reward value as a floating-point number.
        - sensitivity (Optional[float]): Sensitivity value as a floating-point number.
        - coverage (Optional[float]): Coverage value as a floating-point number.
        '''
        log_data = {
            #TODO: compared to the pool np.savetxt(.), np.array2string is less accurate
            # It would be problematic if the precision difference causes reproducibility issue...
            'input': np.array2string(input, separator=',') if input is not None else 'None',
            'oracle': str(oracle) if oracle is not None else 'None',
            'reward': str(reward) if reward is not None else 'None',
            'sensitivity': str(sensitivity) if sensitivity is not None else 'None',
            'coverage': str(coverage) if coverage is not None else 'None'
        }

        # ensures good ordering by using the columns (weakness found when Python version is 3.5)
        log_line = self.delimiter.join([log_data[k] for k in self.columns])

        with open(self.filepath, 'a') as file:
            file.write(log_line + '\n')


    def load_logs(self) -> pd.DataFrame:
        '''
        Load logs from the file and return as a Pandas DataFrame.

        Returns:
        - pd.DataFrame: DataFrame containing the logged data.
        '''
        data = []
        with open(self.filepath, 'r') as file:
            header_line = file.readline().strip()
            assert header_line.split(self.delimiter) == self.columns, header_line.split(self.delimiter)
            for line in file:
                values = [v.strip() for v in line.strip().split(self.delimiter)]
                input = np.array(eval('np.array(' + values[0] + ')')) if values[0] != 'None' else None
                oracle = values[1] == 'True' if values[1] != 'None' else None
                reward = float(values[2]) if values[2] != 'None' else None
                sensitivity = float(values[3]) if values[3] != 'None' else None
                coverage = float(values[4]) if values[4] != 'None' else None

                data.append([input, oracle, reward, sensitivity, coverage])

        return pd.DataFrame(data, columns=self.columns)


    def log_header_line(self):
        with open(self.filepath, 'w') as file:
            file.write(self.delimiter.join(self.columns) + '\n')


if __name__ == '__main__':
    import os
    log_file_example_path = 'log_example.txt'
    if os.path.isfile(log_file_example_path):
        logger = FuzzerLogger(log_file_example_path)
        df = logger.load_logs()
        print(df)