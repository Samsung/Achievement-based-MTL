import os
import pandas as pd
import torch.distributed as dist


class Logger:
    def __init__(self, log_file_path=None):
        self.log_file_path = log_file_path
        self.is_print = (not dist.is_initialized() or not dist.get_rank())

    def __call__(self, log, is_print=True):
        if self.is_print:
            print(log)
        log_file = open(os.path.join(self.log_file_path, 'log.txt'), 'a')
        log_file.write(log + '\n')
        log_file.close()

    def write_summary(self, index: int, losses: dict, train=True, ema=False):
        if not dist.is_initialized() or not dist.get_rank():
            df_name = 'train.csv' if train else 'valid_ema.csv' if ema else 'valid.csv'
            data_path = os.path.join(self.log_file_path, df_name)
            if not os.path.exists(data_path):
                df = pd.DataFrame(losses, index=[index])
            else:
                df = pd.read_csv(data_path, index_col=0)
                df = pd.concat([df, pd.DataFrame.from_dict({index: losses}, orient='index')])
            df.to_csv(data_path)

    def to_csv(self, index: int, year, ap: dict):
        df_path = os.path.join(self.log_file_path, '%s.csv' % year)
        if not os.path.exists(df_path):
            df = pd.DataFrame(ap, index=[index])
        else:
            df = pd.read_csv(df_path, index_col=0)
            df.loc[index] = ap
        df.to_csv(df_path)
        log = df.iloc[-5:].to_string()
        self.__call__(log)
