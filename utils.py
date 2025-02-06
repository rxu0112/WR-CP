import torch
import numpy as np


class Logger:
    def __init__(self):
        self.data = dict()

    def log(self, k, v):
        if not k in self.data:
            self.data[k] = []
        v = self._to_np(v)
        self.data[k].append(v)

    def __getitem__(self, k):
        return self.data.get(k, [])

    def _to_np(self, v):
        if isinstance(v, torch.Tensor):
            with torch.no_grad():
                return v.cpu().numpy()
        if isinstance(v, list):
            return [self._to_np(v_) for v_ in v]
        return v

def print_stats(step, log):
    if 'wr_penalty' in log.data:
        pretty_print(
            np.int32(step),
            np.mean(log['train_l1'][-50:]),
            np.mean(log['wr_penalty'][-50:]),
        )
    else:
        pretty_print(
            np.int32(step),
            np.mean(log['train_l1'][-50:]),
        )

def pretty_print(*values):
    col_width = 13
    def format_val(v):
        if not isinstance(v, str):
            v = np.array2string(v, precision=5, floatmode='fixed')
        return v.ljust(col_width)
    str_values = [format_val(v) for v in values]
    print("   ".join(str_values))

def print_env_info(train_envs, test_envs, cal_envs):
    num_feat = train_envs[0]['images'].shape[1]
    print('training on', len(train_envs), 'environments (using', num_feat, 'features):')
    for e in train_envs:
        print('   ', e['info'], len(e['labels']))
    print('calibrating on:')
    for e in cal_envs:
        print('   ', e['info'], len(e['labels']))
    print('testing on:')
    for e in test_envs:
        print('   ', e['info'], len(e['labels']))
