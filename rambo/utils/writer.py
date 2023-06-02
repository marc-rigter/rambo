import io
import numpy as np
import pdb
import wandb
import tensorboardX as tbx

class Writer():

    def __init__(self, log_dir, log_wandb, wandb_project="", wandb_group="", config=None):
        self.log_dir = log_dir
        self._writer = tbx.SummaryWriter(self.log_dir)
        self._data = {}
        self._data_3d = {}
        self._wandb_run = None
        if log_wandb:
            self._wandb_run = wandb.init(project=wandb_project, group=wandb_group, config=config)
        print('[ Writer ] Log dir: {}'.format(log_dir))

    def __getitem__(self, key):
        if key in self._data:
            return self._data[key]
        else:
            return self._data_3d[key]

    def __getattr__(self, attr):
        return getattr(self._writer, attr)

    def _add_label(self, data, label):
        if label not in data:
            data[label] = 0

    def add_scalar(self, label, val, epoch):
        self._add_label(self._data, label)
        if epoch > self._data[label]:
            self._data[label] = epoch
            self._writer.add_scalar(label, val, epoch)
            if self._wandb_run is not None:
                wandb.log({label: val}, step=epoch)

    def add_dict(self, dictionary, epoch):
        if self._wandb_run is not None:
            wandb.log(dictionary, step=epoch)
        for label, val in dictionary.items():
            self._writer.add_scalar(label, val, epoch)