import torch
from copy import deepcopy
from collections import OrderedDict


class ModelEMA(torch.nn.Module):
    def __init__(self, model, momentum):
        super(ModelEMA, self).__init__()
        self.model = deepcopy(model)
        self._momentum = momentum
        self.update(model)

    def update(self, model):
        state_dict_old = self.model.state_dict()
        state_dict_new = model.state_dict()
        state_dict_updated = OrderedDict()

        with torch.no_grad():
            for (key, value_old), (_, value_new) in zip(state_dict_old.items(), state_dict_new.items()):
                state_dict_updated[key] = value_old * self._momentum + value_new * (1 - self._momentum)
            self.model.load_state_dict(state_dict_updated)
