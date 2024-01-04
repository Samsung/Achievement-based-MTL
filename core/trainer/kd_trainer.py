from typing import Tuple
import torch
from .trainer import Trainer


class KDTrainer(Trainer):
    """
    Trainer for Knowledge Distillation for Multi-Task Learning presented in ECCV Workshop 2020
    Paper: https://arxiv.org/pdf/2007.06889.pdf
    The official code: https://github.com/VICO-UoE/KD4MTL
    """
    def __init__(self, args, net, data_loader, teachers=None):
        assert teachers is not None, "KDTrainer requires teachers"
        super(KDTrainer, self).__init__(args, net, data_loader, teachers)
        self.projections = {app: torch.nn.Conv2d(192, 192, 1) for app in args.app}
        [proj.cuda() for app, proj in self.projections.items()]

        params = [param for proj in self.projections.values() for param in proj.parameters()]
        self.optimizer.add_param_group({'params': params})
        self.kd_loss = dict()

    def _forward(self, sample) -> Tuple[dict, dict]:
        preds = self.net(sample['images'])
        with torch.no_grad():
            features = {app: teacher(sample['images'])['features'].detach() for app, teacher in self.teachers.items()}
        shared_features = {app: proj(preds['features']) for app, proj in self.projections.items()}

        eps = 1e-7
        features = {app: feature / (feature.pow(2).sum(1, keepdims=True) + eps).sqrt()
                    for app, feature in features.items()}
        shared_features = {app: feature / (feature.pow(2).sum(1, keepdims=True) + eps).sqrt()
                           for app, feature in shared_features.items()}

        self.kd_loss = {app: (features[app] - shared_features[app]).pow(2).sum(1).mean() for app, _ in features.items()}
        return preds, dict()

    def _backward(self, loss, scaler, epoch=None):
        self.optimizer.zero_grad(set_to_none=True)

        # add kd loss to cur_loss
        for app, cur_loss in self.kd_loss.items():
            self.loss_function.cur_loss[app] += self.kd_loss[app]
        loss = self.loss_function.get_cur_loss()

        scaler.scale(loss).backward()
        scaler.step(self.optimizer)
        scaler.update()
        self.scheduler.step(epoch)
