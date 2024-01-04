import torch
from copy import deepcopy
from torch.cuda.amp import autocast, GradScaler

from test_net import test_net
from ..losses import loss_factory
from ..metrics import merge_metrics
from ..optimizer import build_optimizer
from ..lr_scheduler import build_scheduler
from ..data.augmentations import TensorDistortion
from ..utils import Timer, FileManager, Logger, ModelEMA


class Trainer:
    def __init__(self, args, net, data_loader, teachers: dict = None):
        self._set_training_configs(args)
        self.file_manager = FileManager(args)
        self.logger = Logger(self.file_manager.get_root_folder())
        self.teachers = dict() if teachers is None else teachers

        self._network_setup(net, args.model_ema, args.model_ema_momentum, args.sync_bn)
        self.loss_function = loss_factory(args, self.net_without_ddp)
        self.optimizer = self._build_optimizer(args)
        self.scheduler = self._build_scheduler(args)
        self._load_state_dict(args.resume, args.resume_epoch, args.ckpt)

        self.data_loader = data_loader
        self.best_metrics = {'total': 0}
        self._augmentation = TensorDistortion()
        self.display_freq = min(args.display_freq, len(self.data_loader['train']))

    def train_test(self):
        self.net.train()
        epoch = self.start_epoch
        while epoch < self.max_epoch:
            self.train(epoch)
            epoch += 1
            if epoch % self.save_freq == 0:
                self.file_manager.save(self._get_state_dict(epoch), epoch)

            if epoch % self.test_freq == 0 and epoch >= self.warmup_epoch:
                self.eval(epoch)

    def train(self, epoch):
        epoch_size = len(self.data_loader['train'])
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        use_amp = self.use_amp & torch.cuda.has_half
        scaler = GradScaler(enabled=use_amp)
        self._set_epoch(epoch)

        timer = Timer()
        # load train data
        timer.tic('batch_time')
        timer.tic('data')
        for batch_iter, sample in enumerate(self.data_loader['train'], 1):
            with autocast(enabled=use_amp):
                sample = self._preprocessing(sample, device)
                timer.toc('data')

                # forward
                timer.tic('forward')
                preds, pseudo = self._forward(sample)
                timer.toc('forward')

                # loss computation
                timer.tic('loss')
                loss = self.loss_function(preds, gt=sample, pseudo=pseudo)
                timer.toc('loss')

            # backpropagation
            timer.tic('backward')
            self._backward(loss, scaler, epoch + batch_iter / epoch_size)
            if hasattr(self, 'model_ema'):
                self.model_ema.update(self.net_without_ddp)
            timer.toc('backward')
            timer.toc('batch_time')

            if batch_iter % self.display_freq == 0:
                self._log(epoch, batch_iter, timer)
            timer.tic('batch_time')
            timer.tic('data')
        if batch_iter % self.display_freq != 0:
            self._log(epoch, batch_iter, timer)

    def _preprocessing(self, sample, device):
        sample = {key: [anno.to(device) for anno in value] if type(value) is list else value.to(device)
                  for key, value in sample.items()}
        sample['images'] = sample['images'].to(memory_format=torch.channels_last)
        sample = self._augmentation(sample)
        return sample

    def _forward(self, sample):
        preds = self.net(sample['images'])
        with torch.no_grad():
            pseudo = {app: teacher(sample['images'])[app]
                      for app, teacher in self.teachers.items()}
        return preds, pseudo

    def _backward(self, loss, scaler, epoch=None):
        self.optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(self.optimizer)
        scaler.update()
        self.scheduler.step(epoch)

    def _log(self, epoch, batch_iter, timer):
        epoch_size = len(self.data_loader['train'])
        lr = self.optimizer.param_groups[0]['lr']

        log = 'Epoch: %3d || epoch_iter: %3d/%3d || %s || ' \
              % (epoch, batch_iter, epoch_size, str(self.loss_function))
        log += str(timer) + ' || LR: %.6f' % lr
        self.logger(log)

        log_items = self.loss_function.items()
        log_items.update(self.loss_function.get_weights())
        if batch_iter == epoch_size:
            log_items['epoch_time'] = timer.get_epoch_time()
        self.logger.write_summary(epoch + batch_iter / epoch_size, log_items)
        self.loss_function.clear()
        timer.clear()

    def eval(self, epoch):
        results = self._evaluate(epoch, self.net_without_ddp.state_dict())
        if hasattr(self, 'model_ema'):
            results = self._evaluate(epoch, self.model_ema.model.state_dict(), ema=True)
        self.update_scheduler(epoch, metrics=results['total'])

    def _evaluate(self, epoch, state_dict, ema=False):
        net_to_eval = deepcopy(self.net_without_ddp)
        net_to_eval.load_state_dict(state_dict)

        results = test_net(self.file_manager, net_to_eval, self.data_loader['test'], self.loss_function,
                           self.logger, epoch, ema=ema)
        results['total'] = merge_metrics(results)
        self.logger('total metric is %3.4f' % results['total'])
        self._compare_to_best_metric(state_dict, results)
        return results

    def _compare_to_best_metric(self, state_dict, metrics):
        if metrics['total'] > self.best_metrics['total']:
            for key, val in metrics.items():
                self.logger('best %s is updated to %3.4f\n' % (key, val))
                self.best_metrics[key] = val
            self.file_manager.save(state_dict, is_best=True)

    def _set_training_configs(self, args):
        self.local_rank = args.local_rank
        self.max_epoch = args.max_epoch
        self.warmup_epoch = args.warmup_epoch
        self.save_freq = args.save_freq
        self.test_freq = args.test_freq
        self.use_amp = args.amp

    def _set_epoch(self, epoch):
        if hasattr(self.data_loader['train'].sampler, 'set_epoch'):
            self.data_loader['train'].sampler.set_epoch(epoch)

    def _network_setup(self, net, model_ema=True, ema_momentum=0.9999, sync_bn=False):
        # Applying SyncBN
        if sync_bn and torch.distributed.is_initialized():
            if torch.cuda.device_count() > 1:  # No need to sync_bn for a single GPU
                print('convert bn to sync_bn')
                net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)

        # CUDA setting for GPU processing
        self.net = net
        self.net_to_eval = self.net  # ToDo
        self.net_without_ddp = self.net
        self._cuda_setup()

        if model_ema:
            self.model_ema = ModelEMA(self.net_without_ddp, ema_momentum)

    def _cuda_setup(self):
        if torch.cuda.device_count() >= 1:
            self.net.cuda()
            self.net_to_eval.cuda()
            if hasattr(self, 'model_ema'):
                self.model_ema.cuda()
            if self.teachers:
                [teacher.cuda() for _, teacher in self.teachers.items()]
            if torch.distributed.is_initialized():
                self.net = torch.nn.parallel.DistributedDataParallel(
                    self.net, device_ids=[self.local_rank],
                    static_graph=True  # optional for accelerating
                )
            else:
                torch.nn.DataParallel(self.net, device_ids=list(range(torch.cuda.device_count())))

    def _build_optimizer(self, args):
        net = self.net.module if hasattr(self.net, 'module') else self.net
        return build_optimizer(args, net)

    def _build_scheduler(self, args):
        return build_scheduler(args, self.optimizer)

    def _load_state_dict(self, resume, resume_epoch, ckpt):
        state_dict, self.start_epoch = self.file_manager.load_state_dict(resume, resume_epoch, ckpt)
        if state_dict is not None:
            if 'model' in state_dict:
                net_without_ddp = self.net.module if hasattr(self.net, 'module') else self.net
                net_without_ddp.load_state_dict(state_dict['model'])
            if 'optimizer' in state_dict:
                self.optimizer.load_state_dict(state_dict['optimizer'])
            if 'scheduler' in state_dict:
                self.scheduler.load_state_dict(state_dict['scheduler'])
            if 'model_ema' in state_dict:
                self.model_ema.load_state_dict(state_dict['model_ema'])

    def _get_state_dict(self, epoch):
        state_dict = {
            'epoch': epoch,
            'best_results': self.best_metrics,
            'model': self.net_without_ddp.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }
        if hasattr(self, 'model_ema'):
            state_dict['model_ema'] = self.model_ema.state_dict()
        return state_dict

    def update_scheduler(self, epoch, metrics=None):
        self.scheduler.step(epoch=epoch, metrics=metrics)
