import torch

from core.trainer import build_trainer
from core.data import build_data_loaders
from models.model_factory import model_factory
from core.utils import FileManager, get_arguments, display_configs, get_flops_and_params

args = get_arguments()

# Define log file and print class
file_manager = FileManager(args)

# load dataset
print('Loading Dataset...')
train_loader, test_loaders = build_data_loaders(args)
data_loader = {'train': train_loader,
               'test': test_loaders}

# build network
num_classes = train_loader.dataset.num_classes

size = args.size
args.size = args.training_size
net = model_factory(args, num_classes).to(memory_format=torch.channels_last)

teachers = None
if args.teacher:
    assert len(args.teacher) == len(args.app)
    args_app = args.app
    args_basenet = args.basenet
    teachers = dict()
    for app, path in zip(args.app, args.teacher):
        args.app = [app]
        teachers[app] = model_factory(args, num_classes).to(memory_format=torch.channels_last)
        assert args.teacher, "Please set teacher checkpoint path using --teacher argument"
        teachers[app].load_state_dict(torch.load(path), strict=True)
        teachers[app].eval()
    args.app = args_app
args.size = size
net_to_eval = model_factory(args, num_classes).to(memory_format=torch.channels_last)
net_to_eval.load_state_dict(net.state_dict())
net_to_eval.flops, net_to_eval.params = get_flops_and_params(net_to_eval, args.size)

trainer = build_trainer(args, net, data_loader, teachers=teachers)

display_configs(args, net_to_eval, train_loader, test_loaders, file_manager.get_root_folder(),
                trainer.loss_function, trainer.logger)

trainer.train_test()
print('training is finished')
