import torch

from test_net import test_net
from core.data import build_data_loaders
from core.utils import FileManager, get_arguments
from models.model_factory import model_factory


args = get_arguments()
file_manager = FileManager(args)

print('Loading Dataset...')
_, test_loaders = build_data_loaders(args)
num_classes = test_loaders[0].dataset.num_classes
net = model_factory(args, num_classes)

state_dict = torch.load(args.ckpt)
net.load_state_dict(state_dict['model'] if 'model' in state_dict else state_dict)
if torch.cuda.is_available():
    net.cuda()

results = test_net(file_manager, net, test_loaders)
