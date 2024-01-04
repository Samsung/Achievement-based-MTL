import os
import torch
import torch.distributed as dist
VALID_FILE_SIZE = 20480000


class FileManager:
    def __init__(self, args):
        self.is_master = not dist.is_initialized() or not dist.get_rank()
        self.root_folder, prefix = self.build_root_folder(args)
        self.prefix = os.path.join(self.root_folder, prefix)
        self.recent_ckpt = None
        self.remove_recent_ckpt = args.remove_ckpt

    def get_root_folder(self):
        return self.root_folder

    def get_test_folder(self):
        test_dir = os.path.join(self.root_folder, 'ss_predict')
        if self.is_master and not os.path.exists(test_dir):
            self._mkdir(test_dir)
        return test_dir

    def build_root_folder(self, args):
        file_prefix = args.arch + '_' + args.basenet + '_' + '+'.join(args.test_dataset) + '_' + \
                      str(args.size[0]) + 'x' + str(args.size[1])
        root_folder = os.path.join(args.root_folder, file_prefix, args.subfolder)
        if self.is_master and not os.path.exists(root_folder):
            self._mkdir(root_folder)
        return root_folder, file_prefix

    def load_state_dict(self, resume=False, resume_epoch=0, ckpt=None):
        if ckpt:
            state_dict = torch.load(ckpt)
            return state_dict, 0

        if resume:
            if resume_epoch:
                self.recent_ckpt = os.path.join(self.prefix + '_iterations_%d.pth' % resume_epoch)
            else:
                self.recent_ckpt, resume_epoch = self._get_latest_checkpoint()
        return torch.load(self.recent_ckpt) if self.recent_ckpt else self.recent_ckpt, resume_epoch

    def _get_latest_checkpoint(self):
        files = []
        torch.cuda.synchronize()
        for file in os.listdir(os.path.join(self.root_folder)):
            if file.endswith('.pth'):
                file = os.path.join(self.root_folder, file)
                if self.prefix not in file:
                    continue
                if os.path.getsize(file) >= VALID_FILE_SIZE:  # check the file is valid
                    files.append(file)

        if files:
            latest_path = max(files, key=os.path.getctime)
            try:
                resume_epoch = int(os.path.splitext(latest_path)[0].split('_')[-1])
            except ValueError:
                latest_path = None
                resume_epoch = 0
        else:
            latest_path, resume_epoch = None, 0
        return latest_path, resume_epoch

    def save(self, state_dict, epoch=None, is_best=False):
        assert (epoch is not None or is_best), "FileManager.save should set either epoch or is_best"
        if self.is_master:
            if epoch:
                torch.save(state_dict, self.prefix + '_epoch_%d.pth' % epoch)
                self._exchange_recent_ckpt(self.prefix + '_epoch_%d.pth' % epoch)

            if is_best:
                torch.save(state_dict, self.prefix + '_best.pth')

    def _set_recent_ckpt(self, recent_ckpt):
        self.recent_ckpt = recent_ckpt

    def _exchange_recent_ckpt(self, recent_ckpt):
        if self.recent_ckpt is not None and self.remove_recent_ckpt:
            os.remove(self.recent_ckpt)
        self.recent_ckpt = recent_ckpt

    @staticmethod
    def _mkdir(path, mode=0o775, gid=1625234492, exist_ok=False):
        old_mask = os.umask(0)
        os.makedirs(path, mode=mode, exist_ok=exist_ok)
        os.chown(path, -1, gid)
        os.umask(old_mask)
