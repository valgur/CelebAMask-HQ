dependencies = ['torch', 'numpy']

import torch.hub

import argparse
import os
import sys


class _add_to_pythonpath:
    def __init__(self, path):
        script_dir = os.path.dirname(os.path.realpath(__file__))
        self.path = os.path.join(script_dir, path)

    def __enter__(self):
        sys.path.append(self.path)

    def __exit__(self, exit_type, exit_value, exit_traceback):
        sys.path.remove(self.path)

    def __call__(self, func):
        def decorator(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return decorator


_label2face_512p_urls = {
    'G': 'https://github.com/valgur/CelebAMask-HQ/releases/download/maskgan-weights/label2face_512p_G-8e6483cf.pt',
    'B': 'https://github.com/valgur/CelebAMask-HQ/releases/download/maskgan-weights/label2face_512p_B-21490111.pt',
    'D': 'https://github.com/valgur/CelebAMask-HQ/releases/download/maskgan-weights/label2face_512p_D-72227c36.pt',
}


def Pix2PixHD(pretrained=False, progress=True, map_location=None, **kwargs):
    with _add_to_pythonpath("MaskGAN_demo"):
        from models.pix2pixHD_model import Pix2PixHDModel
        from options.train_options import TrainOptions

    parser = TrainOptions()
    parser.initialize()
    opt_dict = vars(parser.parser.parse_args([]))
    opt_dict['isTrain'] = False
    opt_dict['gpu_ids'] = list(range(torch.cuda.device_count()))
    opt_dict.update(kwargs)
    opt = argparse.Namespace(**opt_dict)

    model = Pix2PixHDModel()
    load_network_old = model.load_network
    if pretrained:
        def load_network(network, network_label, *args, **kwargs):
            network.load_state_dict(torch.hub.load_state_dict_from_url(
                _label2face_512p_urls[network_label], map_location=map_location, progress=progress, check_hash=True))

        model.load_network = load_network
    else:
        model.load_network = lambda *args, **kwargs: None
    model.initialize(opt)
    model.load_network = load_network_old

    if opt.isTrain and len(opt.gpu_ids):
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

    return model
