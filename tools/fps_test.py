# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os
import torch
import sys
import time
import argparse
import os
import torch
import sys
from siamese.core.config import cfg
from siamese.utils.model_load import load_pretrain
from siamese.models.model_builder import ModelBuilder

sys.path.append(os.getcwd())
sys.path.append(os.path.abspath('.'))



parser = argparse.ArgumentParser(description='siamese')
parser.add_argument('--config', type=str, default='./models/config/config.yaml', help='config file')
parser.add_argument('--snapshot', default='D:/python/siamese/tools/snapshot/checkpoint_e300.pth', type=str,
                    help='snapshot models to eval')
args = parser.parse_args()

torch.set_num_threads(1)


def main():
    cfg.merge_from_file(args.config)

    model = ModelBuilder()

    model = ModelBuilder()

    model = load_pretrain(model, args.snapshot)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval().to(device)

    x, z = torch.randn(1, 3, 255, 255, device=device), torch.randn(1, 3, 127, 127, device=device)

    T_w, T_t = 100, 1000

    with torch.no_grad():
        model.template(z)
        for i in range(T_w):
            model.track(x)
        t_s = time.time()

        for i in range(T_t):
            model.track(x)
        t_e = time.time()
        print('speed: %.2f FPS' % (T_t / (t_e - t_s)))


if __name__ == '__main__':
    main() 