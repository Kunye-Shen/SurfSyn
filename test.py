import torch
import torch.nn as nn
import albumentations as albu
from torch.utils.data import DataLoader
from albumentations.pytorch.transforms import ToTensorV2

import argparse
from tqdm import tqdm

from utils import *
from model import SurfSyn
from data_loader import *


def main(args):
    # ------- model setting -------
    net = SurfSyn(args.version)
    net.load_state_dict(torch.load(args.model_dir))

    if args.parallel == True:
        net = nn.DataParallel(net, device_ids=args.cuda_device)
    
    if torch.cuda.is_available():
        net.cuda()

    test(net, args)

    return

def test(net, args):
    # ------- load testing data -------
    transform_test = albu.Compose([albu.Resize(args.img_size, args.img_size, always_apply=True),
            albu.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    # ------- start testing -------
    print("------- Start Testing -------")

    net.eval()
    for dataset in args.dataset_list:
        # load testing data
        print(f"Dataset: {dataset}")
        dataset_test = DefectDataset(args.dataset_path, transform_test, 'test', dataset)
        dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        for index, data in tqdm(enumerate(dataloader_test), total=dataloader_test.__len__()):
            image, name = data['img'].cuda().to(torch.float32), data['img_dir']

            pred1, *_ = net(image)

            for i in range(pred1.shape[0]):
                pred_i = pred1[i,0,:,:]
                pred_i = normPRED(pred_i)
                save_output(name[i], pred_i, os.path.join(args.pre_dir, dataset, ''))

            del image, pred1, pred_i

    return


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # testing
    p.add_argument("--CUDA", type=str, default='0,1')
    p.add_argument("--version", type=str, default='SurfSyn-T')
    p.add_argument("--parallel", type=bool, default=True)
    p.add_argument("--batch_size", type=int, default=12)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--cuda_device", type=list, default=[0,1])

    # root
    p.add_argument("--dataset_path", type=str, default="")
    p.add_argument("--model_dir", type=str, default="./model_save/")
    p.add_argument("--pre_dir", type=str, default="./predicts/")
    p.add_argument("--dataset_list", type=list, default=['ESDIs', 'CrackSeg9k', 'TUT', 'SSP2000'])
    args = p.parse_args()

    args.pre_dir = os.path.join(args.pre_dir, args.version, '')
    args.model_dir = os.path.join(args.model_dir, args.version, 'SurfSyn.pth')

    os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA

    main(args)