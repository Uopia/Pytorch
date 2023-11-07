from argparse import ArgumentParser
import os
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import matplotlib
import matplotlib.pyplot as plt
from Model_evaluation_cl import ModelEvaluation
import numpy as np
import shutil
import torch

matplotlib.use('TkAgg')
config = r'D:/Swin/Swin-Transformer-Object-Detection/configs/swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.py'
checkpoint = r'D:\Swin\Swin-Transformer-Object-Detection\work_dirs\mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco/epoch_22.pth'
img_folder = r'D:\Swin\Data\coco\val1'
device = 'cuda:0'


def main():
    # parser = ArgumentParser()
    # parser.add_argument('img',help='Image file')
    # parser.add_argument('config', help='Config file')
    # parser.add_argument('checkpoint',help='Checkpoint file')
    # parser.add_argument(
    #     '--device', default='cuda:0', help='Device used for inference')
    # parser.add_argument(
    #     '--score-thr', type=float, default=0.3, help='bbox score threshold')
    # args = parser.parse_args()
    #
    # # build the model from a config file and a checkpoint file
    # model = init_detector(args.config, args.checkpoint, device=args.device)
    # # test a single image
    # result = inference_detector(model, args.img)
    # # show the results
    # show_result_pyplot(model, args.img, result, score_thr=args.score_thr)

    #

    score_thr = 0.2

    # Get all image paths in the folder
    img_local = []
    results = []
    model = init_detector(config, checkpoint, device=device)
    folder = img_folder.split("\\")[-1]

    path = os.path.join(img_folder, f"{folder}{4}")
    if os.path.exists(path):
        for _ in range(100):
            torch.cuda.empty_cache()
        print("cuda memory = {:.3f} GBs".format(torch.cuda.max_memory_allocated() / 1024 ** 3))
        results.extend(inference_detector(model, path))
        print("cuda memory = {:.3f} GBs".format(torch.cuda.max_memory_allocated() / 1024 ** 3))
        img_local.extend([os.path.join(path, f) for f in os.listdir(path) if
                         f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))])

    m = ModelEvaluation(r'D:\Swin\Data\coco', "COCO", results, img_local)
    m.calculate()
    m.print_res(['AJ', 'BX', 'CJ', 'CK', 'CQ', 'FS', 'FZ', 'JG_Down', 'QF', 'SG',
                'SL', 'TL', 'ZW'])
    m.output_log(['AJ', 'BX', 'CJ', 'CK', 'CQ', 'FS', 'FZ', 'JG_Down', 'QF', 'SG',
                 'SL', 'TL', 'ZW'])
    # show_result_pyplot(model, img_path, img_result, score_thr=score_thr)


if __name__ == '__main__':
    main()


