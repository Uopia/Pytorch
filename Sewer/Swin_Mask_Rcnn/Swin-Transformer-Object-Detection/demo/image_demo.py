from argparse import ArgumentParser
import os
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

def main():
    # parser = ArgumentParser()
    # parser.add_argument('img', default='demo/1.png',help='Image file')
    # parser.add_argument('config', default='configs/swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.py', help='Config file')
    # parser.add_argument('checkpoint', default='latest.pth',help='Checkpoint file')
    # parser.add_argument(
    #     '--device', default='cuda:0', help='Device used for inference')
    # parser.add_argument(
    #     '--score-thr', type=float, default=0.3, help='bbox score threshold')
    # args = parser.parse_args()

    # # build the model from a config file and a checkpoint file
    # model = init_detector(args.config, args.checkpoint, device=args.device)
    # # test a single image
    # result = inference_detector(model, args.img)
    # # show the results
    # show_result_pyplot(model, args.img, result, score_thr=args.score_thr)


    config = r'D:/Desktop/Swin/Swin-Transformer-Object-Detection/configs/swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.py'
    checkpoint = r'D:/Desktop/Swin/Swin-Transformer-Object-Detection/latest.pth'
    img_folder = r'D:/Desktop/Swin/Swin-Transformer-Object-Detection/val'
    device = 'cuda:0'
    score_thr = 0.2

    # Get all image paths in the folder
    img_paths = [os.path.join(img_folder, f) for f in os.listdir(img_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]

    model = init_detector(config, checkpoint, device=device)
    results = inference_detector(model, img_folder)

    for img_path, img_result in zip(img_paths, results):
        img_result = img_result[0]
        show_result_pyplot(model, img_path, img_result, score_thr=score_thr)

if __name__ == '__main__':
    main()
