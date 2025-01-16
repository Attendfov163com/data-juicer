import copy
import csv
import sys
sys.path.append('/home/daoyuan_mm/data_juicer/data_juicer/my_pretrained_method/YOLOv8_human/utils')
sys.path.append('/home/daoyuan_mm/data_juicer/data_juicer/my_pretrained_method/YOLOv8_human')
import warnings
from argparse import ArgumentParser

import numpy
import torch
import tqdm
import yaml
from torch.utils import data
from nets import nn
from util import non_max_suppression

warnings.filterwarnings("ignore")


@torch.no_grad()
def demo(img_array, model):
    import cv2

    frame = img_array
    image = frame.copy()
    shape = image.shape[:2]

    r = 640 / max(shape[0], shape[1])
    if r != 1:
        resample = cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA
        image = cv2.resize(image, dsize=(int(shape[1] * r), int(shape[0] * r)), interpolation=resample)
    height, width = image.shape[:2]

    # Scale ratio (new / old)
    r = min(1.0, 640 / height, 640 / width)

    # Compute padding
    pad = int(round(width * r)), int(round(height * r))
    w = numpy.mod((640 - pad[0]), 32) / 2
    h = numpy.mod((640 - pad[1]), 32) / 2

    if (width, height) != pad:  # resize
        image = cv2.resize(image, pad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(h - 0.1)), int(round(h + 0.1))
    left, right = int(round(w - 0.1)), int(round(w + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)  # add border

    # Convert HWC to CHW, BGR to RGB
    x = image.transpose((2, 0, 1))[::-1]
    x = numpy.ascontiguousarray(x)
    x = torch.from_numpy(x)
    x = x.unsqueeze(dim=0)
    x = x.to(next(model.parameters()).device)
    x = x.half()
    x = x / 255
    # Inference
    outputs = model(x)
    # NMS
    outputs = non_max_suppression(outputs, 0.25, 0.7)
    final_output_box_list = []
    for output in outputs:
        output[:, [0, 2]] -= w  # x padding
        output[:, [1, 3]] -= h  # y padding
        output[:, :4] /= min(height / shape[0], width / shape[1])

        output[:, 0].clamp_(0, shape[1])  # x1
        output[:, 1].clamp_(0, shape[0])  # y1
        output[:, 2].clamp_(0, shape[1])  # x2
        output[:, 3].clamp_(0, shape[0])  # y2

        for box in output:
            box = box.cpu().numpy()
            x1, y1, x2, y2, score, index = box
            final_output_box_list.append((x1, y1, x2, y2))
            # cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    del x
    return final_output_box_list



def profile(args, params):
    model = nn.yolo_v8_n(len(params['names']))
    shape = (1, 3, args.input_size, args.input_size)

    model.eval()
    model(torch.zeros(shape))
    params = sum(p.numel() for p in model.parameters())
    if args.local_rank == 0:
        print(f'Number of parameters: {int(params)}')


def human_detect(img_array):
    parser = ArgumentParser()
    parser.add_argument('--input-size', default=640, type=int)
    parser.add_argument('--local_rank', default=0, type=int)

    args = parser.parse_args()

    args.local_rank = int(os.getenv('LOCAL_RANK', 0))
    args.world_size = int(os.getenv('WORLD_SIZE', 1))
    args.distributed = int(os.getenv('WORLD_SIZE', 1)) > 1

    if args.distributed:
        torch.cuda.set_device(device=args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    if args.local_rank == 0:
        if not os.path.exists('weights'):
            os.makedirs('weights')

    profile(args, img_array)

    demo(args,img_array)


if __name__ == "__main__":
    main()
