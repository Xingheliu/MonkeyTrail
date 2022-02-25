import os
import sys
sys.path.append(os.getcwd()+'/pro/deeplearing_pro/yolov5-master/')

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import csv
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (check_file, check_img_size, check_imshow, check_requirements,
                           increment_path, non_max_suppression, scale_coords, xyxy2xywh)
from utils.torch_utils import select_device



@torch.no_grad()
def run(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        save_csv='',  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        input_data=None,
        ):
    source = str(source)
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if True else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= (pt or jit or engine) and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size


    # Run inference
    model.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    csv_all_num, csv_num, frame_num = len(input_data), 0, 0
    for path, im, im0s, vid_cap, s in dataset:
        frame_num = frame_num+1
        if csv_num <csv_all_num:
            if frame_num != input_data[csv_num][0]:
                continue
            else:
                csv_num = csv_num + 1
        else:
            break
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_csv!='' and conf.item()>=0.8:  # Write to file
                        with open(save_csv, 'a', newline='') as f:
                            f_csv = csv.writer(f)
                            headers = [frame_num, cls.item(), conf.item(), xyxy[0].item(), xyxy[2].item(), xyxy[1].item(), xyxy[3].item()]
                            f_csv.writerow(headers)


class YOLO_M():
    def __init__(self,video_save_path):
        self.PATH_TO_OUTPUT_INFO = video_save_path+'/temp2.csv'
        mf = pd.read_csv( video_save_path+'/temp1.csv')  # 返回一个DataFrame的对象，这个是pandas的一个数据结构
        mf.columns = ['frame', 'x1', 'x2', 'y1', 'y2']
        xm = mf[['frame', 'x1', 'x2', 'y1', 'y2']]
        self.input_data = np.array(xm)


    def yolo_detect(self,video_input):
        with open(self.PATH_TO_OUTPUT_INFO, 'w', newline='') as f:
            f_csv = csv.writer(f)
            headers = ['frame_num', 'classname', 'score', 'x1', 'x2', 'y1', 'y2']
            f_csv.writerow(headers)
        check_requirements(exclude=('tensorboard', 'thop'))
        run(weights=os.getcwd()+'/pro/model/yolo/monkey_200.pt',imgsz=[640,640], source=video_input ,save_csv=self.PATH_TO_OUTPUT_INFO ,input_data=self.input_data)





