import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.utils.draw import compute_color_for_labels
from deep_sort_pytorch.deep_sort import DeepSort

sys.path.insert(0, './yolov5')

from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, check_imshow, non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device, time_sync

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.75
font_thickness = 1
font_color = [255, 255, 255]
bbox_linewidth = 2

fps_scale = 0.6
fps_thickness = 2
fps_color = [0, 0, 0]
fps_back = [220, 220, 220]


def draw_fps(img, frametime):
    fps_text = 'FPS: {:.2f}'.format(1 / frametime)
    fps_width, fps_height = cv2.getTextSize(fps_text, font, fps_scale, fps_thickness)[0]
    
    fps_shift = int(fps_height * 1.5)
    fps_height *= 2
    
    cv2.rectangle(img, (0, 0), (fps_width, fps_height), fps_back, cv2.FILLED)
    cv2.putText(img, fps_text, (0, fps_shift), font, fps_scale, fps_color, fps_thickness)
    
    return img


def draw_boxes(img, bbox, identities=None, confs=None, classes=None, names=None, offset=(0,0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1, y1, x2, y2 = x1 + offset[0], y1 + offset[1], x2 + offset[0], y2 + offset[1]
        
        label_id = int(identities[i]) if identities is not None else 0
        label_conf = ' {:.2f}'.format(confs[i]) if confs is not None else ""
        label_class = ' {}'.format(names[int(classes[i])] if names is not None else \
            int(classes[i])) if classes is not None else ""
        
        label = '{:d}'.format(label_id) + label_class + label_conf
        label_width, label_height = cv2.getTextSize(label, font, fontScale=font_scale, \
            thickness=font_thickness)[0]
        
        bbox_color = compute_color_for_labels(label_id)
        cv2.rectangle(img, (x1, y1), (x2, y2), bbox_color, bbox_linewidth)
        cv2.rectangle(img, (x1, y1), (x1 + label_width, y1 - (label_height + 3)), bbox_color, \
            cv2.FILLED)
        cv2.putText(img, label, (x1, y1 - 2), font, font_scale, font_color, font_thickness, \
            lineType=cv2.LINE_AA)
    
    return img


def xyxy_to_xywh(*xyxy):
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    width = abs(xyxy[0].item() - xyxy[2].item())
    height = abs(xyxy[1].item() - xyxy[3].item())
    x_center = (bbox_left + width / 2)
    y_center = (bbox_top + height / 2)
    
    return [x_center, y_center, width, height]


def run(deepsort_config='./deep_sort_pytorch/configs/deep_sort.yaml',
        yolo_weights='./yolov5/weights/yolov5s.pt',
        source='0',
        output='./results',
        imgsz=640,
        conf_thres=0.5,
        iou_thres=0.5,
        device='',
        half=False,
        view_img=False,
        save_img=False,
        classes=None,
        agnostic_nms=False,
        augment=False):
    
    # Initialize the DeepSort Model
    cfg = get_config()
    cfg.merge_from_file(deepsort_config)
    deepsort = DeepSort(model_path=cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST,
                        min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE,
                        n_init=cfg.DEEPSORT.N_INIT,
                        nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)
    
    # Save the video if flag is set and it's not a .txt file
    save_img = save_img and not source.endswith('.txt')
    
    # Determine if the source is a webcam
    webcam = source.isnumeric() or source.endswith('.txt') or \
        source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    
    # Load the GPU
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA
    
    # Initialize the YOLOv5 Model
    model = attempt_load(yolo_weights, map_location=device)  # load FP32 model
    if half:
        model.half()  # to FP16
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    
    # Initialize Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs
    
    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once
    
    # Create directory to save results
    save_path = str(Path(output))
    if save_img and not os.path.isdir(save_path):
        os.makedirs(save_path)
    
    # Loop through the Dataloader
    for path, img, im0s, vid_cap in dataset:
        t1 = time_sync()
        
        # Convert the image as a Tensor
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        
        # YOLOv5 Inference
        pred = model(img, augment=augment)[0]
        
        # YOLOv5 NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms)
        
        # Process YOLOv5 predictions
        for i, det in enumerate(pred):
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], f'{i}: ', im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s.copy()
            
            # Get the source name and width/height
            p = Path(p)
            save_path = str(Path(output) / p.name)
            s += '%gx%g ' % img.shape[2:]  # print string
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                
                # Convert YOLOv5 bbox format to DeepSort format
                xywhs, confs, clss = [], [], []
                for *xyxy, conf, cls in reversed(det):
                    xywhs.append(xyxy_to_xywh(*xyxy))
                    confs.append(conf.item())
                    clss.append(int(cls.item()))
                
                # Forward YOLOv5 detections to DeepSort
                ds_dets = deepsort.update(torch.Tensor(xywhs), torch.Tensor(confs), im0, clss)
                
                # Draw bounding boxes from DeepSort
                if len(ds_dets):
                    bbox = ds_dets[:, :4]
                    identities = ds_dets[:, 4]
                    class_names = ds_dets[:, 5]
                    class_confs = ds_dets[:, 6]
                    draw_boxes(im0, bbox, identities, class_confs, class_names, names)
            else:
                deepsort.update_ages()
            t2 = time_sync()
            
            # Print time (YOLOv5 Detection + YOLOv5 NMS + DeepSort Tracking)
            draw_fps(im0, t2 - t1)
            print(f'{s}Done. ({t2 - t1:.3f}s)')
            
            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond
            
            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deepsort_config", type=str, default="./deep_sort_pytorch/configs/deep_sort.yaml", help='deepsort config file')
    parser.add_argument('--yolo_weights', type=str, default='./yolov5/weights/yolov5s.pt', help='yolov5 weights file')
    parser.add_argument('--source', type=str, default='0', help='video file or 0 for webcam')
    parser.add_argument('--output', type=str, default='./results', help='output folder path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--view-img', action='store_true', help='show video tracking')
    parser.add_argument('--save-img', action='store_true', help='save video tracking results')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    with torch.no_grad():
        run(**vars(opt))

