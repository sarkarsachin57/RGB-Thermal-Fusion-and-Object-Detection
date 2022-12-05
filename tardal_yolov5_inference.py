import os, cv2, shutil, PIL, random, torch, time, torchvision, gc, argparse
import numpy as np
from PIL import Image
# from google.colab.patches import cv2_imshow
# from tqdm.notebook import tqdm
from torch2trt import torch2trt
from torch2trt import TRTModule
        
class Inference_TarDal_YoloV5_TRT:

    def __init__(self, tardal_trt, yolo_trt):

        fusion_trt = TRTModule()
        fusion_trt.load_state_dict(torch.load(tardal_trt))

        det_trt = TRTModule()
        det_trt.load_state_dict(torch.load(yolo_trt))


        self.fusion_trt = fusion_trt
        self.det_trt = det_trt

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # to infer tardal first time
        random = torch.ones([1, 768, 1024]).to(self.device)
        self.fusion_trt(random.unsqueeze(0), random.unsqueeze(0))

        # random = torch.ones([1, 3, 768, 1024]).to(self.device)
        # self.det_trt(random)

        print('TarDal and YoloV5 models loaded successfully.')



    def box_iou(self, box1, box2, eps=1e-7):
        # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            box1 (Tensor[N, 4])
            box2 (Tensor[M, 4])
        Returns:
            iou (Tensor[N, M]): the NxM matrix containing the pairwise
                IoU values for every element in boxes1 and boxes2
        """

        # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
        (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
        inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

        # IoU = inter / (area1 + area2 - inter)
        return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


    def xywh2xyxy(self, x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  if (x[:, 0] - x[:, 2] / 2).all() >= 0 else 0 # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  if (x[:, 1] - x[:, 3] / 2).all() >= 0 else 0  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  if (x[:, 0] + x[:, 2] / 2).all() >= 0 else 0  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  if (x[:, 1] + x[:, 3] / 2).all() >= 0 else 0 # bottom right y
        
        return y


    def non_max_suppression(
            self,
            prediction,
            conf_thres=0.25,
            iou_thres=0.45,
            classes=None,
            agnostic=False,
            multi_label=False,
            labels=(),
            max_det=300,
            nm=0,  # number of masks
    ):
        """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections
        Returns:
            list of detections, on (n,6) tensor per image [xyxy, conf, cls]
        """

        if isinstance(prediction, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
            prediction = prediction[0]  # select only inference output

        device = prediction.device
        mps = 'mps' in device.type  # Apple MPS
        if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
            prediction = prediction.cpu()
        bs = prediction.shape[0]  # batch size
        nc = prediction.shape[2] - nm - 5  # number of classes
        xc = prediction[..., 4] > conf_thres  # candidates

        # Checks
        assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
        assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

        # Settings
        # min_wh = 2  # (pixels) minimum box width and height
        max_wh = 7680  # (pixels) maximum box width and height
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
        time_limit = 0.5 + 0.05 * bs  # seconds to quit after
        redundant = True  # require redundant detections
        multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
        merge = False  # use merge-NMS

        t = time.time()
        mi = 5 + nc  # mask start index
        output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x[xc[xi]]  # confidence

            # Cat apriori labels if autolabelling
            if labels and len(labels[xi]):
                lb = labels[xi]
                v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
                v[:, :4] = lb[:, 1:5]  # box
                v[:, 4] = 1.0  # conf
                v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
                x = torch.cat((x, v), 0)

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Compute conf
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

            # Box/Mask
            box = self.xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
            mask = x[:, mi:]  # zero columns if no masks

            # Detections matrix nx6 (xyxy, conf, cls)
            if multi_label:
                i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
                x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
            else:  # best class only
                conf, j = x[:, 5:mi].max(1, keepdim=True)
                x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

            # Filter by class
            if classes is not None:
                x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

            # Apply finite constraint
            # if not torch.isfinite(x).all():
            #     x = x[torch.isfinite(x).all(1)]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            elif n > max_nms:  # excess boxes
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence
            else:
                x = x[x[:, 4].argsort(descending=True)]  # sort by confidence

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
            if i.shape[0] > max_det:  # limit detections
                i = i[:max_det]
            if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
                # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = self.box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy

            output[xi] = x[i]
            if mps:
                output[xi] = output[xi].to(device)
            if (time.time() - t) > time_limit:
                raise TimeoutError(f'WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded')
                break  # time limit exceeded

        # output = make_relu(output[0])
        return output[0]

    def make_box_limit(self, x, img):
        h, w = x.shape[:2]
        height, width = img.shape[-2:]
        # print(height, width)
        for i in range(h):
            for j in range(w):
                x[i, j] = max(0, x[i, j])
                if j in [0,2]:
                    x[i, j] = int(min(width, x[i, j]))
                if j in [1,3]:
                    x[i, j] = int(min(height, x[i, j]))
        return x


    def draw_text(self, img, text,
            pos=(0, 0),
            font=cv2.FONT_HERSHEY_PLAIN,
            font_scale=3,
            text_color=(0, 255, 0),
            font_thickness=2,
            text_color_bg=(0, 0, 0)
            ):

        #font_thickness=1
        x, y = pos
        font_scale = 1
        font = cv2.FONT_HERSHEY_PLAIN
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_w, text_h = text_size
        cv2.rectangle(img, (x, y - text_h - 10), (x + text_w + 10, y), text_color_bg, -1)
        cv2.putText(img, text, (x+5, y-5), font, font_scale, text_color, font_thickness)


    def draw_bb_text(self, frame, text,
            bbox,
            font=cv2.FONT_HERSHEY_PLAIN,
            font_scale=3,
            text_color=(0, 255, 0),
            font_thickness=2,
            text_color_bg=(255, 255, 255)
            ):

        startX, startY, endX, endY = bbox
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_w, text_h = text_size
        #cv2.rectangle(img, (x, y - text_h - 2), (x + text_w, y + 2), text_color_bg, -1)
        startY = 20 if startY < 20 else startY
        startX = 1 if startX < 1 else startX
        bg = np.ones_like(frame[startY-20:startY,startX-1:startX+text_w+3]).astype('uint8') * 255
        bg[:,:] = text_color_bg
        frame[startY-20:startY,startX-1:startX+text_w+3] = cv2.addWeighted(frame[startY-20:startY,startX-1:startX+text_w+3], 0.0, bg, 1.0, 1)
        cv2.putText(frame, text, (startX, startY-text_h+2), font, font_scale, text_color, font_thickness)




    def infer_det(self, image_path, conf_thres = 0.5, iou_thres = 0.5, visualize = False, save = False, save_path = None):

        # data prepare
        img_np = np.array(Image.open(image_path))
        img = torch.tensor(img_np).permute(2, 0, 1).unsqueeze(0) / 255.
        img = img.to('cuda')

        # inference
        current = time.time() 
        out = self.det_trt(img)
        infer_time = time.time() - current
        print('infer time :', infer_time)
        out = self.non_max_suppression(out, conf_thres = conf_thres, iou_thres = iou_thres).cpu().detach().numpy()
        final_preds = self.make_box_limit(out, img)

        # post processing
        classes = np.array(['People', 'Car', 'Bus', 'Motorcycle', 'Lamp', 'Truck'])
        for startX, startY, endX, endY, conf, cls in final_preds:
            startX, startY, endX, endY = int(startX), int(startY), int(endX), int(endY)
            cv2.rectangle(img_np, (startX, startY), (endX, endY), (200, 200, 200), 2) 
            self.draw_bb_text(img_np,classes[int(cls)] + ' conf: ' + str(round(conf,2)), (startX, startY, endX, endY),cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 0, 0), 1, (200, 200, 200))

        # if visualize:
        #     cv2_imshow(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))

        if save:
            cv2.imwrite(save_path, cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))

        del img_np, img, out, final_preds, classes, startX, startY, endX, endY, conf, cls

        gc.collect(), torch.cuda.empty_cache()




    def infer_fusion(self, ir_path, vi_path, visualize = False, save = False, save_path = None):

        # data preparation
        ir_c = cv2.imread(str(ir_path), cv2.IMREAD_GRAYSCALE)
        vi_c = cv2.imread(str(vi_path), cv2.IMREAD_COLOR)
        ir = torch.tensor(ir_c / 255.).float().unsqueeze(0)
        vi = torch.tensor(cv2.cvtColor(vi_c, cv2.COLOR_BGR2GRAY) / 255.).float().unsqueeze(0)
        ir, vi = ir.to(self.device), vi.to(self.device)

        print(ir.shape, vi.shape)

        # inference 
        current = time.time() 
        res = self.fusion_trt(ir.unsqueeze(0), vi.unsqueeze(0))
        infer_time = time.time() - current
        print('infer time :', infer_time)
        
        #post processing
        res = torch.clamp(res, min = 0.0, max = 1.0)
        res = (res * 255)[0][0].cpu().detach().numpy().astype('uint8')  
        cbcr = cv2.cvtColor(vi_c, cv2.COLOR_BGR2YCrCb)[:, :, -2:]            
        fus = np.concatenate([res[..., np.newaxis], cbcr], axis=2)
        fus = cv2.cvtColor(fus, cv2.COLOR_YCrCb2BGR)
        
        # if visualize:
        #     cv2_imshow(fus)

        if save:
            cv2.imwrite(save_path, fus)

        del ir_c, vi_c, ir, vi, res, cbcr, fus

        gc.collect(), torch.cuda.empty_cache()
        
        


    def infer_e2e(self, ir_path, vi_path, conf_thres = 0.5, iou_thres = 0.5, visualize = False, save = False, save_path = None):
        
        # data preparation
        ir_c = cv2.imread(str(ir_path), cv2.IMREAD_GRAYSCALE)
        vi_c = cv2.imread(str(vi_path), cv2.IMREAD_COLOR)
        ir = torch.tensor(ir_c / 255.).float().unsqueeze(0)
        vi = torch.tensor(cv2.cvtColor(vi_c, cv2.COLOR_BGR2GRAY) / 255.).float().unsqueeze(0)
        ir, vi = ir.to(self.device), vi.to(self.device)

        # inference 
        current = time.time()
        self.fusion_trt.eval()
        res = self.fusion_trt(ir.unsqueeze(0), vi.unsqueeze(0))
        tardal_infer_time = time.time() - current
        print(f'TarDal Fusion completed. Time : {tardal_infer_time}')

        #post processing
        res = torch.clamp(res, min = 0.0, max = 1.0)
        res = (res * 255)[0][0].cpu().detach().numpy().astype('uint8')  
        cbcr = cv2.cvtColor(vi_c, cv2.COLOR_BGR2YCrCb)[:, :, -2:]            
        fus = np.concatenate([res[..., np.newaxis], cbcr], axis=2)
        img_np = cv2.cvtColor(fus, cv2.COLOR_YCrCb2RGB)

        img = torch.tensor(img_np).permute(2, 0, 1).unsqueeze(0) / 255.
        img = img.to('cuda')

        # inference 
        current = time.time()
        out = self.det_trt(img)
        yolo_infer_time = time.time() - current
        print(f'Yolo object detection completed. Time : {yolo_infer_time}')

        total_time = tardal_infer_time + yolo_infer_time
        print(f'Total time for inference : {total_time}')

        out = self.non_max_suppression(out, conf_thres = conf_thres, iou_thres = iou_thres).cpu().detach().numpy()
        final_preds = self.make_box_limit(out, img)

        # post processing
        classes = np.array(['People', 'Car', 'Bus', 'Motorcycle', 'Lamp', 'Truck'])
        for startX, startY, endX, endY, conf, cls in final_preds:
            startX, startY, endX, endY = int(startX), int(startY), int(endX), int(endY)
            cv2.rectangle(img_np, (startX, startY), (endX, endY), (200, 200, 200), 2) 
            self.draw_bb_text(img_np,classes[int(cls)] + ' conf: ' + str(round(conf,2)), (startX, startY, endX, endY),cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 0, 0), 1, (200, 200, 200))

        # if visualize:
        #     cv2_imshow(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))

        if save:
            cv2.imwrite(save_path, cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))

        del ir_c, vi_c, ir, vi, res, cbcr, fus, img_np, img, out, final_preds, classes, startX, startY, endX, endY, conf, cls

        gc.collect(), torch.cuda.empty_cache()




if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                    prog = 'Inference_TarDal_YoloV5_TRT',
                    description = 'Inference the TarDal Image Fusion and YoloV5 Object Detection')

    # Required
    parser.add_argument('--tardal_path', help = 'Path to TarDal model.', type = str, required = True) 
    parser.add_argument('--yolo_path', help = 'Path to YoloV5 model.', type = str, required = True) 
    parser.add_argument('--ir_path', help = 'Path to IR Image', type = str, required = True)
    parser.add_argument('--vi_path', help = 'Path to RGB Image', type = str, required = True)
    parser.add_argument('--out_dir', help = 'Output directory path where results stored.', type = str, required = True)

    # Optional if needed
    parser.add_argument('--conf_thresh', default = 0.5, help = 'Confidence threshold for Object Detections', type = float)
    parser.add_argument('--iou_thresh', default = 0.5, help = 'IoU threshold for Object Detections', type = float)

    args = parser.parse_args()

    inference_pipeline = Inference_TarDal_YoloV5_TRT(args.tardal_path, args.yolo_path)

    file_name = args.ir_path.split('/')[-1]

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    save_path = os.path.join(args.out_dir, file_name)

    # inference_pipeline.infer_det(args.vi_path, args.conf_thresh, args.iou_thresh, save = True, save_path = save_path)

    # inference_pipeline.infer_fusion(args.ir_path, args.vi_path, save = True, save_path = save_path)

    inference_pipeline.infer_e2e(args.ir_path, args.vi_path, args.conf_thresh, args.iou_thresh, save = True, save_path = save_path)


