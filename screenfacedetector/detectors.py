from .facedetectors import S3FD
import numpy as np
from .utils import bb_intersection_over_union


class Detector(object):
    def __init__(self,confidence=0.9, crop_scale=0.40,device='cpu', track_iou_thres=0.1):
        self.confidence = confidence
        self.crop_scale = crop_scale
        self.device = device
        self.track_iou_thres=track_iou_thres

    def detect(self, image):
        pass

    def cropBox(self, box, image, crop_scale=None):
        """
        :param box: [left,top,right,bottom]
        :param image: rgb image
        :return: rgb crop face image
        """
        if crop_scale==None:
            crop_scale = self.crop_scale
        max_len = max(box[2] - box[0], box[3] - box[1])
        bs = max_len / 2
        bsi = int(bs * (1 + 2 * crop_scale))

        frame = np.pad(image, ((bsi, bsi), (bsi, bsi), (0, 0)), 'constant', constant_values=(110, 110))
        my = (box[1] + box[3]) / 2 + bsi  # BBox center Y
        mx = (box[0] + box[2]) / 2 + bsi  # BBox center X

        face = frame[int(my - bs):int(my + bs * (1 + 2 * crop_scale)),
               int(mx - bs * (1 + crop_scale)):int(mx + bs * (1 + crop_scale))]
        return face

    def selectBox(self, bboxes, prvBbox, image):
        """

        :param bboxes: list of bbox [left,top,right,bottom,score]
        :param prv_box: last tracked box
        :param image: rgb image
        :return: rgb crop face image
        """
        if prvBbox is not None:
            # select maximum matching box by iou
            prvBox = prvBbox[:-1]
            matchBox = None
            maxIouScore = 0

            for bbox in bboxes:
                box = bbox[:-1]
                iou = bb_intersection_over_union(box, prvBox)
                if (iou >= self.track_iou_thres and iou > maxIouScore):
                    maxIouScore = iou
                    matchBox = bbox
            if maxIouScore == 0:
                return None
            else:
                return matchBox
        else:
            # select maximum confidence
            maxScore = 0
            matchBox = None
            for bbox in bboxes:
                score = bbox[-1]
                if score > maxScore:
                    maxScore = score
                    matchBox = bbox
            return matchBox


class S3FDDetector(Detector):
    def __init__(self, weights_path, facedet_scale=0.25, **kwargs):
        super().__init__(**kwargs)
        self.facedet_scale = facedet_scale
        self.detector = S3FD(weights_path=weights_path, device=self.device)

    def detect(self, image):
        """
        :param image: image in rgb format
        :return: face bounding box
        """
        # image_np = cv2.cvtColor(im_np, cv2.COLOR_BGR2RGB)
        bboxes = self.detector.detect_faces(image, conf_th=self.confidence, scales=[self.facedet_scale])
        return bboxes
