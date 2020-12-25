from tracker.tracker import Tracker, Track
import numpy as np
import motmetrics as mm
import torch.nn.functional as F
import torch
from torchvision.models.detection.transform import resize_boxes
from torch.jit.annotations import List, Tuple
import matplotlib.pyplot as plt
# from tracker.utils import nms
from scipy.optimize import linear_sum_assignment
import cv2
import copy
# THRESHOLDS

from torchvision.ops.boxes import nms,clip_boxes_to_image


def get_center(pos):
    x1 = pos[0, 0]
    y1 = pos[0, 1]
    x2 = pos[0, 2]
    y2 = pos[0, 3]
    return torch.Tensor([(x2 + x1) / 2, (y2 + y1) / 2]).cuda()


# get detections
def warp_pos(pos, warp_matrix):
    p1 = torch.Tensor([pos[0, 0], pos[0, 1], 1]).view(3, 1)
    p2 = torch.Tensor([pos[0, 2], pos[0, 3], 1]).view(3, 1)
    p1_n = torch.mm(warp_matrix, p1).view(1, 2)
    p2_n = torch.mm(warp_matrix, p2).view(1, 2)
    return torch.cat((p1_n, p2_n), 1).view(1, -1).cuda()


def bbox_overlaps(boxes, query_boxes):
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray or tensor or variable
    query_boxes: (K, 4) ndarray or tensor or variable
    Returns
    -------
    overlaps: (N, K) overlap between boxes and query_boxes
    """
    if isinstance(boxes, np.ndarray):
        boxes = torch.from_numpy(boxes)
        query_boxes = torch.from_numpy(query_boxes)
        out_fn = lambda x: x.numpy()  # If input is ndarray, turn the overlaps back to ndarray when return
    else:
        out_fn = lambda x: x

    box_areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    query_areas = (query_boxes[:, 2] - query_boxes[:, 0] + 1) * (query_boxes[:, 3] - query_boxes[:, 1] + 1)

    iw = (torch.min(boxes[:, 2:3], query_boxes[:, 2:3].t()) - torch.max(boxes[:, 0:1],
                                                                        query_boxes[:, 0:1].t()) + 1).clamp(min=0)
    ih = (torch.min(boxes[:, 3:4], query_boxes[:, 3:4].t()) - torch.max(boxes[:, 1:2],
                                                                        query_boxes[:, 1:2].t()) + 1).clamp(min=0)
    ua = box_areas.view(-1, 1) + query_areas.view(1, -1) - iw * ih
    overlaps = iw * ih / ua
    return out_fn(overlaps)


def plot_boxes(image, boxes):
    fig, ax = plt.subplots(1)
    img = image[0].cpu().mul(255).permute(1, 2, 0).byte().numpy()
    width, height, _ = img.shape
    #
    ax.imshow(img, cmap='gray')
    #     fig.set_size_inches(width / dpi, height / dpi)
    #
    for box in boxes:
        rect = plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, linewidth=1.0, color="r")
        ax.add_patch(rect)
        # box = boxy[0]
        # rect = plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, linewidth=1.0, color="b")
        # ax.add_patch(rect)
    plt.show()


class TrackerIoUAssignment(Tracker):

    def data_association(self, boxes, scores):
        if self.tracks:
            track_ids = [t.id for t in self.tracks]
            track_boxes = np.stack([t.box.numpy() for t in self.tracks], axis=0)

            distance = mm.distances.iou_matrix(track_boxes, boxes.numpy(), max_iou=0.5)

            # update existing tracks
            remove_track_ids = []
            for t, dist in zip(self.tracks, distance):
                if np.isnan(dist).all():
                    remove_track_ids.append(t.id)
                else:
                    match_id = np.nanargmin(dist)
                    t.box = boxes[match_id]
            self.tracks = [t for t in self.tracks
                           if t.id not in remove_track_ids]

            # add new tracks
            new_boxes = []
            new_scores = []
            for i, dist in enumerate(np.transpose(distance)):
                if np.isnan(dist).all():
                    new_boxes.append(boxes[i])
                    new_scores.append(scores[i])
            self.add(new_boxes, new_scores)

        else:
            self.add(boxes, scores)


class Tracktor(Tracker):
    def bounding_box_regression(self, image, prev_boxes):
        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        original_image_sizes.append((image.size()[2], image.size()[3]))

        images, targets = self.obj_detect.transform(image.cuda(), None)
        prev_boxes = torch.Tensor(prev_boxes)
        # plot_boxes(image,prev_boxes)

        prev_boxes = resize_boxes(prev_boxes, original_image_sizes[0], images.image_sizes[0])
        feats = self.obj_detect.backbone(images.tensors)
        roi_heads = self.obj_detect.roi_heads
        box_features = roi_heads.box_roi_pool(feats, [prev_boxes.cuda()], images.image_sizes)
        box_features = roi_heads.box_head(box_features)
        class_logits, box_regression = roi_heads.box_predictor(box_features)

        pred_boxes = roi_heads.box_coder.decode(box_regression, [prev_boxes.cuda()])
        pred_boxes = pred_boxes[:, 1:].squeeze(dim=1).detach()  # new boxes
        pred_boxes = resize_boxes(pred_boxes, images.image_sizes[0], original_image_sizes[0])

        pred_scores = F.softmax(class_logits, -1)  # classification scores for new boxes
        pred_scores = pred_scores[:, 1:].squeeze(dim=1).detach()

        return pred_boxes, pred_scores

    def data_association(self, curr_boxes, scores):

        if self.tracks:

            track_boxes = np.stack([t.box.cpu().numpy() for t in self.tracks], axis=0)
            # iou_matrix = 1 - mm.distances.iou_matrix(track_boxes, track_boxes, max_iou=1)

            pred_boxes, pred_scores = self.bounding_box_regression(self.obj_detect.image, track_boxes)
            # update existing tracks

            remove_track_ids = []
            for t, score, box in zip(self.tracks, pred_scores, pred_boxes):
                if (score < self.s_active).item():
                    remove_track_ids.append(t.id)
                else:
                    t.box = box
                    t.score = score
            self.tracks = [t for t in self.tracks if t.id not in remove_track_ids]
            if len(self.tracks) == 0:
                return
            track_boxes = np.stack([t.box.cpu().numpy() for t in self.tracks], axis=0)
            track_scores = np.stack([t.score.cpu().numpy() for t in self.tracks], axis=0)

            keep_track_ids = nms(torch.Tensor(track_boxes), torch.Tensor(track_scores), self.l_active)  # nms for tracks
            # add new tracks
            new_tracks = []
            for id in keep_track_ids:
                new_tracks.append(self.tracks[id])
            self.tracks = new_tracks
            #
            new_boxes = []
            new_scores = []
            distance = mm.distances.iou_matrix(track_boxes, curr_boxes, max_iou=self.l_new)
            for i, dist in enumerate(np.transpose(distance)):
                if np.isnan(dist).all():
                    new_boxes.append(curr_boxes[i])
                    new_scores.append(scores[i])

            # self.add(det_pos, det_scores)
            self.add(new_boxes, new_scores)

        else:

            self.add(curr_boxes, scores)

    def init_thresholds(self, l_active=0.95, s_active=0.5, l_new=0.1):
        self.l_active = l_active
        self.s_active = s_active
        self.l_new = l_new


class Tracktor_plus(Tracker):
    def __init__(self, obj_detect, reid):
        self.obj_detect = obj_detect
        self.reid_network = reid
        self.active_tracks = []
        self.inactive_tracks = []
        self.track_num = 0
        self.im_index = 0
        self.results = {}
        # self.prev_boxes = None
        self.mot_accum = None
        self.reid_sim_threshold = 2.5
        self.inactive_patience = 10
        self.reid_iou_threshold = 0.2
        self.det_threshold = 0.5

        self.prev_image = []
        self.num_ims = 0
        self.number_of_iterations= 100
        # Threshold increment between two iterations (original 0.001)
        self.termination_eps=0.00001
        print(self.det_threshold)
    def step(self, frame):
        """This function should be called every timestep to perform tracking with a blob
        containing the image information.
        """
        boxes, scores = self.obj_detect.detect(frame['img'])

        self.data_association(boxes, scores)

        # results
        for t in self.active_tracks:
            t.prev_boxes.append(t.box.clone())
            if t.id not in self.results.keys():
                self.results[t.id] = {}
            self.results[t.id][self.im_index] = np.concatenate([t.box.cpu().numpy(), np.array([t.score])])

        self.im_index += 1
        self.tracks = self.active_tracks

        self.prev_image = self.obj_detect.image
        self.num_ims+=1
    def add(self, new_boxes, new_scores, features=None):
        """Initializes new Track objects and saves them."""
        num_new = len(new_boxes)
        for i in range(num_new):
            self.active_tracks.append(Track(new_boxes[i], new_scores[i], self.track_num + i, features[i]))

        self.track_num += num_new

    def init_thresholds(self, l_active=0.95, s_active=0.5, l_new=0.1):
        self.l_active = l_active
        self.s_active = s_active
        self.l_new = l_new

    def bounding_box_regression(self, image, prev_boxes):
        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        original_image_sizes.append((image.size()[2], image.size()[3]))

        images, targets = self.obj_detect.transform(image.cuda(), None)
        prev_boxes = torch.Tensor(prev_boxes)
        # plot_boxes(image,prev_boxes)

        prev_boxes = resize_boxes(prev_boxes.squeeze(1), original_image_sizes[0], images.image_sizes[0])
        feats = self.obj_detect.backbone(images.tensors)
        roi_heads = self.obj_detect.roi_heads
        box_features = roi_heads.box_roi_pool(feats, [prev_boxes.cuda()], images.image_sizes)
        box_features = roi_heads.box_head(box_features)
        class_logits, box_regression = roi_heads.box_predictor(box_features)

        pred_boxes = roi_heads.box_coder.decode(box_regression, [prev_boxes.cuda()])
        pred_boxes = pred_boxes[:, 1:].squeeze(dim=1).detach()  # new boxes
        pred_boxes = resize_boxes(pred_boxes, images.image_sizes[0], original_image_sizes[0])

        pred_scores = F.softmax(class_logits, -1)  # classification scores for new boxes
        pred_scores = pred_scores[:, 1:].squeeze(dim=1).detach()

        return pred_boxes, pred_scores

    def reid(self, image, new_boxes, new_scores):  # a lot of work to do
        """Tries to ReID inactive tracks with provided detections."""
        new_boxes = torch.cat(new_boxes).reshape(len(new_boxes), 4)
        new_det_features = self.reid_network.test_rois(image, new_boxes).data

        if len(self.inactive_tracks) >= 1:
            # calculate appearance distances
            dist_mat, pos = [], []
            for t in self.inactive_tracks:
                dist_mat.append(torch.cat([t.test_features(feat.view(1, -1))
                                           for feat in new_det_features], dim=1))
                pos.append(t.box.cuda())
            if len(dist_mat) > 1:
                dist_mat = torch.cat(dist_mat, 0)
                pos = torch.cat(pos).reshape(len(pos), 4)  # torch.cat(pos, 0)
            else:
                dist_mat = dist_mat[0]
                pos = pos[0]

            # calculate IoU distances

            iou = bbox_overlaps(pos.cpu(), new_boxes.cpu()).cuda()
            iou_mask = torch.ge(iou, self.reid_iou_threshold).cpu()
            iou_neg_mask = ~iou_mask
            # make all impossible assignments to the same add big value
            dist_mat = dist_mat.cpu() * iou_mask.float() + iou_neg_mask.float() * 1000
            dist_mat = dist_mat.detach().cpu().numpy()

            row_ind, col_ind = linear_sum_assignment(dist_mat)

            assigned = []
            remove_inactive = []
            
            for r, c in zip(row_ind, col_ind):
                  sc = dist_mat[r, c]
                  #  print("similarity ",sc)
                  if sc <= self.reid_sim_threshold:
                      t = self.inactive_tracks[r]
                      self.active_tracks.append(t)
                      t.count_inactive = 0
                      t.box = new_boxes[c]  # .view(1, -1)
                      t.reset_previous_boxes()
                      # t.reset_last_pos()
                      t.add_features(new_det_features[c].view(1, -1))
                      # t.features = new_det_features[c].view(1, -1)
                      assigned.append(c)
                      # print("REIDENTIFIED")
                      remove_inactive.append(t)

            for t in remove_inactive:
                self.inactive_tracks.remove(t)

            keep = torch.Tensor([i for i in range(new_boxes.size(0)) if i not in assigned]).long().cuda()
            if keep.nelement() > 0:
                new_det_pos = new_boxes[keep]
                new_det_scores = new_scores[keep]
                new_det_features = new_det_features[keep]
            else:
                new_det_pos = torch.zeros(0).cuda()
                new_det_scores = torch.zeros(0).cuda()
                new_det_features = torch.zeros(0).cuda()

        return new_det_pos, new_det_scores, new_det_features

    def data_association(self, curr_boxes, curr_scores):

        if self.active_tracks:


            self.align()
            self.motion()
            self.active_tracks = [t for t in self.active_tracks if t.has_positive_area()]
            track_boxes = np.stack([t.box.cpu().numpy() for t in self.active_tracks], axis=0)  # get current tracks
            track_boxes = clip_boxes_to_image(torch.Tensor(track_boxes),self.obj_detect.image.shape[-2:])

            pred_boxes, pred_scores = self.bounding_box_regression(self.obj_detect.image, track_boxes)
            # update existing tracks
            pred_boxes = clip_boxes_to_image(pred_boxes,self.obj_detect.image.shape[-2:])

            for t, score, box in zip(self.active_tracks, pred_scores, pred_boxes):
                if (score < self.s_active).item() and t.has_positive_area():
                    t.active = False
                    self.inactive_tracks.append(t)
                else:
                    t.prev_boxes.append(t.box.squeeze(0).clone())
                    t.box = box.cpu()
                    t.score = score
                    feats = self.reid_network.test_rois(self.obj_detect.image.cpu(), t.box.unsqueeze(0)).data

                    t.add_features(feats)
            self.active_tracks = [t for t in self.active_tracks if t.active is True]
            # track_features = for now dont update features at each frame
            if len(self.active_tracks) == 0:
                return
            track_boxes = torch.Tensor(np.stack([t.box.cpu().numpy() for t in self.active_tracks], axis=0))
            track_scores = torch.Tensor(np.stack([t.score.cpu().numpy() for t in self.active_tracks], axis=0))

            keep_track_ids = nms(track_boxes, track_scores, self.l_active)
            # add new tracks
            new_tracks = []
            for id in keep_track_ids:
                new_tracks.append(self.active_tracks[id])
            self.active_tracks = new_tracks

            new_boxes = []
            new_scores = []
            new_features = []
            distance = mm.distances.iou_matrix(track_boxes, curr_boxes, max_iou=self.l_new)
            for i, dist in enumerate(np.transpose(distance)):
                if np.isnan(dist).all():
                  if curr_scores[i] >=self.det_threshold:
                    new_boxes.append(curr_boxes[i])
                    new_scores.append(curr_scores[i])
                    # new_features.append(curr_features[i].cpu())
                    new_features.append(torch.zeros(128))

            # track_features =
            # reid_features =
            if len(new_boxes) != 0:
                if len(self.inactive_tracks) != 0:
                    new_boxes, new_scores, new_features = self.reid(self.obj_detect.image.cpu(), new_boxes,torch.Tensor(new_scores))
                else:
                    new_features = self.reid_network.test_rois(self.obj_detect.image.cpu(), torch.cat(new_boxes).unsqueeze(0).reshape(len(new_boxes),4)).detach().cpu()
                for t in self.inactive_tracks:
                    t.inactive_count += 1
                self.inactive_tracks = [t for t in self.inactive_tracks if
                                        t.inactive_count < self.inactive_patience and t.has_positive_area()]  # move boxes with optical flow
                if len(new_boxes) != 0:
                    self.add(new_boxes, new_scores, new_features)
                #self.add(new_boxes, new_scores, new_features)
            #print("REID with thresholded detections")

        else:
            curr_features = self.reid_network.test_rois(self.obj_detect.image.cpu(),
                                             curr_boxes.cpu().detach().cpu())
            inds = np.array(curr_scores) > self.det_threshold
            self.add(curr_boxes[inds], curr_scores[inds], curr_features[inds].cpu())

    def motion_step(self, track):
        track.box = track.box + track.last_velo

    def motion(self):
        """Applies a simple linear motion model that considers the last n_steps steps."""
        for t in self.active_tracks:  # try this with rungekutta
            prev_locations = list(t.prev_boxes)  # keep positions so motion can be estimated

            # avg velocity between each pair of consecutive positions in t.last_pos
            vs = [p2 - p1 for p1, p2 in zip(prev_locations, prev_locations[1:])]
            if len(vs)!=0:
                t.last_velo = torch.stack(vs).mean(dim=0)
                self.motion_step(t)

        for t in self.inactive_tracks:
            if t.last_velo.nelement() > 0:
                self.motion_step(t)

    def align(self):
        """Aligns the positions of active and inactive tracks depending on camera motion."""
        if self.im_index > 0:
            im1 = np.transpose(self.prev_image.cpu().squeeze(0).numpy(), (1, 2, 0))
            im2 = np.transpose(self.obj_detect.image.cpu().squeeze(0).numpy(), (1, 2, 0))
            im1_gray = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
            im2_gray = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, self.number_of_iterations, self.termination_eps)
            cc, warp_matrix = cv2.findTransformECC(im1_gray, im2_gray, warp_matrix, cv2.MOTION_EUCLIDEAN, criteria,None,5)
            warp_matrix = torch.from_numpy(warp_matrix)

            for t in self.active_tracks:
                box = t.box
                if len(box) != 4:
                    box = box.squeeze(0)
                    t.box = warp_pos(box.unsqueeze(0), warp_matrix)
                else:
                    t.box = warp_pos(t.box.unsqueeze(0), warp_matrix)
            for t in self.inactive_tracks:
                box = t.box
                if len(box) != 4:
                    box = box.squeeze(0)
                    t.box = warp_pos(box.unsqueeze(0), warp_matrix)
                else:
                    t.box = warp_pos(t.box.unsqueeze(0), warp_matrix)

            for t in self.active_tracks:
                for i in range(len(t.prev_boxes)):
                    box = t.prev_boxes[i]
                    if len(box) != 4:
                        box = box.squeeze(0)
                        t.prev_boxes[i] = warp_pos(box.unsqueeze(0), warp_matrix)
                    else:
                        t.prev_boxes[i] = warp_pos(box.unsqueeze(0), warp_matrix)
