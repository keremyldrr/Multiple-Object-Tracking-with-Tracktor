import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
import motmetrics as mm
from torchvision.ops.boxes import clip_boxes_to_image, nms


class Tracker:
	"""The main tracking file, here is where magic happens."""

	def __init__(self, obj_detect):
		self.obj_detect = obj_detect
		self.tracks = []
		self.track_num = 0
		self.im_index = 0
		self.results = {}
		# self.prev_boxes = None
		self.mot_accum = None

	def reset(self, hard=True):
		self.tracks = []

		if hard:
			self.track_num = 0
			self.results = {}
			self.im_index = 0

	def add(self, new_boxes, new_scores,features=None):
		"""Initializes new Track objects and saves them."""
		num_new = len(new_boxes)
		for i in range(num_new):
			if features is None:
				self.tracks.append(Track(new_boxes[i],new_scores[i],self.track_num + i))
			else:
				self.tracks.append(Track(new_boxes[i],new_scores[i],self.track_num + i,features[i]))

		self.track_num += num_new

	def get_pos(self):
		"""Get the positions of all active tracks."""
		if len(self.tracks) == 1:
			box = self.tracks[0].box
		elif len(self.tracks) > 1:
			box = torch.stack([t.box for t in self.tracks], 0)
		else:
			box = torch.zeros(0).cuda()
		return box

	def data_association(self, boxes, scores):
		self.tracks = []
		self.add(boxes, scores)

	def step(self, frame):
		"""This function should be called every timestep to perform tracking with a blob
		containing the image information.
		"""
		boxes, scores = self.obj_detect.detect(frame['img'])
		#features = self.reid_network.test_rois(frame["img"].cpu(), obj_detect.detect(frame["img"].cuda())[0])

		self.data_association(boxes, scores)

		# results
		for t in self.tracks:
			if t.id not in self.results.keys():
				self.results[t.id] = {}
			self.results[t.id][self.im_index] = np.concatenate([t.box.cpu().numpy(), np.array([t.score])])

		self.im_index += 1

	def get_results(self):
		return self.results


class Track(object):
	"""This class contains all necessary for every individual track."""
	def __init__(self, box, score, track_id,feature=None):
		self.id = track_id
		self.box = box
		self.score = score
		self.active = True
		if feature is not None:
			self.features = deque(feature.unsqueeze(0).cuda())
			self.inactive_count = 0
			self.prev_boxes = deque([box.clone()], maxlen=2)# constant velocity assumption
			self.last_velo = torch.Tensor([])

	def has_positive_area(self):
		return self.box[0,2] > self.box[0,0] and self.box[0,3] > self.box[0,1]

	def add_features(self, features):
		"""Adds new appearance features to the object."""
		self.features.append(features[0])
		if len(self.features) > 10:
			self.features.popleft()
	# def test_features(self, test_features):
	# 	# """Compares test_features to features of this Track object"""
	# 	num_feats = len(self.features)
	# 	step_size = 1/num_feats
	# 	if len(self.features) > 1:
	# 		features = torch.cat(list(self.features), dim=0).reshape(len(self.features),128)
	# 		features = torch.Tensor(torch.arange(0, 1, step_size).cpu().detach().numpy().dot(features.cpu().detach().numpy()))
	# 	else:
	# 		features = self.features[0]

	# 	dist = F.pairwise_distance(features.cpu(), test_features.cpu(),keepdim=True)
	# 	return dist
	def test_features(self, test_features):
		"""Compares test_features to features of this Track object"""
		if len(self.features) > 1:
			features = torch.cat(list(self.features), dim=0)
		else:
			features = self.features[0]
		features = features.mean(0, keepdim=True)
		dist = F.pairwise_distance(features, test_features, keepdim=True)
		return dist

	def reset_previous_boxes(self):
		self.prev_boxes.clear()
		self.prev_boxes.append(self.box.clone())