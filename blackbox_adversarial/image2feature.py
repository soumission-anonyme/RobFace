import sys
sys.path.append(".")
sys.path.append('..')

from test_protocol.utils.model_loader import ModelLoader
from backbone.backbone_def import BackboneFactory

import torch
import torch.nn.functional as F

import numpy as np

from insightface.recognition.arcface_torch.backbones import get_model as insf_get_model
from facenet_pytorch import InceptionResnetV1

thresholds = {
	'FaceNet':64.01,
	'r50':76.63,
	'EfficientNet':76.95,
	'ReXNet':76.29,
	'AttentionNet':74,
	'RepVGG':76.61,
	'GhostNet':77.78,
	'TF-NAS':75.86,
	'LightCNN':77.52
}

def get_model(backbone_type):

	if backbone_type in ('r18', 'r34', 'r50', 'r100'):
		m = insf_get_model(backbone_type)
		m.load_state_dict(torch.load("checkpoints/" + backbone_type + ".pt"))

	elif backbone_type == "FaceNet":
		m = InceptionResnetV1(pretrained='vggface2')

	else:	
		backbone_factory = BackboneFactory(backbone_type, "test_protocol/backbone_conf.yaml")
		model_loader = ModelLoader(backbone_factory)
		m = model_loader.load_model("checkpoints/" + backbone_type + ".pt")

	return {
		'm':m.cuda().eval(),
		'threshold':np.deg2rad(thresholds[backbone_type])
	}


		

def get_feature(image, m, batched = True, mini_batch_size = 256): # with mini-batch
	if len(image.shape) == 3:
		image = np.expand_dims(image, axis=0)
		batched = False
	image = (image.transpose((0, 3, 1, 2)) - 127.5) / 128.0
	image = torch.from_numpy(image.astype(np.float32)).contiguous()
	
	features = []
	with torch.no_grad():
		for i in range(int(np.ceil(len(image) / mini_batch_size))):
			image_ = image[mini_batch_size * i:mini_batch_size * (i + 1)].to(next(m.parameters()).device)
			feature_ = m(image_)
			feature_ = F.normalize(feature_)
			features.append(feature_.cpu().numpy())

	feature = np.concatenate(features, axis=0)
		
	if not batched:
		feature = feature[0]

	return feature


