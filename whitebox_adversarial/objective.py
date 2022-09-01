import sys
sys.path.append(".")

import torch
import torch.nn.functional as F

import numpy as np
from blackbox_adversarial.image2feature import get_model


class WhiteboxWithOneSystem:
	def __init__(self, backbone_type):
		super(WhiteboxWithOneSystem, self).__init__()
		self.m1 = get_model(backbone_type)

	def _shared_calculation(self, im1, im2):

		images = np.concatenate((im1, im2), axis=0)
		image = (images.transpose((0, 3, 1, 2)) - 127.5) / 128.0
		image = torch.from_numpy(image.astype(np.float32)).contiguous()

		image_ = image.to(next(self.m1['m'].parameters()).device)
		image_.requires_grad = True
		feature_ = self.m1['m'](image_)
		
		f1 = feature_[:im1.shape[0]]
		f2 = feature_[im1.shape[0]:]
		
		self.m1['m'].zero_grad()
		return F.cosine_similarity(f1, f2), image_

	def cosine_similarity(self, im1, im2):
		with torch.no_grad():
			return self._shared_calculation(im1, im2)[0].cpu().numpy()

	def get_grads(self, im1, im2):
		loss, image_ = self._shared_calculation(im1, im2)
		(-loss.mean()).backward()

		grad = image_.grad.data.cpu().numpy()
		g1 = grad[:im1.shape[0]].transpose((0, 2, 3, 1))
		g2 = grad[im1.shape[0]:].transpose((0, 2, 3, 1))

		return g1, g2

class WhiteboxWithMultipleSystems(WhiteboxWithOneSystem):
	def __init__(self, *backbone_type):
		super(WhiteboxWithMultipleSystems, self).__init__(backbone_type[0])
		self.ms = [get_model(x) for x in backbone_type]

	def _shared_calculation(self, im1, im2):

		images = np.concatenate((im1, im2), axis=0)
		image = (images.transpose((0, 3, 1, 2)) - 127.5) / 128.0
		image = torch.from_numpy(image.astype(np.float32)).contiguous()

		image_ = image.to(next(self.m1['m'].parameters()).device)
		image_.requires_grad = True

		sims = []
		for m in self.ms:
			feature_ = m['m'](image_)
			
			f1 = feature_[:im1.shape[0]]
			f2 = feature_[im1.shape[0]:]
			
			m['m'].zero_grad()
			sims.append(F.cosine_similarity(f1, f2))
		self.sims = [x.item() for x in sims]
		self.y_preds = [int(np.arccos(x) < m['threshold']) for x in self.sims]
		return torch.stack(sims).mean(0), image_