import sys
sys.path.append(".")

import numpy as np

from blackbox_adversarial.image2perturbed import GlassesPerturbation
from algorithms import fgsm_attack

class WhiteboxGlassesPerturbation(GlassesPerturbation):
	def __init__(self, step_size = 2):
		super(WhiteboxGlassesPerturbation, self).__init__()
		self.pname = 'wbglass'
		self.num_step = 1000
		self.step_size = step_size

	def perturb_single_image(self, i1, i1_, g_1):
		mask = np.expand_dims(self.get_glasses_mask_for_image(i1.squeeze(0)).astype(bool), 0)
		i1_[mask] = fgsm_attack(i1_[mask], self.step_size, g_1[mask])
		i1_ = np.clip(i1_, 0, 255)
		return i1_

class WhiteboxRectanglePerturbation:
	def __init__(self, step_size = 2, u=0, d=33, l=40, r=73):
		super(WhiteboxRectanglePerturbation, self).__init__()
		self.pname = 'wbrect'
		self.num_step = 500
		self.step_size = step_size
		self.u = u
		self.d = d
		self.l = l
		self.r = r

	def perturb_single_image(self, i1, i1_, g_1):
		u, d, l, r = self.u, self.d, self.l, self.r
		i1_[:, u:d, l:r] = fgsm_attack(i1_[:, u:d, l:r], self.step_size, g_1[:, u:d, l:r])
		i1_ = np.clip(i1_, 0, 255)
		return i1_

class LinfPerturbation:
	def __init__(self, step_size = 1, max_step_size = 4):
		super(LinfPerturbation, self).__init__()
		self.pname = 'wblinf_' + str(max_step_size)
		self.num_step = 500
		self.step_size = step_size
		self.max_step_size = max_step_size
	
	def perturb_single_image(self, i1, i1_, g_1):
		i1_ = fgsm_attack(i1_, self.step_size, g_1)
		delta = np.clip(i1 - i1_, -self.max_step_size, self.max_step_size)
		i1_ = np.clip(i1 + delta, 0, 255)
		return i1_

class L2Perturbation:
	def __init__(self, step_size = 1, max_step_size = 10):
		super(L2Perturbation, self).__init__()
		self.pname = 'wbl2_' + str(max_step_size)
		self.num_step = 500
		self.step_size = step_size
		self.max_step_size = max_step_size

	def perturb_single_image(self, i1, i1_, g_1):
		i1_ = fgsm_attack(i1_, self.step_size, g_1)
		delta = i1 - i1_
		delta_norms = np.linalg.norm(delta, axis=3, keepdims=True) + 1e-7
		factor = self.max_step_size / delta_norms
		factor = np.minimum(factor, np.ones_like(delta_norms))
		delta = delta * factor
		i1_ = np.clip(i1 + delta.astype(int), 0, 255)
		return i1_