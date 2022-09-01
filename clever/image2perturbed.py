# import sys
# sys.path.append(".")

import numpy as np

# from blackbox_adversarial.image2perturbed import GlassesPerturbation
# from algorithms import fgsm_attack

# class WhiteboxGlassesPerturbation(GlassesPerturbation):
# 	def __init__(self):
# 		super(WhiteboxGlassesPerturbation, self).__init__()
# 		self.pname = 'wbglass'
# 		self.num_step = 1000

# 	def perturb_single_image(self, i1, i1_, g_1, step_size = 2):
# 		mask = np.expand_dims(self.get_glasses_mask_for_image(i1.squeeze(0)).astype(bool), 0)
# 		i1_[mask] = fgsm_attack(i1_[mask], step_size, g_1[mask])
# 		i1_ = np.clip(i1_, 0, 255)
# 		return i1_

# class WhiteboxRectanglePerturbation:
# 	def __init__(self):
# 		super(WhiteboxRectanglePerturbation, self).__init__()
# 		self.pname = 'wbrect'
# 		self.num_step = 500

# 	def perturb_single_image(self, i1, i1_, g_1, step_size = 2):
# 		i1_[:, 0:33, 40:73] = fgsm_attack(i1_[:, 0:33, 40:73], step_size, g_1[:, 0:33, 40:73])
# 		i1_ = np.clip(i1_, 0, 255)
# 		return i1_

class LinfPerturbation:
	def __init__(self):
		super(LinfPerturbation, self).__init__()
		self.pname = 'randlinf'
		self.num_step = 10
	
	def perturb_single_image(self, i1, max_step_size = 4, sample_size = 32):
		delta = np.clip(np.random.randn(*np.repeat(i1, sample_size, axis=0).shape), -max_step_size, max_step_size)
		i1_ = np.clip(i1 + delta, 0, 255)
		return i1_

class L2Perturbation:
	def __init__(self):
		super(L2Perturbation, self).__init__()
		self.pname = 'randl2'
		self.num_step = 10
	
	def perturb_single_image(self, i1, max_step_size = 10, sample_size = 32):
		# sample pertubation from uniform l2 ball and add to image
		delta = np.random.randn(*np.repeat(i1, sample_size, axis=0).shape)
		delta_norms = np.linalg.norm(delta, axis=3, keepdims=True)# + 1e-7
		i1_ = np.clip(delta / delta_norms * max_step_size + i1, 0, 255)
		return i1_