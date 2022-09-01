import csv
import time
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import trange

import sys
sys.path.append(".")
from whitebox_adversarial.objective import WhiteboxWithOneSystem
from image2perturbed import LinfPerturbation, L2Perturbation#, WhiteboxGlassesPerturbation, WhiteboxRectanglePerturbation

from scipy.optimize import fmin as scipy_optimizer
from scipy.stats import weibull_min

c_init = 1.0

my_systems = ['FaceNet', 'r50', 'EfficientNet', 'ReXNet', 'AttentionNet', 'RepVGG', 'GhostNet', 'TF-NAS', 'LightCNN']

clever_scores = {}
clever_time = {}

for Perturbation in [LinfPerturbation, L2Perturbation]:
	
	perturb = Perturbation()

	for my_system in my_systems:

		box = WhiteboxWithOneSystem(my_system)

		clever_scores[my_system, perturb.pname] = []
		t_0 = time.time()
		
		print(my_system, perturb.pname)

		for k in trange(6000):

			i1 = np.expand_dims(np.array(Image.open(f"../lfw/image_{k}_A.jpg").resize((112, 112))), 0) 
			i2 = np.expand_dims(np.array(Image.open(f"../lfw/image_{k}_B.jpg").resize((112, 112))), 0) 
			same = np.sign(int(k%600 < 300) - 0.5)
			if same == -1:
				continue

			max_g_1_norms = []
			for i in range(perturb.num_step):
				i1_ = perturb.perturb_single_image(i1)

				sim_pred = box.cosine_similarity(i1, i2)

				g_1, _ = box.get_grads(i1_, i2)
				# sim_pred_ = box.cosine_similarity(i1_, i2)

				g_1_norms = np.linalg.norm(g_1.reshape(len(g_1), -1), axis=1)
				max_g_1_norms.append(max(g_1_norms))

				# print(max(g_1_norms))
				# Image.fromarray(i1_.squeeze(0).astype(np.uint8)).save(f"../lfw/image_{k}_A_{perturb.pname}_{my_system}.png")

			[_, loc, _] = weibull_min.fit(-np.array(max_g_1_norms), c_init, optimizer=scipy_optimizer)
			clever_score = -(sim_pred * same) / loc

			print(clever_score)
			
			clever_scores[my_system, perturb.pname].append(clever_score[0])

		pd.DataFrame(clever_scores).to_excel('clever_scores.xlsx')
		clever_time[my_system, perturb.pname] = [time.time() - t_0, time.time() - t_0]
		pd.DataFrame(clever_time).to_excel('clever_time.xlsx')


	# for my_system in ('r50', ):

	# 	box = WhiteboxWithMultipleSystems(my_system)

	# 	for k in trange(6000):

	# 		# with open(f'y_preds_rex_{k}.csv', 'w', newline ='') as f:
	# 			# write = csv.writer(f)

	# 		i1 = np.expand_dims(np.array(Image.open(f"../lfw/image_{k}_A.jpg").resize((112, 112))), 0) 
	# 		i2 = np.expand_dims(np.array(Image.open(f"../lfw/image_{k}_B.jpg").resize((112, 112))), 0) 
	# 		same = np.sign(int(k%600 < 300) - 0.5)

	# 		i1_ = np.array(i1, copy=True)

	# 		min_bce = 1

	# 		for i in range(2000):
	# 			g_1, _ = box.get_grads(i1_, i2)
	# 			i1_ = perturb.perturb_single_image(i1, i1_, same * g_1)
	# 			sim_pred_ = box.cosine_similarity(i1_, i2)

	# 			if sim_pred_ * same < min_bce:
	# 				min_bce = sim_pred_ * same
	# 				Image.fromarray(i1_.squeeze(0).astype(np.uint8)).save(f"../lfw/image_{k}_A_{perturb.pname}_{my_system}.png")

	# 			print(box.y_preds)