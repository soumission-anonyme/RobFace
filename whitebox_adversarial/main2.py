import csv
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import trange

from objective import WhiteboxWithOneSystem, WhiteboxWithMultipleSystems
from image2perturbed import L2Perturbation, LinfPerturbation, WhiteboxGlassesPerturbation, WhiteboxRectanglePerturbation


my_systems = ['FaceNet', 'r50', 'EfficientNet', 'ReXNet', 'AttentionNet', 'RepVGG', 'GhostNet', 'TF-NAS', 'LightCNN']

attack_success = {}

for Perturbation in [WhiteboxRectanglePerturbation]:
	
	perturb = Perturbation()

	for my_system in ('r50', ):

		box = WhiteboxWithMultipleSystems(my_system)

		for k in trange(6000):
			try:

			# with open(f'y_preds_rex_{k}.csv', 'w', newline ='') as f:
				# write = csv.writer(f)

				i1 = np.expand_dims(np.array(Image.open(f"../lfw/image_{k}_A.jpg").resize((112, 112))), 0) 
				i2 = np.expand_dims(np.array(Image.open(f"../lfw/image_{k}_B.jpg").resize((112, 112))), 0) 
				same = np.sign(int(k%600 < 300) - 0.5)
				if same == -1:
					continue

				i1_ = np.array(i1, copy=True)

				min_bce = 1

				for i in range(1000):
					g_1, _ = box.get_grads(i1_, i2)
					i1_ = perturb.perturb_single_image(i1, i1_, same * g_1)
					sim_test = box.cosine_similarity(i1_, i2)

					if sim_test * same < min_bce:
						min_bce = sim_test * same
						Image.fromarray(i1_.squeeze(0).astype(np.uint8)).save(f"../lfw/image_{k}_A_{perturb.pname}_{my_system}_{i}.png")

					print(box.y_preds)
			
			except:
				pass