from image2perturbed import GlassesPerturbation
from objective import BlackboxWithOneSystem, BlackboxWithMultipleSystems
import numpy as np
# from scipy.optimize._differentialevolution import differential_evolution
from differential_evolution import differential_evolution


from PIL import Image
i1 = np.array(Image.open('Screenshot 2022-06-28 213031.jpg').resize((112, 112)))

perturb = GlassesPerturbation()


box = BlackboxWithMultipleSystems('EfficientNet', 'ReXNet')#),  'GhostNet'

attack_result = differential_evolution(
	func=lambda x: box.cosine_similarity(i1, perturb.perturb_single_image(x, i1)),
	bounds=[(0,255)] * perturb.n_var,
	maxiter=75,
	# popsize=10,
	recombination=1,
	atol=-1,
	callback=lambda x, convergence: box.cosine_similarity(i1, perturb.perturb_single_image(np.expand_dims(x, axis=0), i1)) < 0.65,
	polish=False,
	# init=inits,
	disp=True,
	# workers=2,
)