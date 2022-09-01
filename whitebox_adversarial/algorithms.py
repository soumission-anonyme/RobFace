import numpy as np

# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
	# Collect the element-wise sign of the data gradient
	sign_data_grad = np.sign(data_grad)
	# Create the perturbed image by adjusting each pixel of the input image
	perturbed_image = image + epsilon * sign_data_grad
	# Adding clipping to maintain [0,1] range
	perturbed_image = np.clip(perturbed_image, 0, 255)
	# Return the perturbed image
	return perturbed_image