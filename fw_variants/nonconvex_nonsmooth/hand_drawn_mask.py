import numpy as np
from PIL import Image

# Load the original image
original_image = np.array(Image.open('house.jpg'))

# Load the image with the hand-drawn mask
mask_image = np.array(Image.open('house_with_mask.jpg').convert('L'))

# Threshold the mask image to create a binary mask
threshold = 128
binary_mask = np.where(mask_image < threshold, True, False)

# Apply the binary mask to the original image
masked_image = np.copy(original_image)
masked_image[~binary_mask] = 0

# Display the original image, hand-drawn mask, and masked image
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(original_image)
axs[0].set_title('Original Image')
axs[0].axis('off')
axs[1].imshow(binary_mask, cmap='gray')
axs[1].set_title('Hand-drawn Mask')
axs[1].axis('off')
axs[2].imshow(masked_image)
axs[2].set_title('Masked Image')
axs[2].axis('off')
plt.tight_layout()
plt.show()