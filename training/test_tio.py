import sys
import numpy as np
import pandas as pd
import torch
import torchio as tio
import cv2
import pydicom
import matplotlib.pyplot as plt

file = "/Users/isaiah/datasets/kaggle_mammograph/train_images/10006/1864590858.dcm"
df = pd.DataFrame({"foo": [1., 2., 10.], "bar": [13., 111., 22.], "laterality": [1., 0., 1.]})
md = df.loc[0, :].copy(deep=True)
swap_axes = {"0.0": 1., "1.0": 0.}
ds = pydicom.dcmread(file)
im = ds.pixel_array

flip_list = [1, 1, 1, 1]
flip_axis = np.random.choice(flip_list, 1).tolist()
if flip_axis == 1:
    md.loc["laterality"] = swap_axes[str(md.loc["laterality"])]

def testfn(im):
    return im

if flip_axis == None:
    flipped = np.zeros_like(im)
elif flip_axis == 0:
    flipped = im[::-1, :]
else:
    flipped = im[:, ::-1]

im_tensor = torch.from_numpy(im.astype(np.int16))
im_tensor.unsqueeze_(0)
im_tensor.unsqueeze_(3)
flipped_tensor = torch.from_numpy(flipped.astype(np.int16))
flipped_tensor.unsqueeze_(0)
flipped_tensor.unsqueeze_(3)

vertical_flip = torch.flip(im_tensor, [flip_axis])
horizontal_flip = torch.flip(im_tensor, [flip_axis])

# Define the transforms
# transforms = tio.Compose([
    # tio.RescaleIntensity(out_min_max=(0, 1)),
    # tio.RandomAffine(scales=(0.9, 1.1), degrees=10),
# ])

# Apply the transforms to a batch of images
# im_trans = transforms(im_tensor)
# flipped_trans = transforms(flipped_tensor)

# Print the list of transformation names
fig, axs = plt.subplots(1, 3, figsize=(14, 6))
axs[0].set_title("Image")
axs[0].imshow(im_tensor.squeeze(), cmap="bone")
axs[1].set_title("Vertical Flipped Image")
axs[1].imshow(vertical_flip.squeeze(), cmap="bone")
axs[2].set_title("Horizontal Flipped Image")
axs[2].imshow(horizontal_flip.squeeze(), cmap="bone")
plt.show()
