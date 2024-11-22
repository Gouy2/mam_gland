import os
import numpy as np
import nibabel as nib
import nibabel.orientations as nio
import matplotlib.pyplot as plt
import pydicom as dicom
from PIL import Image


# dir = "./chaoyang_huigu/BAILIANDI RCC"
dir = "./chaoyang_huigu/BAILIANDI RMLO"

img_nib = nib.load(os.path.join(dir, "1.nii.gz"))
img_data = img_nib.get_fdata() 
nii_image = np.squeeze(img_data)

img_data = np.transpose(nii_image, (1, 0))
print('img shape: ', img_nib.shape)
print('data shape: ', img_data.shape)
print('data type: ', type(img_data))

plt.imshow(img_data, cmap="gray")
plt.title("NII Image Slice Visualization")
plt.axis("off")
plt.show()



# print("img_data: ", img_data[1000])

# # img_dcm = dicom.dcmread(os.path.join(dir, "ser97311img00002.dcm"))
# img_dcm = dicom.dcmread(os.path.join(dir, "ser97311img00001.dcm"))

# # print(img_dcm.pixel_array)
# print(img_dcm.pixel_array.shape)

# data_img = Image.fromarray(img_dcm.pixel_array)
# data_img_rotated = data_img.rotate(angle=45,resample=Image.BICUBIC)


# plt.imshow(img_dcm.pixel_array,cmap=plt.cm.bone)
# plt.show()
