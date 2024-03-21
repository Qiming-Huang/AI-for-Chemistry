import numpy as np
import matplotlib.pyplot as plt
from PIL import  Image


ori = np.array(Image.open("./ori_img.jpg")) 
mask = np.array(Image.open("./mask_img.jpg"))
rec = np.array(Image.open("./rec_img.jpg"))

ori = ori / np.max(ori)
mask = mask / np.max(mask)
rec = rec / np.max(rec)

plt.subplot(1,3,1)
plt.imshow(ori)
plt.axis("off")
plt.title("original")
plt.subplot(1,3,2)
plt.imshow(mask)
plt.axis("off")
plt.title("masked")
plt.subplot(1,3,3)
plt.imshow(rec)
plt.axis("off")
plt.title("reconstruction")
plt.show()
