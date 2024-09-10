"""
Written by Qiming Huang

This script processes Scanning Tunneling Microscopy (STM) images to extract and analyze specific features,
such as contours and lengths of detected structures. It performs preprocessing, thresholding, and filtering
to remove noise and irrelevant details, followed by skeletonization and morphological operations. The output
includes visual representations of detected structures with their corresponding lengths overlaid on the original images.

Input:
- STM images in PNG format located in the './data/' directory (e.g., '2.PNG', '4.PNG', '6.PNG').

Output:
- Displays processed images with detected contours highlighted and annotated with the calculated lengths.

Processing Steps:
1. Read and preprocess each image with gamma correction and color space conversion.
2. Apply Gaussian blur and Otsu thresholding to create a binary image for region isolation.
3. Identify and filter connected components to remove small, irrelevant blobs.
4. Use morphological operations and skeletonization to refine structure detection.
5. Detect contours and calculate the lengths of detected structures.
6. Visualize detected contours on the original image with random colors and length annotations.
7. Display the final annotated images using Matplotlib.

Dependencies:
- Python libraries: OpenCV (cv2), Matplotlib, NumPy, scikit-image, tqdm.

Usage:
- Place the STM images in the './data/' directory.
- Run the script in a Python environment with the required dependencies installed.
- The processed images will be displayed sequentially.

"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import morphology

def RGB_to_Hex(tmp):
    rgb = tmp.split(',')
    strs = '#'
    for i in rgb:
        num = int(i)
        strs += str(hex(num))[-2:].replace('x','0').upper()
        
    return strs


for img_idx in [2,4,6]:
    img = cv2.imread(f"./data/{img_idx}.PNG")
    img = np.power(img/float(np.max(img)), 1.5)
    img = np.uint8(img * 255)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(img_gray,(7,7),10)
    # remove small bright spots
    # light_remove = np.uint8(np.where(blur > 220, 0, blur))
    
    ret1, th1 = cv2.threshold(blur, 155, 230, cv2.THRESH_OTSU)
    
    nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(th1)
    sizes = stats[:, -1]
    sizes = sizes[1:]
    nb_blobs -= 1
    min_size = 500  
    # output image with only the kept components
    im_result = np.zeros_like(im_with_separated_blobs)
    # for every component in the image, keep it only if it's above min_size
    for blob in range(nb_blobs):
        if sizes[blob] >= min_size:
            # see description of im_with_separated_blobs above
            im_result[im_with_separated_blobs == blob + 1] = 255    

    kernel = np.ones((5,5),np.uint8)
    # black hat
    black_hat = cv2.morphologyEx(np.uint8(im_result), cv2.MORPH_BLACKHAT, np.ones((2,2),np.uint8))        
    skeleton = np.uint8(morphology.skeletonize(black_hat))
    skeleton_dilate = cv2.dilate(skeleton, np.ones((3,3),np.uint8), iterations = 1)
    
    # !!
    a = cv2.dilate(skeleton, np.ones((2,2),np.uint8), iterations=5)
    nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(a)
    sizes = stats[:, -1]
    sizes = sizes[1:]
    nb_blobs -= 1
    min_size = 200  
    # output image with only the kept components
    im_result = np.zeros_like(im_with_separated_blobs)
    # for every component in the image, keep it only if it's above min_size
    for blob in range(nb_blobs):
        if sizes[blob] >= min_size:
            # see description of im_with_separated_blobs above
            im_result[im_with_separated_blobs == blob + 1] = 255         
    
    # # black hat
    # black_hat = cv2.morphologyEx(th1, cv2.MORPH_BLACKHAT, np.ones((2,2),np.uint8))
     

    skeleton_final = np.uint8(morphology.skeletonize(im_result))             
    s_contours, _ = cv2.findContours(skeleton_final, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
     
    colors = []
    for _ in range(len(s_contours)):
        colors.append(
            [np.uint8(np.floor(np.random.rand() * 255)), np.uint8(np.floor(np.random.rand() * 255)), np.uint8(np.floor(np.random.rand() * 255))]
        )   
    colors = np.array(colors)         

    lists = []
    lengths = []
    areas = []
    plt.figure(figsize=(16,16),dpi=100)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    # plt.imshow(img)
    plt.axis("off")
    for i in range(len(s_contours)): 
        length = cv2.arcLength(s_contours[i], False) / 2
        rect = cv2.minAreaRect(s_contours[i])
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        area = cv2.contourArea(box)
        areas.append(area)
            
        if length > 30 or True:
            mask1 = cv2.drawContours(img, s_contours, i, [int(j) for j in colors[i]], 4)
            plt.text(
                s_contours[i][0][0][0], s_contours[i][0][0][1], f"{int(length)}",
                bbox=dict(boxstyle='round', facecolor=RGB_to_Hex(str(colors[i][0])+','+str(colors[i][1])+','+str(colors[i][2])), edgecolor='gray', alpha=0.7),
                fontsize=25
            )    
            lengths.append(int(length))
    plt.imshow(mask1)
    plt.savefig(f"./res/{img_idx}.PNG")
    plt.show()
