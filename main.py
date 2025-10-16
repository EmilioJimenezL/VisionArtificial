from preprocessing import preprocess_bike_image
import matplotlib.pyplot as plt
import cv2
import numpy as np

preprocessed_img, edges = preprocess_bike_image("Bicycle annotated/20220815_13_22_16_299_000_Z29NkkNrjjYtQ0CPBMnVk2JJBW13_F_4032_3024.jpg")

# Create an ORB detector
orb = cv2.ORB_create()

# Find keypoints and descriptors
preprocessedEdges, preprocessedDescriptors = orb.detectAndCompute(edges.astype(np.uint8), None)
edgeKeypoints, edgeDescriptors = orb.detectAndCompute(edges.astype(np.uint8), None)

# Draw keypoints on preprocessed image
img_with_keypoints_preprocessed = cv2.drawKeypoints(preprocessed_img, preprocessedEdges, None, color=(0,255,0))
img_with_keypoints_edges = cv2.drawKeypoints(edges, edgeKeypoints, None, color=(0, 255, 0))

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
ax1.imshow(img_with_keypoints_preprocessed)
ax1.set_title('Preprocessed Image')
ax1.axis('off')
ax2.imshow(edges, cmap='gray')
ax2.set_title('Edge Detection')
ax2.axis('off')
ax3.imshow(img_with_keypoints_edges)
ax3.set_title('ORB Features')
ax3.axis('off')
plt.show()
