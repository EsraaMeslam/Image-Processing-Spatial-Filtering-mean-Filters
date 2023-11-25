#!/usr/bin/env python
# coding: utf-8

# In[18]:


import cv2
import numpy as np
import matplotlib.pyplot as plt


# In[19]:


def arithmetic_mean_filter(img, kernel_sz):
  

    output_image = np.zeros_like(img)

    
   
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):

            neighborhood = img[i:i+kernel_sz, j:j+kernel_sz]

            arithmetic=np.sum(neighborhood)/(kernel_sz*kernel_sz)
  


            output_image[i, j] = arithmetic
    
    return output_image


# In[20]:


img = cv2.imread('img.png', cv2.IMREAD_GRAYSCALE)

filtered_image = arithmetic_mean_filter(img, kernel_sz=3)


# In[21]:


# Display the original and filtered images
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')


plt.subplot(1, 2, 2)
plt.imshow(filtered_image, cmap='gray')
plt.title('Arithemtic Filtered Image')


plt.show()


# In[ ]:




