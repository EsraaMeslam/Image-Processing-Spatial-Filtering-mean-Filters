#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import matplotlib.pyplot as plt


# In[10]:


def harmonic_mean_filter(img, kernel_sz):

    filtered_image = np.zeros_like(img)  
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            kernal = img[i:i+kernel_sz, j:j+kernel_sz] 
#             filtered_image[i, j] = kernel_sz**2 / np.sum(1.0 / (kernal + 1e-8)) 
            filtered_image[i, j] = (kernel_sz*kernel_sz)/np.sum(1/kernal)

    return filtered_image


# In[11]:


image = cv2.imread('img.png', cv2.IMREAD_GRAYSCALE)

# Apply the harmonic mean filter
filtered_image = harmonic_mean_filter(image, kernel_sz=3)


# In[12]:


plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')


plt.subplot(1, 2, 2)
plt.imshow(filtered_image, cmap='gray')
plt.title('Harmonic Filtered Image')


plt.show()


# In[ ]:




