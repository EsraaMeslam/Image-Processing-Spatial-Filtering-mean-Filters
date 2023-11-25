#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import cv2
import matplotlib.pyplot as plt


# In[9]:


def geometric_mean_filter(img, kernel_sz):
    

    filtered_image = np.zeros_like(img)  
    

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            kernal = img[i:i+kernel_sz, j:j+kernel_sz]  
            filtered_image[i, j] = np.exp(np.mean(np.log(kernal)))  

    return filtered_image


# In[10]:


img = cv2.imread('img.png', cv2.IMREAD_GRAYSCALE)


filtered_image = geometric_mean_filter(img, kernel_sz=3)


# In[12]:


plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')


plt.subplot(1, 2, 2)
plt.imshow(filtered_image, cmap='gray')
plt.title('Geometric Filtered Image')


plt.show()

