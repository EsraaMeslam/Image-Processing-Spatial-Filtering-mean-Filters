#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import cv2
import matplotlib.pyplot as plt


# In[3]:


def contraharmonic_mean_filter(img, kernel_sz, Q):

    filtered_image = np.zeros_like(img) 

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            kernal = img[i:i+kernel_sz, j:j+kernel_sz] 

            filtered_image[i, j] = np.sum(kernal**(Q+1))/  np.sum(kernal**Q)

    return filtered_image


# In[6]:


img = cv2.imread('img.png', cv2.IMREAD_GRAYSCALE)


filtered_image = contraharmonic_mean_filter(img, kernel_sz=3, Q=1.5)


# In[7]:


plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')


plt.subplot(1, 2, 2)
plt.imshow(filtered_image, cmap='gray')
plt.title('ContraHarmonic Filtered Image')


plt.show()


# In[ ]:




