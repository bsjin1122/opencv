import cv2
import requests
import numpy as np
import pytesseract
from pytesseract import Output
import matplotlib.pyplot as plt

url = 'https://user-images.githubusercontent.com/69428232/148318703-ef6bd43f-ec4f-42f5-a336-3b584a662982.jpg'
# url = 'https://seoulforest.or.kr/wp-content/uploads/2017/09/IMG_0404-1024x683.jpg'
# url = 'http://61.74.211.16/api/v5/image/1dfd2e0b-b7cf-4167-895a-657136aac141'

image_nparray = np.asarray(bytearray(requests.get(url).content), dtype=np.uint8)
org_image = cv2.imdecode(image_nparray, cv2.IMREAD_COLOR)
rgb_image = cv2.cvtColor(org_image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(8, 5))

plt.subplot(1, 1, 1)
plt.imshow(rgb_image)
plt.title('RGB Image')
plt.xticks([]), plt.yticks([])

plt.show()