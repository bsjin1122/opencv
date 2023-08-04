import cv2
import requests
import numpy as np
import pytesseract
from pytesseract import Output
import matplotlib.pyplot as plt

tessdata_dir_config = 'C:/3rdparty/Tesseract-OCR/tessdata/script/Hangul.traineddata'

url = 'https://user-images.githubusercontent.com/69428232/148318703-ef6bd43f-ec4f-42f5-a336-3b584a662982.jpg'
# url = 'http://61.74.211.16/api/v5/image/1dfd2e0b-b7cf-4167-895a-657136aac141'

# 파일 열기 및 바이트 데이터 열기 
# url = r'C:\study\ocrimg\img3.png'
# with open(url, 'rb') as f:
#     byte_data = f.read()

# image_nparray = np.asarray(bytearray(byte_data), dtype=np.uint8)
image_nparray = np.asarray(bytearray(requests.get(url).content), dtype=np.uint8)
org_image = cv2.imdecode(image_nparray, cv2.IMREAD_COLOR)
rgb_image = cv2.cvtColor(org_image, cv2.COLOR_BGR2RGB)

# use Tesseract to OCR the image 
text = pytesseract.image_to_string(rgb_image, lang='kor+eng')
print(text)
text = text.split('\n')[0:-1]
print(text)