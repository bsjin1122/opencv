import cv2
import requests
import numpy as np
import pytesseract
from pytesseract import Output
import matplotlib.pyplot as plt

# url = 'https://user-images.githubusercontent.com/69428232/148330274-237d9b23-4a79-4416-8ef1-bb7b2b52edc4.jpg'
url = 'https://static.news.zumst.com/images/2/2019/10/09/28d9530a7ef54cc5bdd51dcf810963c8.jpg'
# 파일 열기 및 바이트 데이터 열기 
# url = r'C:\study\ocrimg\img1.png'
# with open(url, 'rb') as f:
#     byte_data = f.read()

# image_nparray = np.asarray(bytearray(byte_data), dtype=np.uint8)
image_nparray = np.asarray(bytearray(requests.get(url).content), dtype=np.uint8)
org_image = cv2.imdecode(image_nparray, cv2.IMREAD_COLOR)
rgb_image = cv2.cvtColor(org_image, cv2.COLOR_BGR2RGB)

# use Tesseract to OCR the image 
# text = pytesseract.image_to_string(rgb_image, lang='eng.lstm.best')
text = pytesseract.image_to_string(rgb_image, lang='kor.lstm.best')
print(text)
text = text.split('\n')[0:-1]
print(text)