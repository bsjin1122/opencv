import cv2
import requests
import numpy as np
import pytesseract
from pytesseract import Output
import matplotlib.pyplot as plt

image_path  = './image/test03.jpg'
image = cv2.imread(image_path)
print(type(image))
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# use Tesseract to OCR the image 
text = pytesseract.image_to_string(rgb_image, lang='kor')
print(text)