import pytesseract
import cv2 
#import matplotlib.pyplot as plt

path = r'C:\Users\user\Desktop\tesseract\image\test01.jpg'
image = cv2.imread(path)
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# use Tesseract to OCR the image 
text = pytesseract.image_to_string(rgb_image, lang='kor+eng')
print(text)