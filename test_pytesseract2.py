import cv2
import pytesseract
try: 
    from PIL import Image
except ImportError:
    import Image

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR'
img = Image.open(r'C:\study\ocrimg\img1.png')
print(pytesseract.image_to_string(img))

# imgPath = 'C:\study\ocrimg\imp1.png'

# config = ('-l kor.lstm.best --oem 1 --psm 4')

# ocr_result = pytesseract.image_to_data(image=img, lang=traineddata, config='--oem ' + str(oem) + ' --psm ' + str(psm))

# print(ocr_result)
