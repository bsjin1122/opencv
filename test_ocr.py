from PIL import Image
import pytesseract

filename = "C:\study\ocrimg\img5.png"
image = Image.open(filename)
#text = image_to_string(image, lang="kor")
custome_oem_psm_config = r'--oem 1 --psm 4'
result = pytesseract.image_to_string(image, lang='kor.lstm.best', config=custome_oem_psm_config)

print(pytesseract.get_languages(config=''))
# print(pytesseract.image_to_boxes(Image.open(filename)))
# print(pytesseract.image_to_data(Image.open(filename)))


print(result)

#with open("sample.txt", "w") as f:
#    f.write(text)