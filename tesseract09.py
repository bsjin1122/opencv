def plt_imshow(title='image', img=None, figsize=(8 ,5)):
    plt.figure(figsize=figsize)
 
    if type(img) == list:
        if type(title) == list:
            titles = title
        else:
            titles = []
 
            for i in range(len(img)):
                titles.append(title)
 
        for i in range(len(img)):
            if len(img[i].shape) <= 2:
                rgbImg = cv2.cvtColor(img[i], cv2.COLOR_GRAY2RGB)
            else:
                rgbImg = cv2.cvtColor(img[i], cv2.COLOR_BGR2RGB)
 
            plt.subplot(1, len(img), i + 1), plt.imshow(rgbImg)
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
 
        plt.show()
    else:
        if len(img.shape) < 3:
            rgbImg = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 
        plt.imshow(rgbImg)
        plt.title(title)
        plt.xticks([]), plt.yticks([])
        plt.show()

def preprocess_image(image, width, ksize=(5,5), min_threshold=75, max_threshold=200):
  image_list_title = []
  image_list = []
 
  org_image = image.copy()
  image = imutils.resize(image, width=width)
  ratio = org_image.shape[1] / float(image.shape[1])
 
  # 이미지를 grayscale로 변환하고 blur를 적용
  # 모서리를 찾기위한 이미지 연산
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  blurred = cv2.GaussianBlur(gray, ksize, 0)
  edged = cv2.Canny(blurred, min_threshold, max_threshold)
 
  image_list_title = ['gray', 'blurred', 'edged']
  image_list = [gray, blurred, edged]
 
  # contours를 찾아 크기순으로 정렬
  cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = imutils.grab_contours(cnts)
  cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
 
  findCnt = None
 
  # 정렬된 contours를 반복문으로 수행하며 4개의 꼭지점을 갖는 도형을 검출
  for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
 
    # contours가 크기순으로 정렬되어 있기때문에 제일 첫번째 사각형을 영역으로 판단하고 break
    if len(approx) == 4:
      findCnt = approx
      break
 
  # 만약 추출한 윤곽이 없을 경우 오류
  if findCnt is None:
    raise Exception(("Could not find outline."))
 
  output = image.copy()
  cv2.drawContours(output, [findCnt], -1, (0, 255, 0), 2)
  
  image_list_title.append("Outline")
  image_list.append(output)
 
  # 원본 이미지에 찾은 윤곽을 기준으로 이미지를 보정
  transform_image = four_point_transform(org_image, findCnt.reshape(4, 2) * ratio)
 
  plt_imshow(image_list_title, image_list)
  plt_imshow("Transform", transform_image)
 
  return transform_image

def segment_image(image):
    # 그레이스케일로 변환
    gray = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2GRAY)
    (H, W) = gray.shape
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 20))
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 21))
    # 노이즈를 줄이기 위해 가우시안블러 적용 
    gray = cv2.GaussianBlur(gray, (11, 11), 0)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
    # 흐릿한 Grayscale 이미지에 blackhat 모노폴리 연산을 적용 
    # blackhat연산은 밝은 배경(영수증의 배경)에서 어두운 영역(텍스트)을 드러내기 위해 사용
    grad = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    grad = np.absolute(grad)
    (minVal, maxVal) = (np.min(grad), np.max(grad))
    grad = (grad - minVal) / (maxVal - minVal)
    grad = (grad * 255).astype("uint8")

    # 닫힘 연산을 통해 끊어져보이는 객체를 연결하여 Grouping 
    grad = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, rectKernel)
    thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    close_thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
    close_thresh = cv2.erode(close_thresh, None, iterations=2)
    
    plt_imshow(["Original", "Blackhat", "Gradient", "Rect Close", "Square Close"], [transformed_image, blackhat, grad, thresh, close_thresh], figsize=(16, 10))    
    return close_thresh

def detect_image(transformed_image, segmented_image):
    cnts = cv2.findContours(segmented_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sort_contours(cnts, method="top-to-bottom")[0]
    
    roi_list = []
    roi_title_list = []
    
    margin = 20
    image_grouping = transformed_image.copy()
    
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w // float(h)    
        if ar > 3.0 and ar < 6.5 and (w/2) < x:
            color = (0, 255, 0)
            roi = transformed_image[y - margin:y + h + margin, x - margin:x + w + margin]
            roi_list.append(roi)
            roi_title_list.append("Roi_{}".format(len(roi_list)))
        else:
            color = (0, 0, 255)        
        cv2.rectangle(image_grouping, (x - margin, y - margin), (x + w + margin, y + h + margin), color, 2)
        cv2.putText(image_grouping, "".join(str(ar)), (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)    
    # 식별된 항목을 모두 표시
    plt_imshow(["Grouping Image"], [image_grouping], figsize=(16, 10))    
    # 테이블의 값들만 표시
    plt_imshow(roi_title_list, roi_list, figsize=(16, 10))
    
    for roi in roi_list:
        gray_roi= cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        threshold_roi = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        text = pytesseract.image_to_string(threshold_roi)
        # 정규식으로 가격만 추출
        # price = re.findall(r'(?:NP )([0-9\.\-+_]+\.[0-9\.\-+_]+)', text)
        price = re.findall(r'\d+(?:\.\d+)?', text)
        print(price)        
    
    return image_grouping

def sort_contours(cnts, method="left-to-right"):
	# initialize the reverse flag and sort index
	reverse = False
	i = 0
	# handle if we need to sort in reverse
	if method == "right-to-left" or method == "bottom-to-top":
		reverse = True
	# handle if we are sorting against the y-coordinate rather than
	# the x-coordinate of the bounding box
	if method == "top-to-bottom" or method == "bottom-to-top":
		i = 1
	# construct the list of bounding boxes and sort them from top to
	# bottom
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
		key=lambda b:b[1][i], reverse=reverse))
	# return the list of sorted contours and bounding boxes
	return (cnts, boundingBoxes)    

from imutils.perspective import four_point_transform
import matplotlib.pyplot as plt
import pytesseract
import imutils
import cv2
import re
import requests
import numpy as np

image_path  = './image/test02.jpg'
org_image = cv2.imread(image_path)
plt_imshow("orignal image", org_image)
transformed_image = preprocess_image(org_image, width=200, ksize=(5, 5), min_threshold=20, max_threshold=100)

segmented_image = segment_image(transformed_image)
plt_imshow(["Square Close"], [segmented_image], figsize=(16, 10))

detected_image = detect_image(transformed_image, segmented_image)

options = "--psm 4"
#lang='kor_lstm_best+eng_lstm_best'
lang='eng_lstm_best'
text = pytesseract.image_to_string(cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB), lang=lang, config=options)
# OCR결과 출력
print("[INFO] OCR결과:")
print("==================")
print(text)
print("\n")