import cv2

img = cv2.imread('../Computer-Vision-with-Python/DATA/00-puppy.jpg')

while True:
    
    cv2.imshow('Puppy', img)
    # IF we've waited at least 1 ms AND we'be pressed the Esc
    if cv2.waitKey(1) & 0xff ==27:
        break
        
cv2.destroyAllWindows()