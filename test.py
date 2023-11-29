import cv2

img = cv2.imread("backend/python/1.jpg")

cv2.imshow("Anh goc: ", img)
cv2.waitKey(4)

[w, h] = img.shape[:2]
for i in range(w):
    for j in range(h):
        img[i][j] = 255 - img[i][j]
cv2.imshow('Anh am ban:', img)
cv2.waitKey()