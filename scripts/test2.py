import cv2
import numpy as np

print("Starting minimal test...")
cv2.namedWindow("test", cv2.WINDOW_NORMAL)
print("Window created, showing image...")
img = np.zeros((200, 200, 3), dtype=np.uint8)
cv2.imshow("test", img)
print("Image shown, waiting for key...")
cv2.waitKey(0)
cv2.destroyAllWindows()
print("Done")