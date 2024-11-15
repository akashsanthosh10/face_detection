from retinaface import RetinaFace
import cv2
import matplotlib.pyplot as plt

path = "D:\\COLLEGE\\Mini Project\\Python 3.10\\Face Detection\\test.jpg"
img = cv2.imread(path)
obj = RetinaFace.detect_faces(path)

for key in obj.keys():
    identity = obj[key]
    facial_area = identity["facial_area"]
    cv2.rectangle(img, (facial_area[2], facial_area[3]), (facial_area[0], facial_area[1]), (255, 0, 0), 10)

plt.imshow(img)
plt.show()