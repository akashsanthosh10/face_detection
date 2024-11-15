from retinaface import RetinaFace
import cv2
import json
# initialize the camera
cap = cv2.VideoCapture(0)

# Set desired framerate
cap.set(cv2.CAP_PROP_FPS, 30)

while True:
    # read a frame from the camera
    ret, frame = cap.read()
    height, width = frame.shape[:2]
        # Downsample factor (e.g., reducing size to 50%)
    
    # convert the frame to RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # detect faces in the frame
    faces = RetinaFace.detect_faces(frame_rgb)

    # draw bounding boxes around detected faces
    for face in faces:
        identity = faces[face]
        facial_area = identity["facial_area"]
        cv2.rectangle(frame, (facial_area[2], facial_area[3]), (facial_area[0], facial_area[1]), (255, 0, 0), 10)

    # display the frame with detected faces
    cv2.imshow('RetinaFace Real-time Demo', frame)

    # exit the loop if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release the camera and destroy all windows
cap.release()
cv2.destroyAllWindows()