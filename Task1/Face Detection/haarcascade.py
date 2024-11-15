import cv2

# Load the Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('D:\COLLEGE\Mini Project\Python 3.10\Face Detection\haarcascade_frontalface_default.xml')

# Open the webcam video capture
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert frame to grayscale (Haar cascades work better on grayscale)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Draw bounding boxes around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display the resulting frame with bounding boxes
    cv2.imshow('Face Detection', frame)

    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
