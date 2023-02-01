import cv2

# Load the cascade files for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load the cascade files for eye detection
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Initialize the video capture object
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Get the face ROI
        face_roi = gray[y:y+h, x:x+w]

        # Detect eyes in the face ROI
        eyes = eye_cascade.detectMultiScale(face_roi, 1.1, 4)

        # Draw a rectangle around the eyes
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('frame', frame)

    # Check if the user pressed 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
