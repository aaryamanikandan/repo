import cv2


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


cap = cv2.VideoCapture(0)

print("Press 'c' to capture image when face is visible.")
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Cannot access webcam.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Webcam - Face Detection", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('c'):
        if len(faces) > 0:
            cv2.imwrite("captured_face.jpg", frame)
            print("Image saved as 'captured_face.jpg'")
        else:
            print("No face found. Lets try again.")
cap.release()
cv2.destroyAllWindows()
