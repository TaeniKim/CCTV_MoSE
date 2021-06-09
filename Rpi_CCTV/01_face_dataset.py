import cv2

'''
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # set video width
cam.set(4, 480)  # set video height
'''

cam = cv2.VideoCapture(r'C:\Users\taeni\Videos\Lee.mp4')
face_detector = cv2.CascadeClassifier('./weight/haarcascade_frontalface_default.xml')

# For each person, enter one numeric face id
face_id = input('\n Enter user id end press <return> ==>  ')

print("\n [INFO] Initializing face capture. Look the camera and wait ...")
# Initialize individual sampling face count
count = 0
interval = 0

while True:
    ret, img = cam.read()
    # img = cv2.flip(img, -1)  # flip video image vertically
    if img is None:
        break

    if interval > 30:
        interval = 0
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            count += 1
            print(count)

            # Save the captured image into the datasets folder
            cv2.imwrite("dataset/" + str(face_id) + ".User." + str(count) + ".jpg", gray[y:y + h, x:x + w])

        cv2.imshow('image', img)

        k = cv2.waitKey(500) & 0xff  # Press 'ESC' for exiting video
        if k == 27:
            break
    else:
        interval += 1

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
