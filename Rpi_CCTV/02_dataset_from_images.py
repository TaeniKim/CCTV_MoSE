import glob
import cv2
import numpy as np


def hangulFilePathImageRead(file_path):
    stream = open(file_path.encode("utf-8"), "rb")
    bbytes = bytearray(stream.read())
    numpyArray = np.asarray(bbytes, dtype=np.uint8)

    return cv2.imdecode(numpyArray, cv2.IMREAD_UNCHANGED)

face_detector = cv2.CascadeClassifier('./weight/haarcascade_frontalface_default.xml')

# For each person, enter one numeric face id
face_id = input('\n Enter user id end press <return> ==>  ')

print("\n [INFO] Initializing face capture. Look the camera and wait ...")
# Initialize individual sampling face count
count = 0

fl = glob.glob('C:/Users/taeni/Pictures/Lee/*.*')

for f in fl:
    print(f)
    #img = cv2.imread(f)
    img = hangulFilePathImageRead(f)
    if img is None:
        print("None")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        count += 1
        print(count)

        # Save the captured image into the datasets folder
        cv2.imwrite("dataset/" + str(face_id) + ".User." + str(count) + ".jpg", gray[y:y + h, x:x + w])

    cv2.imshow('image', img)


# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cv2.destroyAllWindows()