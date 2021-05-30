import time
import datetime
import io
import threading
import cv2
import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage
from uuid import uuid4
import schedule


class Camera(object):
    img = None
    thread = None
    frame = None
    last_access = 0
    bucket = None

    def initialize(self):
        # firebase
        PROJECT_ID = "rpi-test-cfcd4"  # 자신의 project id

        cred = credentials.Certificate("./firebase_certificate/rpi-test-cfcd4-firebase-adminsdk-pvuun-4bd8b1d100.json")
        default_app = firebase_admin.initialize_app(cred, {
            # gs://smart-mirror-cf119.appspot.com
            'storageBucket': f"{PROJECT_ID}.appspot.com"
        })
        # 버킷은 바이너리 객체의 상위 컨테이너이다. 버킷은 Storage에서 데이터를 보관하는 기본 컨테이너이다.
        self.bucket = storage.bucket()  # 기본 버킷 사용

        if Camera.thread is None:
            Camera.thread = threading.Thread(target=self._thread)
            Camera.thread.start()

            while self.frame is None:
                time.sleep(0)

    def set_restart(self):
        self.initialize()

    def get_frame(self):
        Camera.last_access = time.time()
        # self.initialize()
        return self.frame

    def get_frame2(self):
        frame = cv2.imencode('.jpg', self.img)[1].tobytes()
        Camera.last_access = time.time()
        return frame

    def get_camera(self):
        return self.img

    def task_camera(self):
        print('task_camera: start')
        camera = cv2.VideoCapture(0)

        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')

        print('task_camera: camera read start..!!')
        while True:
            # read current frame
            ret, img = camera.read()
            self.img = cv2.flip(img, 1)  # 0:ud, 1:lr, 2: udlr

            cv2.waitKey(int(1000 / 12.))
            # if time.time() - cls.last_access > 10:
            #   break

    @classmethod
    def fileUpload_image(cls, file):
        print('fileUpload_image!!')
        blob = cls.bucket.blob('captureImages/' + file)
        # new token and metadata 설정
        new_token = uuid4()
        metadata = {"firebaseStorageDownloadTokens": new_token}  # access token이 필요하다.
        blob.metadata = metadata

        # upload file
        blob.upload_from_filename(filename='./captureImages/' + file, content_type='image/jpeg')
        print(blob.public_url)

    @classmethod
    def fileUpload_video(cls, file):
        print('fileUpload_video!!')
        blob = cls.bucket.blob('recordVideos/' + file)
        # new token and metadata 설정
        new_token = uuid4()
        metadata = {"firebaseStorageDownloadTokens": new_token}  # access token이 필요하다.
        blob.metadata = metadata

        # upload file
        blob.upload_from_filename(filename='./recordVideos/' + file, content_type='video/x-msvideo')
        print(blob.public_url)

    @classmethod
    def create_record_video(cls, cap):
        # 웹캠의 속성 값을 받아오기
        # 정수 형태로 변환하기 위해 round
        w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # fps = cap.get(cv2.CAP_PROP_FPS)  # 카메라에 따라 값이 정상적, 비정상적
        fps = 12.0  # 카메라에 따라 값이 정상적, 비정상적

        # fourcc 값 받아오기, *는 문자를 풀어쓰는 방식, *'DIVX' == 'D', 'I', 'V', 'X'
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')

        # 웹캠으로 찰영한 영상을 저장하기
        # cv2.VideoWriter 객체 생성, 기존에 받아온 속성값 입력
        basename = "smr"
        suffix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + '.avi'
        filename = "_".join([basename, suffix])
        out = cv2.VideoWriter(r'./recordVideos/' + filename, fourcc, fps, (w, h))

        print('[create_record_video] record video start')
        return out, filename

    @classmethod
    def task_detect_motion(cls, frame1, frame2):
        is_detect = False
        diff = cv2.absdiff(frame1, frame2)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(diff_gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            if cv2.contourArea(contour) < 900:
                continue
            is_detect = True
            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame1, "Status: {}".format('Movement'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 0, 0), 3)

        cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2)

        return frame1, is_detect

    @classmethod
    def create_capture_image(cls, cap):
        print('execute_capture!!')
        # 사진찍기
        # 중복없는 파일명 만들기
        basename = "smr"
        suffix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + '.jpg'
        filename = "_".join([basename, suffix])

        cv2.imwrite('./captureImages/' + filename, cap, params=[cv2.IMWRITE_JPEG_QUALITY, 100])

        # 사진 파일을 파이어베이스에 업로드 한다.
        #cls.fileUpload_image(filename)

    @classmethod
    def task_face_recognition(cls, img, face_cascade, min_w, min_h, train_change, recognizer, names):
        user_id = 0  # iniciate id counter
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(min_w), int(min_h)),
        )

        for (x, y, w, h) in faces:
            if train_change is True:  # new training and weight updated
                recognizer.read('trainer/trainer.yml')

            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            user_id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

            # Check if confidence is less them 100 ==> "0" is perfect match
            if confidence < 100:
                user_id = names[user_id]
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                user_id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))

            cv2.putText(img, str(user_id), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(img, str(confidence), (x + 5, y + h - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)

        return img, user_id

    @classmethod
    def _thread(cls):
        print('task_camera: start')
        camera = cv2.VideoCapture(0)
        ret, frame1 = camera.read()
        ret, frame2 = camera.read()

        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')

        # for detect motion
        flag_motion_detect = False
        flag_recode_video = False
        tm_motion_detect = time.time()
        filename_video = 'default.avi'

        # for face recognition
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read('trainer/trainer.yml')
        cascade_path = "./weight/haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path);
        names = ['None', 'Marcelo', 'Paula', 'Ilza', 'Z', 'W']  # names related to ids: example ==> Marcelo: id=1,  etc
        min_w = 0.1 * camera.get(3)  # Define min window size to be recognized as a face
        min_h = 0.1 * camera.get(4)
        train_change = False

        print('task_camera: camera read start..!!')
        while True:
            # read current frame
            ret, img = camera.read()
            cls.img = cv2.flip(img, 1)  # 0:ud, 1:lr, 2: udlr

            cls.frame = cv2.imencode('.jpg', img)[1].tobytes()  # for stream

            # motion detection  -------------------------------------------------------------------------------
            frame1, is_detect_motion = cls.task_detect_motion(frame1, frame2)
            if is_detect_motion:
                tm_motion_detect = time.time()
                if not flag_motion_detect:
                    flag_motion_detect = True
                    flag_recode_video = True
                    print(f'{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")} event: motion detect start..!!')
                    out, filename_video = cls.create_record_video(camera)
                    tm_motion_detect2 = time.time()

            if flag_recode_video:
                if out.isOpened():
                    if ret:
                        out.write(img)  # 영상 데이터만 저장. 소리는 X
                    else:
                        print('error: image cature fail..!!')

                    if (time.time() - tm_motion_detect) > 5.0:
                        print(time.time() - tm_motion_detect2)
                        flag_motion_detect = False
                        flag_recode_video = False
                        print(f'{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")} event: motion detect End..!!')
                        tm_motion_detect = time.time()
                        out.release()  # file save complete
                        #cls.fileUpload_video(filename_video)
                else:
                    print('error: video file open fail..!!')
                    flag_motion_detect = False
                    flag_recode_video = False

            frame1 = frame2
            frame2 = img

            # face recognition  -------------------------------------------------------------------------------
            img, user_id = cls.task_face_recognition(img,
                                                     face_cascade,
                                                     min_w, min_h,
                                                     train_change,
                                                     recognizer,
                                                     names)

            cv2.imshow("Video", img)

            cv2.waitKey(int(1000 / 12.))
            # if time.time() - cls.last_access > 10:
            #   break
