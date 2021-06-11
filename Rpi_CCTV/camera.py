import os
import time
import datetime
import threading
import multiprocessing
import cv2
import firebase_admin
import playsound as playsound
from firebase_admin import credentials
from firebase_admin import storage
from firebase_admin import firestore
from uuid import uuid4
import numpy as np
from PIL import Image
from playsound import playsound
from queue import Queue
import socket

que = Queue()

buf = None
buf_lst = None

class Camera(object):
    img = None
    thread = None
    frame = None
    last_access = 0
    bucket = None
    db = None
    p = None

    flag_connection = False
    flag_train_model = False
    flag_update_model = False

    flag_db_polling = False
    flag_play_audio = False
    flag_playing_audio = False
    flag_pre_playing_audio = False
    info_audio_file = ""


    path_dataset = 'dataset'  # visitor training

    def init_firebase(self):
        # firebase

        # PROJECT_ID = "rpi-test-cfcd4"  # 자신의 project id
        # cred = credentials.Certificate("./firebase_certificate/rpi-test-cfcd4-firebase-adminsdk-pvuun-4bd8b1d100.json")
        PROJECT_ID = "mose-cctv"  # 자신의 project id
        cred = credentials.Certificate("./firebase_certificate/mose-cctv-firebase-adminsdk-90ev1-1933d48ad7.json")
        default_app = firebase_admin.initialize_app(cred, {
            'storageBucket': f"{PROJECT_ID}.appspot.com"
        })
        # 버킷은 바이너리 객체의 상위 컨테이너이다. 버킷은 Storage에서 데이터를 보관하는 기본 컨테이너이다.
        self.bucket = storage.bucket()  # 기본 버킷 사용
        # self.db = firestore.client()  # 사용전 함수안에서 호출해야 유효..왜 그런지 모름..

    def initialize(self):
        self.init_firebase()

        self.intervalTimer()
        time.sleep(2.0)  # Timer 간 동기
        self.audioTimer()
        self.queTimer()

        self.fb_updateRefURL(socket.gethostbyname(socket.gethostname()) + ':5000')

        if Camera.thread is None:
            Camera.thread = threading.Thread(target=self._thread)
            Camera.thread.start()

            while self.frame is None:
                time.sleep(0)

    @classmethod
    def intervalTimer(cls):
        if cls.flag_db_polling is True:
            cls.fb_checkControlAudio()
            cls.fb_checkControlTrain()
            if cls.flag_connection == 0:
                cls.flag_connection = 1
            else:
                cls.flag_connection = 0
            cls.fb_set_connection()

        timer = threading.Timer(5, cls.intervalTimer)
        timer.start()

    @classmethod
    def queTimer(cls):
        # print(f'check Queue: {que.empty()}')
        while not que.empty():
            (t, eid, comment, vid, file_name) = que.get()
            # print(f'queTimer: {t} , {eid}, {comment}, {vid} , {file_name}')
            cls.fb_setHistory(t, eid, comment, vid, file_name)
            time.sleep(0.1)

        timer = threading.Timer(3, cls.queTimer)
        timer.start()

    @classmethod
    def audioTimer(cls):
        if cls.flag_play_audio is True:
            f = "./audio/" + cls.info_audio_file
            # File Search
            ret = True
            if os.path.isfile(f) is False:
                print('Local audio file is not exist..!!, find db..!!')
                ret = cls.fileDownload_audio(cls.info_audio_file)

            if ret is True:
                # print(f"Play audio: {f}")
                cls.que_log(6, f"Start play audio.! : {f}")
                cls.flag_playing_audio = True
                cls.p = multiprocessing.Process(target=playsound, args=(f,))
                cls.p.start()
            else:
                cls.flag_playing_audio = False
                print(f'Audio file {cls.info_audio_file} is not exist..!!')
        else:
            if cls.p is not None:
                # print("Stop audio")
                cls.que_log(7, f"Stop play audio.!")
                cls.p.terminate()
                cls.flag_playing_audio = False

        if cls.flag_pre_playing_audio != cls.flag_playing_audio:
            cls.fb_update_audio_state(cls.flag_playing_audio)
        cls.flag_pre_playing_audio = cls.flag_playing_audio

        timer = threading.Timer(6, cls.audioTimer)
        timer.start()

    def set_restart(self):
        self.initialize()

    def get_frame(self):
        Camera.last_access = time.time()
        # self.initialize()
        return self.frame

    @classmethod
    def fb_checkControlAudio(cls):
        db = firestore.client()
        doc_ref = db.collection(u'cctv').document(u'Control_Audio')
        doc = doc_ref.get()
        if doc.exists:
            # print(f'Document data: {doc.to_dict()}')
            cls.flag_play_audio = doc.to_dict()['cmd']
            cls.info_audio_file = doc.to_dict()['file_name']
        else:
            print('[fb_checkControlAudio] No Document..!!')

    @classmethod
    def fb_update_audio_state(cls, state):
        db = firestore.client()
        doc_ref = db.collection(u'cctv').document(u'Control_Audio')
        doc_ref.update({u'state': state})
        print(f'fb_set_connection: Connect --> {state}')

    @classmethod
    def fb_checkControlTrain(cls):
        db = firestore.client()
        doc_ref = db.collection(u'cctv').document(u'Control_Train')
        doc = doc_ref.get()
        if doc.exists:
            # print(f'Document data: {doc.to_dict()}')
            state_train_model = doc.to_dict()['state']
            if state_train_model == 1:  # train start
                cls.flag_train_model = True
                cls.fb_setTrainControl(2)
            elif state_train_model == 3:  # Update Complete
                cls.fb_setTrainControl(0)
        else:
            print('[fb_checkAudioCmd] No Document..!!')

    @classmethod
    def fb_set_connection(cls):
        db = firestore.client()
        doc_ref = db.collection(u'cctv').document(u'Connect')
        doc_ref.set({u'state': cls.flag_connection})
        print(f'fb_set_connection: Connect --> {cls.flag_connection}')

    @classmethod
    def fb_setTrainControl(cls, state):
        db = firestore.client()
        doc_ref = db.collection(u'cctv').document(u'Control_Train')
        doc_ref.set({u'state': state})
        print(f'fb_setTrainControl: state --> {state}')

    @classmethod
    def fb_updateRefURL(cls, str_url):
        db = firestore.client()
        doc_ref = db.collection(u'cctv').document(u'Reference')
        doc_ref.update({u'URL': str_url})
        print(f'fb_updateRefURL: URL --> {str_url}')

    @classmethod
    def fb_setHistory(cls, t, eid, comment, vid=0, file_name=""):
        basename = "I"  # Info
        if eid == 1:  # Motion Start
            basename = "E"  # Event
            fields = {u'ei': basename, 'dt': t, u'id': eid, u'comment': comment}
        elif eid == 2:  # Motion End
            basename = "E"  # Event
            fields = {u'ei': basename, 'dt': t, u'id': eid, u'comment': comment, u'file_name': file_name}
        elif eid == 3:  # Face Recognition
            basename = "E"  # Event
            fields = {u'ei': basename, 'dt': t, u'id': eid, u'comment': comment, u'visitorID': vid, u'file_name': file_name}
        else:
            basename = "I"  # Info
            fields = {u'ei': basename, 'dt': t, u'id': eid, u'comment': comment}

        name = "_".join([basename, t])
        db = firestore.client()
        doc_ref = db.collection(u'CCTV_History').document(name)

        doc_ref.set(fields)
        print(f'fb_setHistory: {name} -> {eid} {comment} {vid}')

    @classmethod
    def fileDownload_audio(cls, file):
        bucket = storage.bucket()  # 기본 버킷 사용
        blob = bucket.blob('audio/' + file)
        # new token and metadata 설정
        new_token = uuid4()
        metadata = {"firebaseStorageDownloadTokens": new_token}  # access token이 필요하다.
        blob.metadata = metadata

        # upload file
        try:
            blob.download_to_filename(filename='./audio/' + file)
            print(f'fileDownload_audio!! - {file}')
            print(blob.public_url)
            return True
        except:
            print('download fail..!!')
            os.remove('./audio/' + file)
            return False

    @classmethod
    def fileDownload_dataset(cls):
        bucket = storage.bucket()  # 기본 버킷 사용
        blob = bucket.blob('user_dataset/')
        # new token and metadata 설정
        new_token = uuid4()
        metadata = {"firebaseStorageDownloadTokens": new_token}  # access token이 필요하다.
        blob.metadata = metadata

        try:
            blob.download_to_filename(filename='./audio/')
            print(blob.public_url)
            return True
        except:
            print('download fail..!!')
            return False

    @classmethod
    def fileUpload_image(cls, file):
        print('fileUpload_image!!')
        bucket = storage.bucket()  # 기본 버킷 사용
        blob = bucket.blob('captureImages/' + file)
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
        bucket = storage.bucket()  # 기본 버킷 사용
        blob = bucket.blob('recordVideos/' + file)
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
        basename = "mose"
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
    def create_capture_image(cls, cap, vid, visitor):
        print('execute_capture!!')
        cls.que_log(3, f'Face Recognition {visitor} detect..!!', vid, visitor)
        # 사진찍기
        # 중복없는 파일명 만들기
        basename = "visitor"
        suffix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + '.jpg'
        filename = "_".join([basename, suffix])

        cv2.imwrite('./captureImages/' + filename, cap, params=[cv2.IMWRITE_JPEG_QUALITY, 100])

        # 사진 파일을 파이어베이스에 업로드 한다.
        cls.fileUpload_image(filename)

    @classmethod
    def task_face_recognition(cls, img, face_cascade, min_w, min_h, recognizer, names):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(min_w), int(min_h)),
        )

        visitors = {}
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            visitor_id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

            if(visitor_id < 0) | (visitor_id >= len(names)):
                print(f"invalid visitor id: {visitor_id}")
                continue

            # Check if confidence is less them 100 ==> "0" is perfect match
            if confidence < 100:
                visitor_name = names[visitor_id]
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                visitor_name = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))

            visitors[visitor_id] = visitor_name
            cv2.putText(img, str(visitor_name), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(img, str(confidence), (x + 5, y + h - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)

        return img, visitors

    @classmethod
    def getImagesAndLabels(cls, path, detector):
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        faceSamples = []
        ids = []
        names = ['Unknown']

        for imagePath in imagePaths:

            PIL_img = Image.open(imagePath).convert('L')  # convert it to grayscale
            img_numpy = np.array(PIL_img, 'uint8')

            visitor_id = int(os.path.split(imagePath)[-1].split(".")[0])
            name = os.path.split(imagePath)[-1].split(".")[1]
            faces = detector.detectMultiScale(img_numpy)

            for (x, y, w, h) in faces:
                faceSamples.append(img_numpy[y:y + h, x:x + w])
                ids.append(visitor_id)
                names.append(name)

        return faceSamples, ids, names

    @classmethod
    def exec_training(cls, face_cascade):
        print('Execute training start!!')
        cls.que_log(100, 'Execute training start!!')

        recognizer = cv2.face.LBPHFaceRecognizer_create()

        faces, ids, names = cls.getImagesAndLabels(cls.path_dataset, face_cascade)
        recognizer.train(faces, np.array(ids))

        recognizer.write('trainer/trainer.yml')

    @classmethod
    def que_log(cls, eid, comment, vid=0, file_name=""):
        t = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f")[:-5]
        que.put((t, eid, comment, vid, file_name))
        # print(f'que_log: {t} {eid} {comment} {vid} {file_name}')

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
        tm_motion_detect2 = time.time()
        filename_video = 'default.avi'
        out = None

        # for face recognition
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read('trainer/trainer.yml')
        cascade_path = "weight/haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
        names = ['unknown', 'Unknown', 'TaeinKim', 'Park', 'HanHyojoo',
                 'Youjaesuk', 'Junjihyun', 'TaeinKim']  # names related to ids: example ==> Marcelo: id=1,  etc
        min_w = 0.1 * camera.get(3)  # Define min window size to be recognized as a face
        min_h = 0.1 * camera.get(4)
        visitors_pre = {}

        print('task_camera: camera read start..!!')
        #cls.que_log(1, 'task_camera: camera read start..!!')
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
                    cls.que_log(1, 'motion detect start..!!')
                    out, filename_video = cls.create_record_video(camera)
                    tm_motion_detect2 = time.time()

            if flag_recode_video & (out is not None):
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
                        cls.que_log(2, 'motion detect End..!!', filename_video)
                        tm_motion_detect = time.time()
                        out.release()  # file save complete
                        cls.fileUpload_video(filename_video)
                        visitors_pre = {}  #
                else:
                    print('error: video file open fail..!!')
                    flag_motion_detect = False
                    flag_recode_video = False

            # cv2.imshow("Video", frame1)
            frame1 = frame2
            frame2 = img

            # face recognition  -------------------------------------------------------------------------------
            if cls.flag_update_model is True:  # new training and weight updated
                print('Update Model..!!')
                cls.que_log(101, 'Update Model..!!')
                recognizer = cv2.face.LBPHFaceRecognizer_create()
                recognizer.read('trainer/trainer.yml')
                cls.flag_update_model = False
                cls.fb_setTrainControl(3)

            img, visitors = cls.task_face_recognition(img,
                                                      face_cascade,
                                                      min_w, min_h,
                                                      recognizer,
                                                      names)
            enble_capture = False
            new_visitor = ""
            if len(visitors) > 0:
                for k, v in visitors.items():
                    if k in visitors_pre:
                        # print('skip')
                        continue
                    else:
                        enble_capture = True
                        new_visitor = v
                        visitors_pre[k] = v
            if enble_capture:
                cls.create_capture_image(img, names.index(new_visitor), new_visitor)

            # Training Model  -------------------------------------------------------------------------------
            if cls.flag_train_model:
                cls.exec_training(face_cascade)
                cls.flag_train_model = False
                cls.flag_update_model = True

            cv2.imshow("Video", img)

            ret_key = cv2.waitKey(int(1000 / 12.))

            if cls.flag_db_polling is False:
                cls.flag_db_polling = True

            # if time.time() - cls.last_access > 10:
            #   break
