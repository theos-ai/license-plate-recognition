import utils
import cv2
import time
import os


URL = 'https://inf-fb95ed88-6696-4816-b355-b64c07c85e09-no4xvrhsfq-uc.a.run.app/detect'
OCR_MODEL = 'large'
OCR_CLASSES = 'license-plate'
FOLDER_PATH = 'license-plates'

start = time.time()
wait_time = 1
frame_to_detect = None

if not os.path.exists(FOLDER_PATH):
    os.makedirs(FOLDER_PATH)

camera = cv2.VideoCapture(1)

if camera.isOpened():
    recording, frame = camera.read()
else:
    recording = False

while recording:
    recording, frame = camera.read()
    elapsed = time.time() - start

    if elapsed >= wait_time:
        start = time.time()
        image_bytes = cv2.imencode('.jpg', frame)[1].tobytes()
        now = utils.get_time()
        try:
          detection_start = time.time()
          detections = utils.detect(image_bytes, url=URL, ocr_model=OCR_MODEL, ocr_classes=OCR_CLASSES)
          wait_time = (time.time() - detection_start) + 2

          if len(detections) > 0:
            detected_frame = utils.draw(frame, detections)
            license_plates = []
            
            for detection in detections:
              if detection['class'] == OCR_CLASSES and detection['text']:
                license_plates.append(detection['text'].upper())
            
            if len(license_plates) > 0:
              utils.save_frame(detected_frame, os.path.join(FOLDER_PATH, now + '.jpg'))
              utils.save_json(detections, os.path.join(FOLDER_PATH, now + '.json'))
              print(f'[{now}] [+] License plates saved:', ', '.join(license_plates))

            detections = []
        except Exception as error:
          pass

camera.release()