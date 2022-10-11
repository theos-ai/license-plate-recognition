import utils
import cv2
import time
import os


URL = 'https://inf-fb95ed88-6696-4816-b355-b64c07c85e09-no4xvrhsfq-uc.a.run.app/detect'
OCR_MODEL = 'large'
OCR_CLASS = 'license-plate'
FOLDER_PATH = 'license-plates'

seconds_to_wait = 2

if not os.path.exists(FOLDER_PATH):
    os.makedirs(FOLDER_PATH)

camera = cv2.VideoCapture(0)

if camera.isOpened():
    camera_open, frame = camera.read()
else:
    camera_open = False

start = time.time()

while camera_open:
    recording, frame = camera.read()
    elapsed = time.time() - start

    if elapsed >= seconds_to_wait:
        start = time.time()
        image_bytes = cv2.imencode('.jpg', frame)[1].tobytes()
        now = utils.get_time()

        try:
          detections = utils.detect(image_bytes, url=URL, ocr_model=OCR_MODEL, ocr_classes=OCR_CLASS)

          if len(detections) > 0:
            detected_frame = utils.draw(frame, detections)
            license_plates = []
            
            for detection in detections:
              if detection['class'] == OCR_CLASS and detection['text']:
                license_plates.append(detection['text'].upper())
            
            if len(license_plates) > 0:
              utils.save_frame(detected_frame, os.path.join(FOLDER_PATH, now + '.jpg'))
              utils.save_json(detections, os.path.join(FOLDER_PATH, now + '.json'))
              print(f'[{now}] [+] License plates saved:', ', '.join(license_plates))
        except Exception as error:
          pass

camera.release()