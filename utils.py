from datetime import datetime
import numpy as np
import requests
import json
import time
import cv2


def detect(image_bytes, url, ocr_model, ocr_classes, fallback_url=None, conf_thres=0.25, iou_thres=0.45, retries=10, delay=0):
    response = requests.post(url, data={'conf_thres':conf_thres, 'iou_thres':iou_thres, 'ocr_model':ocr_model, 'ocr_classes':ocr_classes}, files={'image':image_bytes})
    if response.status_code in [200, 500]:
        data = response.json()
        if 'error' in data:
            print('[!]', data['message'])
        else:
            return data
    elif response.status_code == 403:
        print('[!] you reached your monthly requests limit. Upgrade your plan to unlock unlimited requests.')
    elif retries > 0:
        if delay > 0:
            time.sleep(delay)
        return detect(image_bytes, url=fallback_url if fallback_url else url, retries=retries-1, delay=2)
    return []

def draw_border(image, top_left_point, bottom_right_point, color, thickness, radius=5, length=5):
    x1, y1 = top_left_point
    x2, y2 = bottom_right_point
    res_scale = (image.shape[0] + image.shape[1])/2000
    radius = int(radius * res_scale)
 
    # Top left
    cv2.line(image, (x1 + radius, y1), (x2 - radius - length, y1), color, thickness, cv2.LINE_AA)
    cv2.line(image, (x1, y1 + radius), (x1, y2 - radius - length), color, thickness, cv2.LINE_AA)
    cv2.ellipse(image, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness, cv2.LINE_AA)
 
    # Top right
    cv2.line(image, (x2 - radius, y1), (x1 + radius + length, y1), color, thickness, cv2.LINE_AA)
    cv2.line(image, (x2, y1 + radius), (x2, y2 - radius - length), color, thickness, cv2.LINE_AA)
    cv2.ellipse(image, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness, cv2.LINE_AA)
 
    # Bottom left
    cv2.line(image, (x1 + radius, y2), (x2 - radius - length, y2), color, thickness, cv2.LINE_AA)
    cv2.line(image, (x1, y2 - radius), (x1, y1 + radius + length), color, thickness, cv2.LINE_AA)
    cv2.ellipse(image, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness, cv2.LINE_AA)
 
    # Bottom right
    cv2.line(image, (x2 - radius, y2), (x1 + radius + length, y2), color, thickness, cv2.LINE_AA)
    cv2.line(image, (x2, y2 - radius), (x2, y1 + radius + length), color, thickness, cv2.LINE_AA)
    cv2.ellipse(image, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness, cv2.LINE_AA)

def plot_box(image, top_left_point, bottom_right_point, width, height, label, color=(210,240,0), padding=6, font_scale=0.375, alpha=0.15):
    if alpha > 1:
        alpha = 1
    
    if alpha > 0:
        box_crop = image[top_left_point['y']:top_left_point['y']+height, top_left_point['x']:top_left_point['x']+width]
        colored_rect = np.ones(box_crop.shape, dtype=np.uint8)
        colored_rect[:,:,0] = color[0] - 90 if color[0] - 90 >= 0 else 0
        colored_rect[:,:,1] = color[1] - 90 if color[1] - 90 >= 0 else 0
        colored_rect[:,:,2] = color[2] - 90 if color[2] - 90 >= 0 else 0
        box_crop_weighted = cv2.addWeighted(box_crop, 1 - alpha, colored_rect, alpha, 1.0)
        image[top_left_point['y']:top_left_point['y']+height, top_left_point['x']:top_left_point['x']+width] = box_crop_weighted

    cv2.rectangle(image, (top_left_point['x'] - 1, top_left_point['y']), (bottom_right_point['x'], bottom_right_point['y']), color, thickness=2, lineType=cv2.LINE_AA)
    res_scale = (image.shape[0] + image.shape[1])/1600
    font_scale = font_scale * res_scale
    font_width, font_height = 0, 0
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(label, font_face, fontScale=font_scale, thickness=1)[0]

    if text_size[0] > font_width:
        font_width = text_size[0]
    if text_size[1] > font_height:
        font_height = text_size[1]
    if top_left_point['x'] - 1 < 0:
        top_left_point['x'] = 1
    if top_left_point['x'] + font_width + padding*2 > image.shape[1]:
        top_left_point['x'] = image.shape[1] - font_width - padding*2
    if top_left_point['y'] - font_height - padding*2  < 0:
        top_left_point['y'] = font_height + padding*2
    
    p3 = top_left_point['x'] + font_width + padding*2, top_left_point['y'] - font_height - padding*2
    cv2.rectangle(image, (top_left_point['x'] - 1, top_left_point['y']), p3, color, -1, lineType=cv2.LINE_AA)
    x = top_left_point['x'] + padding
    y = top_left_point['y'] - padding
    cv2.putText(image, label, (x, y), font_face, font_scale, [0, 0, 0], thickness=1, lineType=cv2.LINE_AA)

    return image

def draw(image, detections, classes=None, alpha=0.15):
    image_copy = image.copy()

    for box in detections:
        draw_box = False
        class_name = box['class']
        conf = box['confidence']
        label = f'{class_name} {str(int(conf*100))}%' + (f' | {box["text"].upper()}' if 'text' in box else '')
        width = box['width']
        height = box['height']
        top_left_point = {'x':box['x'], 'y':box['y']}
        bottom_right_point = {'x':box['x'] + width, 'y':box['y'] + height}
        if (classes is None) or (classes is not None and class_name in classes):
            draw_box = True
        if draw_box:
            image_copy = plot_box(image_copy, top_left_point, bottom_right_point, width, height, label, alpha=alpha)
    
    return image_copy

def get_time():
    return datetime.now().strftime('%d-%m-%Y_%I:%M:%S_%p')

def save_frame(frame, path):
    cv2.imwrite(path, frame)

def save_json(data, path):
    with open(path, 'w') as json_file:
        json_file.write(json.dumps(data, indent=4))