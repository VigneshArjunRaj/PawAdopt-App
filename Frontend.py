import torch
from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from PIL import Image, ImageColor

import cv2
from collections import Counter
from sklearn.cluster import KMeans
from matplotlib import colors
import webcolors as wc
from skimage import io, color
import warnings
warnings.filterwarnings('ignore') 

# from DogAdop import cv
model_dog = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt')

def cv_dog(img):

    x1 = 0
    y1 = 0
    w = 0
    h = 0
    # Run detection on the image
    img1 = img.copy()
    img = np.array(img) 
    # Convert RGB to BGR 
    img = img[:, :, ::-1].copy() 
    results = model_dog(img)
    print(results)

    # Get the detected object labels, scores, and bounding boxes
    labels = results.xyxy[0][:, -1].tolist()
    scores = results.xyxy[0][:, -2].tolist()
    bboxes = results.xyxy[0][:, :-2].tolist()


    # Extract the x1, y1, w, and h values of each bounding box
    bboxes_xywh = []
    for bbox in bboxes:
        x1, y1, x2, y2 = map(int, bbox)
        w = x2 - x1
        h = y2 - y1
        bboxes_xywh.append([x1, y1, w, h])

    # Print the x1, y1, w, and h values of each bounding box
    for label, score, bbox_xywh in zip(labels, scores, bboxes_xywh):
        x1, y1, w, h = bbox_xywh
        print(f" x1={x1}, y1={y1}, w={w}, h={h}, score={score:.2f}")


    # Get detections
    detections = results.xyxy[0]

    count = 0
    # Draw bounding boxes around detected targets
    for detection in detections:
        x1, y1, x2, y2, confidence, class_id = detection.tolist()
        w, h = x2 - x1, y2 - y1
        label = model_dog.names[int(class_id)]
        color = (0, 0,255)  # Green color for bounding box
        thickness = 2
        print(label)
        if label == "dog":
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
            cv2.putText(img, f"{label} {confidence:.2f}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
            count += 1

    print('Number of Dogs :',count)


    #Position of the Bounding box
    # Calculate the center coordinates of the image
    image_center_x = img.shape[1] / 2
    image_center_y = img.shape[0] / 2

    # Calculate the center coordinates of the bounding box
    bbox_center_x = x1 + w / 2
    bbox_center_y = y1 + h / 2

    # Determine the position of the bounding box relative to the center of the image
    position =""
    if count == 1:
        condition = 'Satisfied'
        if bbox_center_x < image_center_x - img.shape[1] / 15:
            position = "left side of the image"
            print("Bounding box is on the left side of the image")
        elif bbox_center_x > image_center_x + img.shape[1] / 15:
            position = "right side of the image"
            print("Bounding box is on the right side of the image")
        else:
            position = "center of the image"
            print("Bounding box is in the center of the image")

    elif count == 0:
        condition = 'No Dogs Detected'
        print('No Dogs Detected')

    else:
        condition = 'More than 1 dog detected'
        print('More than 1 dog detected')
    return condition,count,position


def plot_colors(img):
    # Read the image and convert to RGB
    img1 = np.array(img) 
    # Convert RGB to BGR 
    img1 = img1[:, :, ::-1].copy() 
    hsv_img = color.rgb2hsv(img1)
    
    raw_img = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    
    # Define a function to convert RGB color to hex color
    def rgb_to_hex(rgb_color):
        hex_color = '#'
        for i in rgb_color:
            i = int(i)
            hex_color += ('{:02x}'.format(i))
        return hex_color
    
    # Resize the image and reshape to 1D array
    img = cv2.resize(raw_img, (900, 600), interpolation=cv2.INTER_AREA)
    img = img.reshape(img.shape[0] * img.shape[1], 3)
    
    # Perform KMeans clustering to quantize colors
    clf = KMeans(n_clusters=5)
    color_labels = clf.fit_predict(img)
    center_colors = clf.cluster_centers_
    counts = Counter(color_labels)
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [rgb_to_hex(ordered_colors[i]) for i in counts.keys()]
    
    # Create the pie chart and return it
    fig = plt.figure(figsize=(12, 8))
    plt.pie(counts.values(), labels=hex_colors, autopct='%1.1f%%', colors=hex_colors)
    return fig

def image_info(image):
    

    # Get the image size
    width, height = image.size

    # Get the image format
    image_format = image.format

    # Get the image mode
    image_mode = image.mode

    # Get the image quality
    image_quality = image.info.get('quality')

    
    # Convert the image to grayscale
    grayscale_img = image.convert('L')

    # Calculate the brightness level
    brightness = 0
    pixels = grayscale_img.getdata()
    for pixel in pixels:
        brightness += pixel
    brightness /= len(pixels)


    # Convert to percentage
    brightness_percentage = round((brightness / 255) * 100,2)


    # Convert the image to the HSV color space
    img1 = np.array(image) 
    # Convert RGB to BGR 
    img1 = img1[:, :, ::-1].copy() 
    hsv_img = color.rgb2hsv(img1)

    # Calculate the saturation level
    saturation = hsv_img[:, :, 1].mean()

    # Convert to percentage
    saturation_percentage = round(saturation * 100,2)


    resolution = f'{width}x{height} pixels'
    # Print the result
#     print(f"Saturation level: {saturation_percentage:.2f}%")

#     # Print the result
#     print(f"Brightness level: {brightness_percentage:.2f}%")
#     # Print the results
#     print(f"Resolution: {width}x{height} pixels")
#     print(f"Image format: {image_format}")
#     print(f"Image mode: {image_mode}")
#     print(f"Image quality: {image_quality}")
    
    return brightness_percentage,saturation_percentage,resolution,image_format,image_mode,image_quality


def plot(file_path):
    # Read the image and convert to RGB
    raw_img = cv2.imread(file_path)
    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    
    # Define a function to convert RGB color to hex color
    def rgb_to_hex(rgb_color):
        hex_color = '#'
        for i in rgb_color:
            i = int(i)
            hex_color += ('{:02x}'.format(i))
        return hex_color
    
    # Resize the image and reshape to 1D array
    img = cv2.resize(raw_img, (900, 600), interpolation=cv2.INTER_AREA)
    img = img.reshape(img.shape[0] * img.shape[1], 3)
    
    # Perform KMeans clustering to quantize colors
    clf = KMeans(n_clusters=5)
    color_labels = clf.fit_predict(img)
    center_colors = clf.cluster_centers_
    counts = Counter(color_labels)
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [rgb_to_hex(ordered_colors[i]) for i in counts.keys()]
    
    # Create the pie chart and return it
    fig = plt.figure(figsize=(12, 8))
    plt.pie(counts.values(), labels=hex_colors, autopct='%1.1f%%', colors=hex_colors)
    return fig


app = Flask(__name__,template_folder='templates')

# Load pre-trained CNN model
model = tf.keras.models.load_model('latest_weights.h5')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get uploaded image
        file = request.files['file']
        img = Image.open(file)
        img_org = img.copy()
        # img_byte_arr = BytesIO()
        # img.save(img_byte_arr, format='JPEG')
        # img_byte_arr = img_byte_arr.getvalue()
        # img = image.load_img(img_byte_arr, target_size=(224, 224))
        # x = image.img_to_array(img)
        img = img.resize((224,224))
        condition,count,position=cv_dog(img)

        if count == 0:
            return render_template('index.html')

        x = np.asarray(img)
        x = np.expand_dims(x, axis=0)
        x = tf.keras.applications.resnet.preprocess_input(x)

        # Run image through CNN model
        preds = model.predict(x)
        class_names = ["Negative","Neutral","Positive"]
        top_pred_idx = np.argmax(preds[0])
        top_pred_class = class_names[top_pred_idx]
        top_pred_prob = preds[0][top_pred_idx]
        
        arr=x

        unique_colors = np.unique(arr.reshape(-1, arr.shape[-1]), axis=0)
        hex_colors_ = ['#' + ''.join([hex(int(c))[2:].rjust(2, '0').replace("x","") for c in color]) for color in unique_colors][:7]
        hex_colors = [ImageColor.getcolor(hex_color,"RGB") for hex_color in hex_colors_]

        # # Create a pie chart
        # plt.pie([1]*len(hex_colors), labels=hex_colors,colors=hex_colors_)
        # plt.axis('equal')
        # plt.tight_layout()

        # # Convert plot to base64-encoded string
        buffer = BytesIO()
        # plt.savefig(buffer, format='png')
        # buffer.seek(0)
        fig = plot_colors(img_org)
        fig.savefig(buffer, format='png')
        buffer.seek(0)
        chart_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

        # Convert image to base64-encoded string
        buffer = BytesIO()
        img.save(buffer, format='JPEG')
        buffer.seek(0)
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        brightness_percentage,saturation_percentage,resolution,image_format,image_mode,image_quality = image_info(img_org)

        return render_template('result.html',
                               class_name=top_pred_class,
                               class_prob=preds[0],
                               img_str=img_str,
                               chart_image=chart_image,
                               brightness_percentage = brightness_percentage,
                               saturation_percentage=saturation_percentage,resolution = resolution,
                               image_format=image_format,image_mode=image_mode,image_quality=image_quality,
                               position=position

                               )

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
