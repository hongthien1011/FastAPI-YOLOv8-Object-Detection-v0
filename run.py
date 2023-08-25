import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

input_image_name = r'D:\Phase2\Gray\2D_Basic_Floor_Plan-1024x695.jpg'
api_host = 'http://10.182.220.137:8000/detection/'
type_rq = 'img_object_detection_to_img'

files = {'file': open(input_image_name, 'rb')}

response = requests.post(api_host+type_rq, files=files)

img = Image.open(BytesIO(response.content)) 
plt.imshow(img)