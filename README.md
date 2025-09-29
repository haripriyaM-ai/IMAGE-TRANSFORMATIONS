# EXP 04 : IMAGE-TRANSFORMATIONS
### NAME : HARI PRIYA M
### REG NO : 212224240047

## Aim
To perform image transformation such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping using OpenCV and Python.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step1: Load and Display Image
Read the image from a file and show it using Matplotlib.

### Step2: Geometric Transformations
   - Translate: Move image along x and y axes.  
   - Scale: Resize image using width and height factors.  
   - Shear: Slant image using a shear factor.


### Step3: Reflection / Flip
 - Flip the image horizontally (or vertically) to create a mirror effect.

### Step4: Rotation
- Rotate the image around its center by a specified angle.

### Step5: Cropping
 - Extract a portion of the image by selecting start and end rows and columns.
<br>

## Program:
```python
Developed By: HARI PRIYA M
Register Number: 212224240047


import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to display image using Matplotlib
def display_image(image, title):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for proper color display
    plt.imshow(image_rgb)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Load an image
image = cv2.imread('/content/buildings-different-designs-glass.jpg')
display_image(image, 'Original Image')
```


i)Image Translation
```python
def translate(img, x, y):
    M = np.float32([[1, 0, x], [0, 1, y]])
    translated = cv2.warpAffine(img, M, (img.shape[1] + 200, img.shape[0] + 200))
    return translated

translated_image = translate(image, 100, 50)
display_image(translated_image, 'Translated Image')
```

 ii) Image Scaling
 ```python
def scale(img, scale_x, scale_y):
    scaled = cv2.resize(img, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
    return scaled

scaled_image = scale(image, 1.5, 1.5)
display_image(scaled_image, 'Scaled Image')
```

iii)Image shearing
```python
def shear(img, shear_factor):
    rows, cols, _ = img.shape
    M = np.float32([[1, shear_factor, 0], [0, 1, 0]])
    sheared = cv2.warpAffine(img, M, (cols, rows))
    return sheared

sheared_image = shear(image, 0.5)
display_image(sheared_image, 'Sheared Image')
```

iv)Image Reflection
```python
def reflect(img):
    reflected = cv2.flip(img, 1)  # 1 for horizontal flip
    return reflected

reflected_image = reflect(image)
display_image(reflected_image, 'Reflected Image')
```

v)Image Rotation
```python
def rotate(img, angle):
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    return rotated

rotated_image = rotate(image, 45)
display_image(rotated_image, 'Rotated Image')
```


vi)Image Cropping
```python
def crop(img, start_row, start_col, end_row, end_col):
    cropped = img[start_row:end_row, start_col:end_col]
    return cropped

cropped_image = crop(image, 100, 100, 250, 250)
display_image(cropped_image, 'Cropped Image')

```

## Output:

### Original image
<br>
<img width="500" height="500" alt="Screenshot 2025-09-29 101313" src="https://github.com/user-attachments/assets/b9d09552-ff6b-4f75-93e6-b19debaa6b0c" />
<br>

### i)Image Translation
<br>
<img width="500" height="500" alt="Screenshot 2025-09-29 101329" src="https://github.com/user-attachments/assets/4b1423a5-1add-4f45-b8ea-d599f02cb365" />

<br>

### ii) Image Scaling
<br>
<img width="500" height="500" alt="Screenshot 2025-09-29 101346" src="https://github.com/user-attachments/assets/82717975-d6bd-4b47-9496-575c1278729e" />

<br>


### iii)Image shearing
<br>
<img width="500" height="500" alt="Screenshot 2025-09-29 101408" src="https://github.com/user-attachments/assets/f3c22ef4-5756-49bf-805e-7614b54bb9f6" />

<br>


### iv)Image Reflection
<br>
<img width="500" height="500" alt="Screenshot 2025-09-29 101420" src="https://github.com/user-attachments/assets/2a875a0e-ea00-435a-99a2-c94efcd22f13" />

<br>



### v)Image Rotation
<br>
<img width="500" height="500" alt="Screenshot 2025-09-29 101433" src="https://github.com/user-attachments/assets/d8216104-cd29-4add-8a25-fdd1e6dc3bb9" />

<br>



### vi)Image Cropping
<br>
<img width="500" height="400" alt="Screenshot 2025-09-29 101442" src="https://github.com/user-attachments/assets/b70339a4-a33e-449d-b039-91058562b428" />

<br>




## Result: 

Thus the different image transformations such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping are done using OpenCV and python programming.
