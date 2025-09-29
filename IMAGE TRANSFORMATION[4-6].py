## DEVELOPED BY
## NAME : HARI PRIYA M
## REG NO : 212224240047

##iv)Image Reflection
import numpy as np
import cv2
import matplotlib.pyplot as plt
input_image=cv2.imread("got.jpg") 
input_image=cv2.cvtColor(input_image,) 
plt.axis("off") 
plt.imshow(input_image)
plt.show()
rows, cols, dim = 
M_x=np.float
reflected_img_xaxis=cv2.warpPerspective(input_image,M_x,(cols,rows))
reflected_img_yaxis=cv2.warpPerspective(input_image,M_y,(cols,rows))
plt.imshow(reflected_)
plt.show()

##v)Image Rotation:
import numpy as np
import cv2
import matplotlib.pyplot as plt
input_image=cv2.imread("got.jpg") 
input_image=cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
angle=np.radians(45)
M=np.float32([[np.cos(angle),-(np.sin(angle)),0],
               [np.
rotated_img=

##vi)Image Cropping:

import numpy as np
import cv2
import matplotlib.pyplot as plt
input_image=cv2.imread("got.jpg") 
input_image=cv2.cvtColor(input_image,
plt.show()
