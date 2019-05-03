#resize
#crop
#flip
#brightness & contrast
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.pyplot import imshow

"""--------------------------------------"""
"""
name = './heart.jpg'
img_string = tf.read_file(name)
img_decoded = tf.image.decode_image(img_string)
sess = tf.Session()
img_decoded_val = sess.run(img_decoded)
print(img_decoded_val.shape)

imshow(img_decoded_val)
"""
"""show img"""
"""
img_string = mpimg.imread('heart.jpg')
img_string.shape
plt.imshow(img_string)
plt.axis('off')
plt.show()
"""
"""--------------------------------------"""
"""
img_string_1 = img_string[:,:,0]
plt.imshow(img_string_1)
plt.show()   #单通到显示
"""


"""--------------------------------------"""
"""
#resize
#tf.image.resize_area
#tf.image.resize_bicubic
#tf.image.resize_nearest_neighbor

name = './heart.jpg'
img_string = tf.read_file(name)
img_decoded = tf.image.decode_image(img_string)
img_decoded = tf.reshape(img_decoded,[1,208,242,3])
resize_img = tf.image.resize_bicubic(img_decoded,[416,484])
sess = tf.Session()
img_decoded_val = sess.run(resize_img)
img_decoded_val = img_decoded_val.reshape((416,484,3))
img_decoded_val = np.asarray(img_decoded_val,np.uint8)
print(img_decoded_val.shape)

plt.imshow(img_decoded_val)
plt.show()
"""

"""--------------------------------------"""
"""#corp 裁剪
#tf.image.pad_to_bounding_box
#tf.image.crop_to_bounding_box
#tf.random_crop
name = './heart.jpg'
img_string = tf.read_file(name)
img_decoded = tf.image.decode_image(img_string)
img_decoded = tf.reshape(img_decoded,[1,208,242,3])

padded_img = tf.image.pad_to_bounding_box(img_decoded,50,100,300,400)
sess = tf.Session()
img_decoded_val = sess.run(padded_img)
img_decoded_val = img_decoded_val.reshape((300,400,3))
img_decoded_val = np.asarray(img_decoded_val,np.uint8)
print(img_decoded_val.shape)

plt.imshow(img_decoded_val)
plt.show()

"""
"""--------------------------------------"""
"""
#flip 翻转
#tf.image.filp_up_down
#tf.image.filp_left_right
#tf.image.random_filp_up_down
#tf.image.random_filp_left_right
name = './heart.jpg'
img_string = tf.read_file(name)
img_decoded = tf.image.decode_image(img_string)
img_decoded = tf.reshape(img_decoded,[1,208,242,3])

flipped_img = tf.image.flip_up_down(img_decoded)

sess = tf.Session()
img_decoded_val = sess.run(flipped_img)
img_decoded_val = img_decoded_val.reshape((208,242,3))
img_decoded_val = np.asarray(img_decoded_val,np.uint8)
print(img_decoded_val.shape)

plt.imshow(img_decoded_val)
plt.show()
"""

"""--------------------------------------"""
# brightness
#tf.image.adjust_brightness
#tf.image.random_brightness
#tf.image.adjust_constrast
#tf.image.random_constrast
name = './heart.jpg'
img_string = tf.read_file(name)
img_decoded = tf.image.decode_image(img_string)
img_decoded = tf.reshape(img_decoded,[1,208,242,3])

new_img = tf.image.adjust_brightness(img_decoded,+0.3)

sess = tf.Session()
img_decoded_val = sess.run(new_img)
img_decoded_val = img_decoded_val.reshape((208,242,3))
img_decoded_val = np.asarray(img_decoded_val,np.uint8)
print(img_decoded_val.shape)

plt.imshow(img_decoded_val)
plt.show()


