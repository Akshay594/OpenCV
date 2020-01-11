import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
Affine transform that simply shifts the position of an image.
we use warpAffine from cv2.

For translating an image from one position to another
we will be needing a translation matrix
"""
image = cv2.imread('goku.png')
# cv2.imshow('image', image)

## Translation matrix
M = np.float32([[1, 0, image.shape[0]//4],
				[0, 1, image.shape[1]//4]])

# print(image.shape[0])
# print(image.shape[0]//4)
## We will use warpAffine to transform the image using matrix M
# translated_image = cv2.warpAffine(image, M, (image.shape[0], image.shape[1]))
# cv2.imshow('image', image)
# cv2.imshow('translated_image', translated_image)

## Rotation - For this we need rcenter(x, y), theta, scale
height, width = image.shape[:2]
# rotation_matrix = cv2.getRotationMatrix2D((width//2, height//2), 90, .5)
# rotated_image = cv2.warpAffine(image, rotation_matrix, (height, width))
# cv2.imshow('rotated_image', rotated_image)

# rotated_image_2 = cv2.transpose(image)
# cv2.imshow('transposed image', rotated_image_2)

## Flipping an Image

## Horizontal Flip
flipped_image = cv2.flip(image, 1)
ver_image = cv2.flip(image, 0)
both_flipped = cv2.flip(image, -1)

# cv2.imshow('image', image)
# cv2.imshow('flipped_image', flipped_image)
# cv2.imshow('ver_image', ver_image)
# cv2.imshow('both_flipped', both_flipped)


## Scaling, resizing and interpolation
## cv2.resize(image, dsize(output image, size), x_scale, y_scale, interpolation)
image_scaled = cv2.resize(image, None, fx=0.15, fy=0.15)
# print(f"{image.shape[0]}, {image.shape[1]}")
# print(f"{image.shape[0]*0.15}, {image.shape[1]*0.15}")

# cv2.imshow('image_scaled', image_scaled)

## Image Pyramids
## These are useful with scaling images in object detection

"""
Gaussian: Imagine a pyramid as a set of layers in which the higher the
layer, the smaller the size.

Every layer is from bottom to top and we need to convolve the
entire image array with some scaler value.


Laplacian

"""

# smaller = cv2.pyrDown(image)
# larger = cv2.pyrUp(smaller)
# print(image.shape)
# print(smaller.shape)
# print(larger.shape)
# cv2.imshow('smaller', smaller)
# cv2.imshow('larger', larger)

## Cropping
x, y = image.shape[:2]
# Let's get the starting pixel coordiantes (top  left of cropping rectangle)
start_row, start_col = int(height * .25), int(width * .25)

# Let's get the ending pixel coordinates (bottom right)
end_row, end_col = int(height * .75), int(width * .75)

# Simply use indexing to crop out the rectangle we desire
cropped = image[start_row:end_row , start_col:end_col]

# cv2.imshow('image', image)
# cv2.imshow("cropped", cropped)

## Airthmetic Operations
# M = np.ones(image.shape, dtype="uint8") * 255
# added = cv2.add(image, M)
# sub = cv2.subtract(image, M)

# cv2.imshow('add', added)
# cv2.imshow('subtract', sub)

# ## Bitwise Operations and Masking

square = np.zeros((300, 300), np.uint8)
cv2.rectangle(square, (50, 50), (250, 250), 255, -1)
ellipse = np.zeros((300, 300), np.uint8)
cv2.ellipse(ellipse, (150, 150), (150, 150), 30, 0, 180, 255, -1)

## Bitwise AND, OR, XOR, NOT
And = cv2.bitwise_and(square, ellipse)
Or = cv2.bitwise_or(square, ellipse)
Xor = cv2.bitwise_xor(square, ellipse)



# cv2.imshow('And', And)
# cv2.imshow('Xor', Xor)
# cv2.imshow('Or', Or)



## Sharpening

kernel_sharpening = np.array([[-1,-1,-1], 
                              [-1,9,-1], 
                              [-1,-1,-1]])
sharped_image = cv2.filter2D(image, -1, kernel_sharpening)
# cv2.imshow('sharped_image', sharped_image)

kernel_3x3 = np.ones((3, 3), np.float32)/9

## We use .filter2D to convolve the kernel with an image
## -1 means anchor is at kernel center
blurred = cv2.filter2D(image, -1, kernel_3x3)
# cv2.imshow('blurred', blurred)

kernel_7x7 = np.ones((7, 7), np.float32)/9
# blurred_2 = cv2.filter2D(image, -1, kernel_7x7)
cv2.imshow('blurred 2', blurred_2)

cv2.waitKey(0)




