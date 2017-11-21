from PIL import Image
import time
import pytesseract
import argparse
import cv2
import os

def get_pic_from_webcam():

    port = 1
    cap = cv2.VideoCapture(port)
    cap.set(3, 1440)
    cap.set(4, 960)
    for j in range(1, 20):
        retval, temp = cap.read()
    retval, image = cap.read()
    print(retval)
    filename = "testimage.png"
    cv2.imwrite(filename, image)
    return filename

# # construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True,
#                 help="path to input image to be OCR'd")
# ap.add_argument("-p", "--preprocess", type=str, default="thresh",
#                 help="type of preprocessing to be done")
# args = vars(ap.parse_args())
#
# # load the example image and convert it to grayscale
# image = cv2.imread(args["image"])
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# check to see if we should apply thresholding to preprocess the
# image
# if args["preprocess"] == "thresh":
#     gray = cv2.threshold(gray, 0, 255,
#                          cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
#
# # make a check to see if median blurring should be done to remove
# # noise
# elif args["preprocess"] == "blur":
#     gray = cv2.medianBlur(gray, 3)

# write the grayscale image to disk as a temporary file so we can
# apply OCR to it
# filename = "{}.png".format(os.getpid())
# cv2.imwrite(filename, gray)

# load the image as a PIL/Pillow image, apply OCR, and then delete
# the temporary file
img = get_pic_from_webcam()

# # load the example image and convert it to grayscale
image = cv2.imread("mangatest.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# gray = cv2.threshold(gray, 0, 255,
#  		cv2.THRESH_OTSU)[1]
filename = "{}.png".format(time.strftime("%H_%M_%S"))
cv2.imwrite(filename, gray)

print(filename)
text = pytesseract.image_to_string(Image.open(filename))
print(text)
