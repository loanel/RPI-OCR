from PIL import Image
import time
import pytesseract
import argparse
import cv2
import os
import numpy as np


def binary_treshold(image):
    return cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)[1]


def adaptive_threshold(image):
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 3)


def gaussian_threshold(image):
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3)


def reverse_threshold(image):
    return cv2.threshold(image, 100, 255, cv2.THRESH_BINARY_INV)[1]


def otsu_threshold_no_blurring(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]


def otsu_threshold_blurring(image):
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    return cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

def reverse_gaussian_threshold(image):
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3)
    return 255 - image

def non_thresholded(image):
    return image

def scan(cap, thresholding_function):
    while (1):
        # Capture frame-by-frame
        ret, frame = cap.read()
        # Our operations on the frame come here
        # Image processing, sharpening, etc
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = thresholding_function(gray)
        # check to see if we should apply threshold+
        # ing to preprocess the
        # image
        # if args["preprocess"] == "thresh":
        #     gray = cv2.threshold(gray, 0, 255,
        #                          cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        #
        # # make a check to see if median blurring should be done to remove
        # # noise
        # elif args["preprocess"] == "blur":
        #     gray = cv2.medianBlur(gray, 3)


        test_image = gray.copy()
        # cv2.line(test_image, (320, 480), (960, 480), (255, 0, 0), 2)
        # Display the resulting frame
        cv2.imshow('frame', test_image)

        p = cv2.waitKey(1)

        if p & 0xFF == ord('q'):
            print("Taking picture for tessaract evaluation")
            return gray

        elif p & 0xFF == ord('b'):
            print("Changed thresholding to Binary (default)")
            thresholding_function = binary_treshold

        elif p & 0xFF == ord('a'):
            print("Changed thresholding to adaptive using mean of neighbours")
            thresholding_function = adaptive_threshold

        elif p & 0xFF == ord('g'):
            print("Changed thresholding to adaptive using gaussian neighbours")
            thresholding_function = gaussian_threshold

        elif p & 0xFF == ord('r'):
            print("Changed thresholding to reverse")
            thresholding_function = reverse_threshold

        elif p & 0xFF == ord('o'):
            print("Changed thresholding to otsu without blurring")
            thresholding_function = gaussian_threshold

        elif p & 0xFF == ord('p'):
            print("Changed thresholding to otsu with blurring")
            thresholding_function = gaussian_threshold

        elif p & 0xFF == ord('d'):
            print("Changed thresholding to reverse gaussian")
            thresholding_function = reverse_gaussian_threshold

        elif p & 0xFF == ord('n'):
            print("No thresholding")
            thresholding_function = non_thresholded


def deskew_text(image):
    #convert image to grayscale for safety, set background to black
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)

    # threshold the image, setting all foreground pixels to
    # 255 and all background pixels to 0
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # grab the (x, y) coordinates of all pixel values that
    # are greater than zero, then use these coordinates to
    # compute a rotated bounding box that contains all
    # coordinates
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]

    # the `cv2.minAreaRect` function returns values in the
    # range [-90, 0); as the rectangle rotates clockwise the
    # returned angle trends to 0 -- in this special case we
    # need to add 90 degrees to the angle
    if angle < -45:
        angle = -(90 + angle)

    # otherwise, just take the inverse of the angle to make
    # it positive
    else:
        angle = -angle

    # rotate the image to deskew it
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # draw the correction angle on the image so we can validate it
    cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # show the output image
    print("[INFO] angle: {:.3f}".format(angle))
    cv2.imshow("Input", image)
    cv2.imshow("Rotated", rotated)
    cv2.waitKey(0)
    return rotated


def main():
    port = 0
    cap = cv2.VideoCapture(port)
    #cap.set(3, 1200)
    #cap.set(4, 900)
    name = time.strftime("%H_%M_%S")
    img = "{}.jpg".format(name)

    cv2.imwrite(img, scan(cap, non_thresholded))
    cap.release()
    #trying to deskew the text image
    print("Do you wish to try rotating your image using a deskewing function?")
    print("This is advised only if your text was contained in a box (your text had borders on each side)"\
          "and you used binary thresholding")
    nb = input("Apply deskewing? y/n: ")
    if nb == 'y':
        image = cv2.imread(img)
        image = deskew_text(image)
        img = "{}.jpg".format(name + "DESKEWED")
        cv2.imwrite(img, image)



    print("\nAttempting to retrieve text from scanned picture, this might take a while for bigger documents\n")
    text = pytesseract.image_to_string(Image.open(img))
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
    print(text)

    cap.release()
    cv2.destroyAllWindows()


main()
