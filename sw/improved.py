from PIL import Image
import time
import pytesseract
import argparse
import cv2
import os

def scan(cap):
    while (1):
        # Capture frame-by-frame
        ret, frame = cap.read()
        # Our operations on the frame come here
        # Image processing, sharpening, etc
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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
        cv2.line(test_image, (320, 480), (960, 480), (255, 0, 0), 2)
        # Display the resulting frame
        cv2.imshow('frame', test_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            return gray


def main():
    port = 1
    cap = cv2.VideoCapture(port)
    cap.set(3, 1280)
    cap.set(4, 800)
    img = "{}.jpg".format(time.strftime("%H_%M_%S"))

    cv2.imwrite(img,scan(cap))

    text = pytesseract.image_to_string(Image.open(img))
    print(text)

    cap.release()
    cv2.destroyAllWindows()

main()