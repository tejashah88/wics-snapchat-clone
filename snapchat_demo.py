from tkinter import *
from PIL import Image
from PIL import ImageTk
import cv2
import threading
import os
import time
from threading import Thread
from os import listdir
from os.path import isfile, join

import dlib
from imutils import face_utils, rotate_bound
import math


def put_sprite(num):
    global SPRITES, BTNS
    SPRITES[num] = (1 - SPRITES[num])  # not actual value
    BTNS[num].config(relief=SUNKEN if SPRITES[num] else RAISED)


def draw_sprite(frame, sprite, x_offset, y_offset):
    (h, w) = (sprite.shape[0], sprite.shape[1])
    (imgH, imgW) = (frame.shape[0], frame.shape[1])

    if y_offset + h >= imgH:  # if sprite gets out of image in the bottom
        sprite = sprite[0:imgH - y_offset, :, :]

    if x_offset + w >= imgW:  # if sprite gets out of image to the right
        sprite = sprite[:, 0:imgW - x_offset, :]

    if x_offset < 0:  # if sprite gets out of image to the left
        sprite = sprite[:, abs(x_offset)::, :]
        w = sprite.shape[1]
        x_offset = 0

    # for each RGB chanel
    for c in range(3):
        # chanel 4 is alpha: 255 is not transparent, 0 is transparent background
        frame[y_offset:y_offset + h, x_offset:x_offset + w, c] =  \
            sprite[:, :, c] * (sprite[:, :, 3] / 255.0) + frame[y_offset:y_offset + h, x_offset:x_offset + w, c] * (1.0 - sprite[:, :, 3] / 255.0)
    return frame


def adjust_sprite2head(sprite, head_width, head_ypos, ontop=True):
    (h_sprite, w_sprite) = (sprite.shape[0], sprite.shape[1])
    factor = 1.0 * head_width / w_sprite
    # adjust to have the same width as head
    sprite = cv2.resize(sprite, (0, 0), fx=factor, fy=factor)
    (h_sprite, w_sprite) = (sprite.shape[0], sprite.shape[1])

    # adjust the position of sprite to end where the head begins
    y_orig = head_ypos - h_sprite if ontop else head_ypos
    if (y_orig < 0):  # check if the head is not to close to the top of the image and the sprite would not fit in the screen
        sprite = sprite[abs(y_orig)::, :, :]  # in that case, we cut the sprite
        y_orig = 0  # the sprite then begins at the top of the image
    return (sprite, y_orig)


def apply_sprite(image, path2sprite, w, x, y, angle, ontop=True):
    sprite = cv2.imread(path2sprite, -1)
    # print sprite.shape
    sprite = rotate_bound(sprite, angle)
    (sprite, y_final) = adjust_sprite2head(sprite, w, y, ontop)
    image = draw_sprite(image, sprite, x, y_final)


def calc_slope(point1, point2):
    x1, x2, y1, y2 = point1[0], point2[0], point1[1], point2[1]
    incl = 180 / math.pi * math.atan((float(y2 - y1)) / (x2 - x1))
    return incl


def calculate_boundbox(list_coordinates):
    x = min(list_coordinates[:, 0])
    y = min(list_coordinates[:, 1])
    w = max(list_coordinates[:, 0]) - x
    h = max(list_coordinates[:, 1]) - y
    return (x, y, w, h)


def get_face_boundbox(points, face_part):
    if face_part == 1:
        (x, y, w, h) = calculate_boundbox(points[17:22])  # left eyebrow
    elif face_part == 2:
        (x, y, w, h) = calculate_boundbox(points[22:27])  # right eyebrow
    elif face_part == 3:
        (x, y, w, h) = calculate_boundbox(points[36:42])  # left eye
    elif face_part == 4:
        (x, y, w, h) = calculate_boundbox(points[42:48])  # right eye
    elif face_part == 5:
        (x, y, w, h) = calculate_boundbox(points[29:36])  # nose
    elif face_part == 6:
        (x, y, w, h) = calculate_boundbox(points[48:68])  # mouth
    return (x, y, w, h)


def cvloop(run_event):
    global panelA
    global SPRITES

    # for flies animation
    dir_ = "./sprites/flies/"
    flies = [f for f in listdir(dir_) if isfile(join(dir_, f))]
    i = 0

    video_capture = cv2.VideoCapture(0)  # read from webcam
    (x, y, w, h) = (0, 0, 10, 10)  # whatever initial values

    # acquire dlib's face detector
    detector = dlib.get_frontal_face_detector()

    # Facial landmarks
    print("[INFO] loading facial landmark predictor...")
    model = "filters/shape_predictor_68_face_landmarks.dat"

    # link to model: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    predictor = dlib.shape_predictor(model)

    while run_event.is_set():  # while the thread is active we loop
        ret, image = video_capture.read()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 0)

        for face in faces:  # if there are faces
            (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
            # find facial landmarks
            shape = predictor(gray, face)
            shape = face_utils.shape_to_np(shape)

            # get face tilt inclination based on eyebrows
            incl = calc_slope(shape[17], shape[26])

            # check if mouth is open for doggy filter
            # y coordiantes of landmark points of lips
            is_mouth_open = (shape[66][1] - shape[62][1]) >= 10

            # add a hat
            if SPRITES[0]:
                apply_sprite(image, "./sprites/hat.png", w, x, y, incl)

            # add a mustache
            if SPRITES[1]:
                (x1, y1, w1, h1) = get_face_boundbox(shape, 6)
                apply_sprite(image, "./sprites/mustache.png", w1, x1, y1, incl)

            # add some animated flies
            if SPRITES[2]:
                apply_sprite(image, dir_ + flies[i], w, x, y, incl)
                # when done with all images of that folder, begin again
                i = (i + 1) % len(flies)

            # add some glasses
            if SPRITES[3]:
                (x3, y3, _, h3) = get_face_boundbox(shape, 1)
                apply_sprite(image, "./sprites/glasses.png", w, x, y3, incl, ontop=False)

            # add some doggy things
            (x0, y0, w0, h0) = get_face_boundbox(shape, 6)  # bound box of mouth
            if SPRITES[4]:
                (x3, y3, w3, h3) = get_face_boundbox(shape, 5)  # nose
                apply_sprite(image, "./sprites/doggy_nose.png", w3, x3, y3, incl, ontop=False)
                apply_sprite(image, "./sprites/doggy_ears.png", w, x, y, incl)

                if is_mouth_open:
                    apply_sprite(image, "./sprites/doggy_tongue.png", w0, x0, y0, incl, ontop=False)
            # if SPRITES[5]:
            #     (left_x5, left_y5, left_w5, left_h5) = get_face_boundbox(shape, 3)
            #     (right_x5, right_y5, right_w5, right_h5) = get_face_boundbox(shape, 4)
            #     apply_sprite(image, "./sprites/eye.png", w // 2, x, left_y5, incl, ontop=False)
            #     apply_sprite(image, "./sprites/eye.png", w // 2, x, right_y5, incl, ontop=False)
            else:
                if is_mouth_open:
                    apply_sprite(image, "./sprites/rainbow.png", w0, x0, y0, incl, ontop=False)

        # OpenCV == BGR; PIL == RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        panelA.configure(image=image)
        panelA.image = image

    video_capture.release()


# Initialize GUI object
root = Tk()
root.title("Not really Snapchat")
this_dir = os.path.dirname(os.path.realpath(__file__))

# Create 5 buttons and assign their corresponding function to active sprites
btn1 = Button(root, text="Hat", command=lambda: put_sprite(0))
btn1.pack(side="top", fill="both", expand="no", padx="5", pady="5")

btn2 = Button(root, text="Mustache", command=lambda: put_sprite(1))
btn2.pack(side="top", fill="both", expand="no", padx="5", pady="5")

btn3 = Button(root, text="Flies", command=lambda: put_sprite(2))
btn3.pack(side="top", fill="both", expand="no", padx="5", pady="5")

btn4 = Button(root, text="Glasses", command=lambda: put_sprite(3))
btn4.pack(side="top", fill="both", expand="no", padx="5", pady="5")

btn5 = Button(root, text="Doggy", command=lambda: put_sprite(4))
btn5.pack(side="top", fill="both", expand="no", padx="5", pady="5")

# btn6 = Button(root, text="Googly Eyes", command=lambda: put_sprite(5))
# btn6.pack(side="top", fill="both", expand="no", padx="5", pady="5")

panelA = Label(root)
panelA.pack(padx=10, pady=10)


# hat, mustache, flies, glasses, doggy, eyes -> 1 is visible, 0 is not visible
SPRITES = [0, 0, 0, 0, 0]
BTNS = [btn1, btn2, btn3, btn4, btn5]

# Creates a thread for openCV processing
run_event = threading.Event()
run_event.set()
action = Thread(target=cvloop, args=(run_event,))
action.setDaemon(True)
action.start()


# Function to clean everything up
def terminate():
    global root, run_event, action
    print("Closing thread opencv...")
    run_event.clear()
    time.sleep(1)
    # action.join() #strangely in Linux this thread does not terminate properly, so .join never finishes
    root.destroy()
    print("All closed!")


# When the GUI is closed it actives the terminate function
root.protocol("WM_DELETE_WINDOW", terminate)
root.mainloop()  # creates loop of GUI
