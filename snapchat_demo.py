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


# body part IDs for retrieving face landmark info
LEFT_EYEBROW = 1
RIGHT_EYEBROW = 2
LEFT_EYE = 3
RIGHT_EYE = 4
NOSE = 5
MOUTH = 6

ALL_SPRITES = ["Hat", "Mustache", "Flies", "Glasses", "Doggy", "Rainbow", "Googly Eyes"]
WINDOW_NAME = "Not really snapchat"


# class Sprite:
#     def __init__(self, name, id):
#         self.name = name
#         self.id = id
#         self.enabled = False


# class StaticSprite(Sprite):
#     def __init__(self, name, id, file):
#         super().__init__(name, id)
#         self.file = file

#     def get_sprite(self):
#         return self.file


# class AnimatedSprite():
#     def __init__(self, name, id, files_dir):
#         super().__init__(name, id)
#         self.files = files_dir
#         self.counter = 0

#     def get_sprite(self):
#         return self.files[self.counter]

#     def animate(self):
#         self.counter = (self.counter + 1) % len(self.files)


def toggle_sprite(num):
    global SPRITES, BTNS
    SPRITES[num] = not SPRITES[num]
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
    for ch in range(3):
        # chanel 4 is alpha: 255 is not transparent, 0 is transparent background
        sprite_pixel = sprite[:, :, ch]
        sprite_alpha = (sprite[:, :, 3] / 255.0)

        img_pixel = frame[y_offset:y_offset + h, x_offset:x_offset + w, ch]
        img_alpha = (1.0 - sprite_alpha)

        frame[y_offset:y_offset + h, x_offset:x_offset + w, ch] = sprite_pixel * sprite_alpha + img_pixel * img_alpha

    return frame


def adjust_sprite2head(sprite, head_width, head_ypos, ontop=True):
    (h_sprite, w_sprite) = (sprite.shape[0], sprite.shape[1])
    factor = 1.0 * head_width / w_sprite
    # adjust to have the same width as head
    sprite = cv2.resize(sprite, (0, 0), fx=factor, fy=factor)
    (h_sprite, w_sprite) = (sprite.shape[0], sprite.shape[1])

    # adjust the position of sprite to end where the head begins
    y_orig = (head_ypos - h_sprite) if ontop else head_ypos
    if (y_orig < 0):  # check if the head is not to close to the top of the image and the sprite would not fit in the screen
        sprite = sprite[abs(y_orig)::, :, :]  # in that case, we cut the sprite
        y_orig = 0  # the sprite then begins at the top of the image
    return (sprite, y_orig)


def apply_sprite(image, path2sprite, w, x, y, angle, ontop=True):
    sprite = cv2.imread(path2sprite, -1)
    sprite = rotate_bound(sprite, angle)
    (sprite, y_final) = adjust_sprite2head(sprite, w, y, ontop)
    image = draw_sprite(image, sprite, x, y_final)


def calc_slope(point1, point2):
    x1, x2, y1, y2 = point1[0], point2[0], point1[1], point2[1]
    incl_rad = math.atan((float(y2 - y1)) / (x2 - x1))
    incl_deg = 180 / math.pi * incl_rad
    return incl_deg


def calculate_boundbox(coords):
    x = min(coords[:, 0])
    y = min(coords[:, 1])
    w = max(coords[:, 0]) - x
    h = max(coords[:, 1]) - y
    return (x, y, w, h)


def get_face_boundbox(points, face_part):
    input_points = None
    if face_part == LEFT_EYEBROW:
        input_points = points[17:22]
    elif face_part == RIGHT_EYEBROW:
        input_points = points[22:27]
    elif face_part == LEFT_EYE:
        input_points = points[36:42]
    elif face_part == RIGHT_EYE:
        input_points = points[42:48]
    elif face_part == NOSE:
        input_points = points[29:36]
    elif face_part == MOUTH:
        input_points = points[48:68]
    else:
        raise NotImplementedError(f'Invalid face part requested for bounding box! ID: {face_part}')

    (x, y, w, h) = calculate_boundbox(input_points)
    return (x, y, w, h)


def cvloop(run_event):
    global main_panel
    global SPRITES

    # for flies animation
    dir_ = "./sprites/flies/"
    flies = [f for f in listdir(dir_) if isfile(join(dir_, f))]
    i = 0

    video_capture = cv2.VideoCapture(0)

    # acquire dlib's face detector
    face_detector = dlib.get_frontal_face_detector()

    # load facial landmarks model
    print("[INFO] loading facial landmark predictor...")
    model = "filters/shape_predictor_68_face_landmarks.dat"

    # link to model: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    predictor = dlib.shape_predictor(model)

    while run_event.is_set():  # while the thread is active we loop
        ret, image = video_capture.read()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray, 0)

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

            if SPRITES[5]:
                if is_mouth_open:
                    apply_sprite(image, "./sprites/rainbow.png", w0, x0, y0, incl, ontop=False)

            if SPRITES[6]:
                (left_x5, left_y5, left_w5, left_h5) = get_face_boundbox(shape, 3)
                (right_x5, right_y5, right_w5, right_h5) = get_face_boundbox(shape, 4)
                apply_sprite(image, "./sprites/eye.png", w // 6, left_x5, left_y5, incl, ontop=False)
                apply_sprite(image, "./sprites/eye.png", w // 6, right_x5, right_y5, incl, ontop=False)

        # OpenCV == BGR; PIL == RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        main_panel.configure(image=image)
        main_panel.image = image

    video_capture.release()


# Initialize GUI object
root = Tk()
root.title("Not really Snapchat")

BTNS = []
for (id, name) in enumerate(ALL_SPRITES):
    def do_toggle_sprite(id):
        return lambda: toggle_sprite(id)

    btn = Button(root, text=name, command=do_toggle_sprite(id))
    btn.pack(side="top", fill="both", expand="no", padx="5", pady="5")
    BTNS.append(btn)

main_panel = Label(root)
main_panel.pack(padx=10, pady=10)

SPRITES = [False] * len(BTNS)

# Creates a thread for openCV processing
run_event = threading.Event()
run_event.set()
action = Thread(target=cvloop, args=(run_event,))
action.setDaemon(True)
action.start()


# Function to clean everything up
def terminate():
    global root, run_event, action
    print("Cleaning up OpenCV resources...")
    run_event.clear()
    time.sleep(1)
    # action.join() #strangely in Linux this thread does not terminate properly, so .join never finishes
    root.destroy()
    print("All closed!")


# When the GUI is closed it actives the terminate function
root.protocol("WM_DELETE_WINDOW", terminate)
root.mainloop()  # creates loop of GUI
