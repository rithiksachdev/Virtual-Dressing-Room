import cv2, os
import dlib
from imutils import face_utils, rotate_bound
import math
import numpy as np

ACTIVE_IMAGES = [0 for i in range(10)]
SPRITES = [0 for i in range(10)]
IMAGES = {i: [] for i in range(10)}
PHOTOS = {i: [] for i in range(10)}

def initialize_images_and_photos(file_path):
    global IMAGES, PHOTOS
    idx_str_l = file_path.split('/')
    idx_str = idx_str_l[-1]
    idx = int(''.join(filter(str.isdigit, idx_str)))
    idx = (idx // 10) % 10
    if os.path.isfile(file_path):
        sprite_image = cv2.imread(file_path, -1)
        IMAGES[idx].append(sprite_image)
        photo = cv2.resize(sprite_image, (150, 100))
        PHOTOS[idx].append(photo) if idx in PHOTOS else PHOTOS.update({idx: [photo]})

def put_sprite(num, k):
    global SPRITES, ACTIVE_IMAGES
    SPRITES[num] = 1
    ACTIVE_IMAGES[num] = k if SPRITES[num] else None

def draw_sprite(frame, sprite, x_offset, y_offset):
    (h, w) = (sprite.shape[0], sprite.shape[1])
    (imgH, imgW) = (frame.shape[0], frame.shape[1])
    if y_offset + h >= imgH: sprite = sprite[0:imgH - y_offset, :, :]
    if x_offset + w >= imgW: sprite = sprite[:, 0:imgW - x_offset, :]
    if x_offset < 0: 
        sprite = sprite[:, abs(x_offset)::, :]
        x_offset = 0
    for c in range(3):
        frame[y_offset:y_offset + h, x_offset:x_offset + w, c] = \
            sprite[:, :, c] * (sprite[:, :, 3] / 255.0) + \
            frame[y_offset:y_offset + h, x_offset:x_offset + w, c] * (1.0 - sprite[:, :, 3] / 255.0)
    return frame

def adjust_sprite2head(sprite, head_width, head_ypos, ontop=True):
    factor = 1.0 * head_width / sprite.shape[1]
    sprite = cv2.resize(sprite, (0, 0), fx=factor, fy=factor)
    y_orig = head_ypos - sprite.shape[0] if ontop else head_ypos
    if y_orig < 0: sprite = sprite[abs(y_orig)::, :, :]
    return sprite, max(y_orig, 0)

def apply_Haar_filter(img, haar_cascade, scaleFact=1.05, minNeigh=3, minSizeW=30):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return haar_cascade.detectMultiScale(gray, scaleFactor=scaleFact, minNeighbors=minNeigh, minSize=(minSizeW, minSizeW), flags=cv2.CASCADE_SCALE_IMAGE)

def get_face_boundbox(points, face_part):
    mapping = {1: points[17:22], 2: points[22:27], 3: points[36:42], 4: points[42:48], 5: points[29:36], 6: points[1:17], 7: points[0:6], 8: points[11:17]}
    x, y, w, h = calculate_boundbox(mapping.get(face_part, []))
    return x, y, w, h

def calculate_inclination(point1, point2):
    return 180 / math.pi * math.atan((float(point2[1] - point1[1])) / (point2[0] - point1[0]))

def calculate_boundbox(list_coordinates):
    x, y, w, h = min(list_coordinates[:, 0]), min(list_coordinates[:, 1]), max(list_coordinates[:, 0]) - min(list_coordinates[:, 0]), max(list_coordinates[:, 1]) - min(list_coordinates[:, 1])
    return x, y, w, h

def get_category_number(file_path):
    parts = file_path.split('/')
    category_part = parts[-1]
    if category_part:
        val = int(''.join(filter(str.isdigit, category_part)))
        val = (val // 10) % 10
        return val
    else:
        return -1

def get_k(file_path):
    parts = file_path.split('/')
    category_part = parts[-1]
    if category_part:
        val = int(''.join(filter(str.isdigit, category_part)))
        val = val % 10
        return val
    else:
        return -1

def apply_sprite(frame, sprite, w, x, y, angle, ontop=True):
    # Since sprite is already an image (numpy array), we don't need cv2.imread
    # Just apply the sprite directly
    #print("Applying sprite:")
    sprite = rotate_bound(sprite, angle)
    (sprite, y_final) = adjust_sprite2head(sprite, w, y, ontop)
    frame = draw_sprite(frame, sprite, x, y_final)

def process_frame(frame, file_path):
    global SPRITES, ACTIVE_IMAGES, IMAGES
    sprite_applied = False
    # try:
    if isinstance(frame, bytes):
        nparr = np.frombuffer(frame, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is None or not hasattr(frame, 'shape'):
        print("Invalid frame")
        return None
    
    initialize_images_and_photos(file_path)

    (x,y,w,h) = (0,0,10,10) 
    detector = dlib.get_frontal_face_detector()
    #fullbody = cv2.CascadeClassifier('data/haarcascade_fullbody.xml')
    model = "data/shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(model)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)
    
    for face in faces: 
        (x,y,w,h) = (face.left(), face.top(), face.width(), face.height())
        #print("Face detected at:", face.left(), face.top(), face.width(), face.height())
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)
        incl = calculate_inclination(shape[17], shape[26])
        is_mouth_open = (shape[66][1] -shape[62][1]) >= 10
        index = get_category_number(file_path)
        k = get_k(file_path)
        #print("index, k", index, k)
        put_sprite(index, k)
        # print("sprites")
        # print(SPRITES)
        # print("images")
        # print(len(IMAGES))
        # print(len(IMAGES[1]))
        
        if SPRITES[3]:#Tiara
            apply_sprite(frame, IMAGES[3][ACTIVE_IMAGES[3]],w+45,x-20,y+20, incl, ontop = True)
            sprite_applied = True

        #Necklaces
        if SPRITES[1]:
            (x1,y1,w1,h1) = get_face_boundbox(shape, 6)
            if(len(IMAGES[1])>=2):
                apply_sprite(frame, IMAGES[1][ACTIVE_IMAGES[1]],w1,x1,y1+125, incl,  ontop = False)
                sprite_applied = True
        
        #Goggles
        if SPRITES[6]:
            (x3,y3,_,h3) = get_face_boundbox(shape, 1)
            apply_sprite(frame, IMAGES[6][ACTIVE_IMAGES[6]],w,x,y3-5, incl, ontop = False)
            sprite_applied = True

        #Earrings
        (x0,y0,w0,h0) = get_face_boundbox(shape, 6) #bound box of mouth
        if SPRITES[2]:
            (x3,y3,w3,h3) = get_face_boundbox(shape, 7) #nose
            apply_sprite(frame, IMAGES[2][ACTIVE_IMAGES[2]],w3,x3-40,y3+30, incl,ontop=False)
            (x3,y3,w3,h3) = get_face_boundbox(shape, 8) #nose
            apply_sprite(frame, IMAGES[2][ACTIVE_IMAGES[2]],w3,x3+40,y3+75, incl) 
            sprite_applied = True

#            if SPRITES[5]:
#                apply_sprite(image, IMAGES[5][ACTIVE_IMAGES[5]],w,x,y, incl, ontop = True)
        
        

        #Tops
        if SPRITES[4]:
            # (x,y,w,h) = (0,0,10,10)
            # apply_sprite2feature(image, IMAGES[7][ACTIVE_IMAGES[7]], fullbody, w//4, 2*h//3, h//2, True, w//2, x, y, w, h)
            (x1,y1,w1,h1) = get_face_boundbox(shape, 8)
            apply_sprite(frame, IMAGES[4][ACTIVE_IMAGES[4]],w1+350,x1-230,y1+100, incl,  ontop = False)
            sprite_applied = True
    # IMAGES = {i: [] for i in range(10)}
    return frame if sprite_applied else None 
    # except Exception as e:
    #     print(f"An error occurred: {e}")
    #     return None
