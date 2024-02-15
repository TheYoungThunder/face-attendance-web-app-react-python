import os
import string
import urllib
import uuid
import sys
import pickle
import datetime
import time
import shutil

# the steps below are done so I can import the test.py file properly, this is temporarily adding them to the PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the relative path to your folder
relative_path = os.path.join(current_dir, 'Silent-Face-Anti-Spoofing') 

if relative_path not in sys.path:
    sys.path.append(relative_path)
    print("added to the pythonpath")

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, UploadFile, Response
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from perform_test import test
import face_recognition
import starlette
from io import BytesIO

ATTENDANCE_LOG_DIR = './logs'
DB_PATH = './db'
for dir_ in [ATTENDANCE_LOG_DIR, DB_PATH]:
    if not os.path.exists(dir_):
        os.mkdir(dir_)

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_image_data(file: UploadFile) -> bytes:
    contents = file.file.read()
    return contents

# this is to handle face_reco requirements of images to be RGB, I need this after changing the way algo processes the images from disk to memory
def convert_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

@app.post("/login")
async def login(file: UploadFile = File(...)):

    # in the bottom code, the image is processed directly from memory which more performant and resource efficient
    # the bottom approach is good for deployment!!! I will keep the orignial approach as it's more useful for debugging
    # you can find the old approach in the oldApproach.py
    image_data = get_image_data(file)
    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), -1)
    
    # Convert image to RGB
    rgb_image = convert_to_rgb(image)

    label = test(
                image=rgb_image, model_dir='/home/thunder/face-attendance-web-app-react-python/backend/Silent-Face-Anti-Spoofing/resources/anti_spoof_models', device_id=0)

    print(label)
    if label == 1:
        user_name, match_status = recognize(rgb_image)

        if match_status:
            epoch_time = time.time()
            date = time.strftime('%Y%m%d', time.localtime(epoch_time))
            with open(os.path.join(ATTENDANCE_LOG_DIR, '{}.csv'.format(date)), 'a') as f:
                f.write('{},{},{}\n'.format(user_name, datetime.datetime.now(), 'IN'))
                f.close()

        return {'user': user_name, 'match_status': match_status}
    else:
        return {'match_status': False, "message": "failed spoof verification"}


@app.post("/logout")
async def logout(file: UploadFile = File(...)):

    image_data = get_image_data(file)
    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), -1)
    
    # Convert image to RGB
    rgb_image = convert_to_rgb(image)

    user_name, match_status = recognize(rgb_image)

    if match_status:
        epoch_time = time.time()
        date = time.strftime('%Y%m%d', time.localtime(epoch_time))
        with open(os.path.join(ATTENDANCE_LOG_DIR, '{}.csv'.format(date)), 'a') as f:
            f.write('{},{},{}\n'.format(user_name, datetime.datetime.now(), 'OUT'))
            f.close()

    return {'user': user_name, 'match_status': match_status}


@app.post("/register_new_user")
async def register_new_user(file: UploadFile = File(...), text=None):
    # Get the image data from the uploaded file
    image_data = get_image_data(file)
    
    # Decode the image data using OpenCV
    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), -1)

    # Convert the image to RGB format
    rgb_image = convert_to_rgb(image)

    # Extract face embeddings from the RGB image using face_recognition library
    embeddings = face_recognition.face_encodings(rgb_image)

    # Use open directly without creating a BytesIO object
    # Write the image data to a file in the DB_PATH directory with the specified filename
    with open(os.path.join(DB_PATH, '{}.png'.format(text)), 'wb') as f:
        f.write(image_data)

    # Continue with the rest of the registration logic

    # Open a file to save the face embeddings using the specified filename
    file_ = open(os.path.join(DB_PATH, '{}.pickle'.format(text)), 'wb')

    # Serialize and save the face embeddings to the file using pickle
    pickle.dump(embeddings, file_)

    # Print the provided text (filename or identifier)
    print(text)

    # Return a dictionary indicating the registration status
    return {'registration_status': 200}



@app.get("/get_attendance_logs")
async def get_attendance_logs():

    filename = 'out.zip'

    shutil.make_archive(filename[:-4], 'zip', ATTENDANCE_LOG_DIR)

    return starlette.responses.FileResponse(filename, media_type='application/zip', filename=filename)


def recognize(img):
    print("recognize function just ran")
    # it is assumed there will be at most 1 match in the db

    embeddings_unknown = face_recognition.face_encodings(img)
    if len(embeddings_unknown) == 0:
        return 'no_persons_found', False
    else:
        embeddings_unknown = embeddings_unknown[0]

    match = False
    j = 0

    db_dir = sorted([j for j in os.listdir(DB_PATH) if j.endswith('.pickle')])
    # db_dir = sorted(os.listdir(DB_PATH))    
    print("db_dir: " ,db_dir)
    while ((not match) and (j < len(db_dir))):

        path_ = os.path.join(DB_PATH, db_dir[j])
        print(path_)

        # imporvement, Handles opening and closing of files better, handles opening and closing the file properly
        with open(path_, 'rb') as file:
            embeddings_list = pickle.load(file)
            print(embeddings_list)

        print (embeddings_list)
        if embeddings_list:  # Check if the list is not empty
            print(embeddings_list)
            embeddings = embeddings_list[0]
            match = face_recognition.compare_faces([embeddings], embeddings_unknown)[0]
        else:
            print("empty embeddings_list")

        j += 1

    if match:
        return db_dir[j - 1][:-7], True
    else:
        return 'unknown_person', False
