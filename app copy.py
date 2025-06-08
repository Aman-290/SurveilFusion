from flask import Flask, render_template, Response, request, redirect, url_for
from ultralytics import YOLO
import cv2
import math 
import telepot
import time
import numpy as np
import asyncio
from sound_detection_pack import keras_yamnet
from datetime import datetime
from multiprocessing import Process, Queue, Manager
import sys
# from flask_tunnel import FlaskTunnel
# import signal

import numpy as np
import pyaudio
import wave
import time
import os
import telepot
from datetime import datetime

from pydub import AudioSegment

from matplotlib import pyplot as plt
import pandas as pd
import sounddevice as sd
import threading

from keras_yamnet import params
from keras_yamnet.yamnet import YAMNet, class_names
from keras_yamnet.preprocessing import preprocess_input

from plot import Plotter
import requests
from datetime import datetime

from HostedLink.runTerminal import GetHostedLink

import face_recognition
import cv2
import numpy as np
import time
import os
from multiprocessing import Process, Queue, Manager
from telepot.namedtuple import InlineKeyboardMarkup
import telepot
from telepot.loop import MessageLoop
from dotenv import load_dotenv

app = Flask(__name__)
# FlaskTunnel(app, auth="/put your key", subdomain="")


# Load environment variables from .env file
load_dotenv()

def checkface(shared_vars):
    current_image_index = 0
    
    def send_initial_image(bot, chat_id):
        nonlocal current_image_index
        sendtel_image(bot, chat_id, current_image_index)

    def sendtel_image(bot, chat_id, image_index):
        if os.path.exists(f'images/image{image_index}.jpg'):
            keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [dict(text='add unknown person', callback_data=f'add-person_{image_index}')]
            ])
            bot.sendPhoto(chat_id, open(f'images2/image{image_index}.jpg', 'rb'),caption="Unkown person/people detected!", reply_markup=keyboard)
        else:
            bot.sendMessage(chat_id, "Image doesn't exist.")

    def on_chat_message(msg):
        content_type, chat_type, chat_id = telepot.glance(msg)

    def on_callback_query(msg):
        query_id, from_id, query_data = telepot.glance(msg, flavor='callback_query')
        if query_data.startswith('add-person_'):
            image_index = int(query_data.split('_')[1])
            # Get the face encodings from the image
            unknown_image = face_recognition.load_image_file(f'images/image{image_index}.jpg')
            unknown_face_encodings = face_recognition.face_encodings(unknown_image)
            
            # Initialize a list to store the face encodings of unknown faces in the image
            unknown_faces_encodings = []
            
            # Compare each face encoding in the image with the known face encodings
            for unknown_face_encoding in unknown_face_encodings:
                # Check if the face encoding matches any known face encoding
                matches = face_recognition.compare_faces(known_face_encodings, unknown_face_encoding)
                if not any(matches):
                    # If no match is found, add the face encoding to the list of unknown faces
                    unknown_faces_encodings.append(unknown_face_encoding)
            
            # Add the unknown face encodings to the list of known face encodings
            known_face_encodings.extend(unknown_faces_encodings)
            
            # Assign random names to the newly added faces
            for _ in range(len(unknown_faces_encodings)):
                new_name = f"New_{len(known_face_encodings)}"
                known_face_names.append(new_name)
                bot.getUpdates()
            
            # Update the known face list in the bot's memory (or any other necessary actions)
            # Determine the message based on the number of new unknown faces added
            if len(unknown_faces_encodings) == 1:
                message = f"Added {len(unknown_faces_encodings)} new person to the list."
            else:
                message = f"Added {len(unknown_faces_encodings)} new people to the list."
            
            # Update the known face list in the bot's memory (or any other necessary actions)
            bot.sendMessage(from_id, message)
        # def on_callback_query(msg):
    #     query_id, from_id, query_data = telepot.glance(msg, flavor='callback_query')
    #     if query_data.startswith('add_person_'):
    #         image_index = int(query_data.split('_')[1])
    #         # i want to add that unknown person to the known list with a random name key assigned, but I don't know how?
    #         # also, if there are multiple person in the list, only the the unknown person has to be added to the known person list, "knwon_face_encoding and known_face_names" 

    print("Starting ImageBot")

    # try:
    bot = telepot.Bot(os.getenv("TELEGRAM_BOT_TOKEN_CHECKFACE"))
    print("going to call messageLoop")
    MessageLoop(bot, {'chat': on_chat_message, 'callback_query': on_callback_query}).run_as_thread()
    print("message loop called")
        # while True:
        #     pass
    # except KeyboardInterrupt:
    #     print("Stopping ImageBot")
    # Load a sample picture and learn how to recognize it.
    krish_image = face_recognition.load_image_file("Krish/ref2.jpg")
    krish_face_encoding = face_recognition.face_encodings(krish_image)[0]

    # Load a second sample picture and learn how to recognize it.
    bradley_image = face_recognition.load_image_file("Bradley/bradley.jpg")
    bradley_face_encoding = face_recognition.face_encodings(bradley_image)[0]

    # Create arrays of known face encodings and their names
    known_face_encodings = [
        krish_face_encoding,
        bradley_face_encoding
    ]
    known_face_names = [
        "Aman",
        "Bradley"
    ]

    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []

    process_this_frame = True
    last_check_time = time.time()
    # bot = telepot.Bot('/put your key')

    while True:
        if shared_vars.image is not None:  # Check every 1 second
            print("shared var image is not None proceeding with image check")
            frame = shared_vars.image
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]

            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)
                #do something here or somewhere to add the unknwon person to a list or something which the bot can use afterwards on call back to add the person to the list, i don't know how to do it? 
            # Save frame if unknown face is found
            if "Unknown" in face_names:
                for (top, right, bottom, left), name in zip(face_locations, face_names):
                    if name == "Unknown":
                        # Scale up face locations since the frame we detected in was scaled to 1/4 size
                        # Save the frame as a JPEG image                        
                        if time.time() - last_check_time >= 10:  # Check every 1 second
                            last_check_time = time.time()
                            # try:
                            current_image_index += 1
                            cv2.imwrite(f"images/image{current_image_index}.jpg", frame)
                            for (top, right, bottom, left), name in zip(face_locations, face_names):
                                top *= 4
                                right *= 4
                                bottom *= 4
                                left *= 4
                                # Draw a box around the face
                                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                                # Draw a label with a name below the face
                                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                                font = cv2.FONT_HERSHEY_DUPLEX
                                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
                            cv2.imwrite(f"images2/image{current_image_index}.jpg", frame)
                            try:
                                send_initial_image(bot, 1400097718)
                                send_message("+919946295010", ".\n.\n.\n.\n.\n.\n")
                                send_message("+919946295010", "Unknown Person Identified!")                                
                                unknowonpath = rf"C:\Users\amans\Desktop\capstone web\images2\image{current_image_index}.jpg"
                                send_image("+919946295010", unknowonpath, "👤check telegram to add person")
                            except:
                                print("some error occured")


            shared_vars.image = None
            print ("shared var is set none")
        time.sleep(0.5)
    

# Function to send a text message
def send_message(number, text):
    url = "http://localhost:3000/send-message"
    payload = {
        "number": number,
        "text": text
    }
    response = requests.post(url, json=payload)
    print(response.text)

# Function to send an image
def send_image(number, image_path, caption):
    url = "http://localhost:3000/send-image"
    payload = {
        "number": number,
        "imagePath": image_path,
        "caption": caption
    }
    response = requests.post(url, json=payload)
    print(response.text)

# Function to send an audio
def send_audio(number, audio_path, caption):
    url = "http://localhost:3000/send-audio"
    payload = {
        "number": number,
        "audioPath": audio_path,
        "caption": caption
    }
    response = requests.post(url, json=payload)
    print(response.text)

    
# Get the current date and time
current_datetime = datetime.now()
formatted_datetime = current_datetime.strftime("Date:%d-%m-%Y\nTime:%I:%M:%S %p")
message = "New message👇⚠️\nFire Detected! on " + formatted_datetime + "\nlocation identified as👇⚠️\nhttps://www.google.com/maps/place/12.972477,79.164777"

valid_credentials = {"Aman@290": "Aman@290"}

def CallSoundDetect(shared_vars):
    HostedLink = shared_vars.hostlink
    Detected_sound = False
    bot = telepot.Bot(os.getenv("TELEGRAM_BOT_TOKEN_SOUND"))
    p_flag = False
    def record():
        nonlocal Detected_sound
        nonlocal  p_flag
        nonlocal HostedLink
        while True:
            FORMAT = pyaudio.paInt16  # Format of audio samples (16-bit signed integers)
            CHANNELS = 1              # Number of audio channels (1 for mono, 2 for stereo)
            RATE = 44100              # Sample rate (samples per second)
            CHUNK = 1024              # Number of frames per buffer
            RECORD_SECONDS = 15       # Duration of recording in seconds

            p = pyaudio.PyAudio()
            p_flag = False

            stream = p.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            frames_per_buffer=CHUNK)

            print("Recording 1 started...")

            frames = []
            start_time2 = time.time()
            while True:
                data = stream.read(CHUNK)
                frames.append(data)
                if time.time() - start_time2 >= 12:
                    break
                if Detected_sound:
                    start_time = time.time()  # Start the timer when sound is detected
                    while True:
                        data = stream.read(CHUNK)
                        frames.append(data)
                        if time.time() - start_time >= 2:  # Check if 2 seconds have passed since sound detection
                            volume_factor = 1.5   # Multiply recorded audio by this factor
                            WAVE_OUTPUT_FILENAME = "recorded_audio.wav"

                            if os.path.exists(WAVE_OUTPUT_FILENAME):
                                # Get the duration of the WAV file
                                with wave.open(WAVE_OUTPUT_FILENAME, 'rb') as wf:
                                    duration = wf.getnframes() / float(wf.getframerate())
                                    
                                # Check if duration is greater than 10 seconds
                                if duration > 10:
                                    os.remove(WAVE_OUTPUT_FILENAME)
                                    print(f"Deleted {WAVE_OUTPUT_FILENAME} as it exceeds 10 seconds.")

                            wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
                            # Set the parameters for the WAV file
                            wf.setnchannels(CHANNELS)
                            wf.setsampwidth(p.get_sample_size(FORMAT))
                            wf.setframerate(RATE)

                            # Convert the frames to a numpy array
                            frames_np = np.frombuffer(b''.join(frames), dtype=np.int16).copy()

                            # Convert volume_factor to the same data type as frames_np
                            volume_factor = frames_np.dtype.type(volume_factor)

                            # Increase the volume by multiplying with the volume factor
                            frames_np = (frames_np * volume_factor).astype(np.int16)

                            # Convert the numpy array back to bytes and write to the WAV file
                            wf.writeframes(frames_np.tobytes())

                            # Close the WAV file
                            wf.close()
                            print("Recording saved as", WAVE_OUTPUT_FILENAME)

                            p.terminate()
                            p_flag = True

                            audio = AudioSegment.from_wav(WAVE_OUTPUT_FILENAME)

                            if len(audio) < 7000:  # duration in milliseconds
                                print("Audio duration is less than 7 seconds. Saving as is.")
                                mp3_file = "recorded_audio.mp3"
                                audio.export(mp3_file, format="mp3")
                            else:
                                # Get the last 7 seconds of the audio
                                last_7_seconds = audio[-7000:]

                                # Convert to MP3
                                mp3_file = "recorded_audio.mp3"
                                last_7_seconds.export(mp3_file, format="mp3")

                                print("Conversion from WAV to MP3 complete.")
                            
                            try:
                                current_datetime = datetime.now()
                                formatted_datetime = current_datetime.strftime("Date:%d-%m-%Y\nTime:%I:%M:%S %p")
                                message = "New message!👇⚠️\n🔊!!SCREAM/THREAT DETECTED!!\n" + formatted_datetime + "\nlocation identified as👇⚠️\nhttps://www.google.com/maps/place/12.970670,79.164068"
                                f = open('recorded_audio.mp3', 'rb')
                                bot.sendAudio(1400097718, f, title="Last 7second Audio")
                                url = HostedLink
                                message22 = '<a href="{}">Click here</a> to see live view.'.format(url)
                                bot.sendMessage(1400097718, message22, parse_mode='HTML')
                                #whatsapp
                                send_message("+919946295010", ".\n.\n.\n.\n.\n.\n")
                                send_message("+919946295010", message)
                                screamaudio_path = r"C:\Users\amans\Desktop\capstone web\recorded_audio.mp3"
                                send_audio("+919946295010", screamaudio_path, "Audio caption")
                                send_message("+919946295010", "check live camera feed at :"+HostedLink)
                            except:
                                print("An exception occurred")

                            break
                    break  # Break out of the outer loop after 2 seconds have passed since sound detection

            print("Recording finished.")

            stream.stop_stream()
            stream.close()

            if not p_flag:
                p.terminate()

            Detected_sound = False



    def detect():
        nonlocal  Detected_sound
        ################### SETTINGS ###################
        plt_classes = [0,132,420,494,11,20] # Speech, Music, Explosion, Silence 
        class_labels=True
        FORMAT = pyaudio.paFloat32
        CHANNELS = 1
        RATE = params.SAMPLE_RATE
        WIN_SIZE_SEC = 0.975
        CHUNK = int(WIN_SIZE_SEC * RATE)
        RECORD_SECONDS = 500

        print(sd.query_devices())
        MIC = None

        #################### MODEL #####################
        
        model = YAMNet(weights='keras_yamnet/yamnet.h5')
        yamnet_classes = class_names('keras_yamnet/yamnet_class_map.csv')

        #################### STREAM ####################
        audio = pyaudio.PyAudio()

        # start Recording
        stream = audio.open(format=FORMAT,
                            input_device_index=MIC,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            frames_per_buffer=CHUNK)
        print("recording 2 started...")
        

        if plt_classes is not None:
            plt_classes_lab = yamnet_classes[plt_classes]
            n_classes = len(plt_classes)
        else:
            plt_classes = [k for k in range(len(yamnet_classes))]
            plt_classes_lab = yamnet_classes if class_labels else None
            n_classes = len(yamnet_classes)

        monitor = Plotter(n_classes=n_classes, FIG_SIZE=(12,6), msd_labels=plt_classes_lab)

        while True:
            # Waveform
            data = preprocess_input(np.fromstring(
                stream.read(CHUNK), dtype=np.float32), RATE)
            prediction = model.predict(np.expand_dims(data,0))[0]

            if prediction[11] > 0.05 and Detected_sound == False:  # Adjust the threshold as per your requirement
                print("Scream Detected")
                current_datetime = datetime.now()
                formatted_datetime = current_datetime.strftime("Date:%d-%m-%Y\nTime:%I:%M:%S %p")
                message = "New message!👇⚠️\n🔊!!SCREAM/THREAT DETECTED!!\n" + formatted_datetime + "\nlocation identified as👇⚠️\nhttps://www.google.com/maps/place/12.970670,79.164068"
                
                #telegram
                bot.sendMessage(1400097718, ".\n.\n.\n.\n.\n.\n")
                bot.sendMessage(1400097718, message)
                Detected_sound = True

            monitor(data.transpose(), np.expand_dims(prediction[plt_classes],-1))

        # print("finished recording")
        # # stop Recording
        # stream.stop_stream()
        # stream.close()
        # audio.terminate()

    def start_recording_and_detecting():
        record_thread = threading.Thread(target=record)
        detect_thread = threading.Thread(target=detect)

        record_thread.start()
        detect_thread.start()

        record_thread.join()
        detect_thread.join()

    # call the new function instead of record() or detect()
    start_recording_and_detecting()



# Create a semaphore with initial value 1
# whatsapp_semaphore = threading.Semaphore(1)

# def send_whatsapp_message_thread(filename, phone_number, message):
#     threading.Thread(target=send_whatsapp_message, args=(filename, phone_number, message)).start()


# Function to handle smartphone detection and message sending
def send_message_process(shared_vars,queue):
    HostedLink = shared_vars.hostlink
    bot = telepot.Bot(os.getenv("TELEGRAM_BOT_TOKEN_FIRE"))
    #whatsapp api functions
    while True:
        if not queue.empty():
            # Get the message from the queue
            message = queue.get()
            current_datetime = datetime.now()
            formatted_datetime = current_datetime.strftime("Date:%d-%m-%Y\nTime:%I:%M:%S %p")
            message = "New message!👇⚠️\n🔥!!FIRE DETECTED!!🔥\n" + formatted_datetime + "\nlocation identified as👇⚠️\nhttps://www.google.com/maps/place/12.970670,79.164068\n"

            try:
                # Send the message and image
                bot.sendMessage(1400097718, ".\n.\n.\n.\n.\n.\n")
                bot.sendMessage(1400097718, message)
                # bot.sendMessage(6001277574, message)
                f = open('kang.jpg', 'rb')
                bot.sendPhoto(1400097718, f)
                f.seek(0)
                url = HostedLink
                message22 = '<a href="{}">Click here</a> to see live view.'.format(url)
                bot.sendMessage(1400097718, message22, parse_mode='HTML')
                # Create the message with HTML markup for the hyperlink
                
                # bot.sendMessage(1400097718, message22)
                # Send the message with parse_mode='HTML' to enable HTML formatting
                # bot.sendMessage(1400097718, message22, parse_mode='HTML')
                # bot.sendPhoto(6001277574, f)
                send_message("+919946295010", ".\n.\n.\n.\n.\n.\n")
                send_message("+919946295010", message)
                fireimage_path = r"C:\Users\amans\Desktop\capstone web\kang.jpg"
                send_image("+919946295010", fireimage_path, "🔥Fire Detected Instance image")
                send_message("+919946295010", "check live camera feed at :"+HostedLink)
                # send_image("+919946295010", fireimage_path, "check live feed at :"+HostedLink)
            except:
                print("An exception occurred")

            shared_vars.imagesave = False

            if not queue.empty():
                queue.get()

            # time.sleep(2)
            # Reset the queue
            # queue.put(None)
        time.sleep(0.3)
# Function to open and read from the webcam


cap = cv2.VideoCapture(0)
def generate_frames(shared_vars,queue):
    
    cap.set(3, 640)
    cap.set(4, 480)

    model = YOLO("yolo-Weights/fire49mb.pt")
    classNames = ["firemen", "fire", "smoke", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                "teddy bear", "hair drier", "toothbrush"
                ]
    last_check_time = time.time()
    smartphone_detected = False
    while True:
        success, img = cap.read()
        results = model(img, stream=True)

        current_time = time.time()
        if current_time - last_check_time >= 1:  # Check every 1 second
            last_check_time = current_time
            if shared_vars.image is None:
                print("shared var is None")
                shared_vars.image = img
                print("shared var is set with frame")

        for r in results:
            boxes = r.boxes

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                confidence = math.ceil((box.conf[0] * 100)) / 100
                print("Confidence --->", confidence)
                cls = int(box.cls[0])
                print("Class name -->", classNames[cls])

                if classNames[cls] == "fire" and confidence > 0.82:
                    smartphone_detected = True

                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2
                if classNames[cls] == "fire" and confidence > 0.78:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                    cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

        # cv2.imshow('Webcam', img)

                
        resized_img = cv2.resize(img, (320, 240))
        ret, buffer = cv2.imencode('.jpg', resized_img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        if smartphone_detected:
            if shared_vars.imagesave == False:
                cv2.imwrite('kang.jpg', img) 
                shared_vars.imagesave = True
            if not queue.empty():
                queue.get()
            else:
                queue.put(message)
                smartphone_detected = False

        if cv2.waitKey(1) == ord('q'):
            sys.exit()

    cap.release()


def webcam_process(shared_vars,queue):
    generate_frames(shared_vars,queue)


    

@app.route('/')
def login_page():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username')
    password = request.form.get('password')

    if username in valid_credentials and password == valid_credentials[username]:
        return redirect(url_for('index'))
    else:
        # Invalid credentials, redirect back to the login page
        return redirect(url_for('login_page'))

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(shared_vars,message_queue), mimetype='multipart/x-mixed-replace; boundary=frame')

# Main process to control the other two processes
if __name__ == '__main__':

    manager = Manager()
    shared_vars = manager.Namespace()
    shared_vars.imagesave = False  # Initialize the shared variable
    shared_vars.hostlink = GetHostedLink()
    # shared_vars.hostlink = "https://pulling-schema-oil-uzbekistan.trycloudflare.com/"
    shared_vars.image = None
    gotlink = False
    # def forgetinglink():
    #     nonlocal gotlink
    #     if not gotlink:
    #         HostedLink = GetHostedLink()
    #         gotlink = True
    #     return HostedLink

    # HostedLink = forgetinglink()


    message_queue = Queue()
    
    sound_proc = Process(target=CallSoundDetect, args=(shared_vars,))
    sound_proc.start()

    # Start the webcam process
    webcam_proc = Process(target=webcam_process, args=(shared_vars,message_queue))
    webcam_proc.start()

    # Start the send message process
    send_message_proc = Process(target=send_message_process, args=(shared_vars,message_queue))
    send_message_proc.start()

    checkproc = Process(target=checkface, args=(shared_vars,))
    checkproc.start()

    # app.run(host='0.0.0.0', port=5000, debug=True)
    app.run(host='0.0.0.0',port=5000)

    

    try:
        # Wait for processes to finish
        sound_proc.join()
        webcam_proc.join()
        send_message_proc.join()
    except KeyboardInterrupt:
        # Handle Ctrl+C to gracefully terminate the processes
        print("Terminating processes...")
        sound_proc.terminate()
        webcam_proc.terminate()
        send_message_proc.terminate()
        
        sound_proc.join()
        webcam_proc.join()
        send_message_proc.join()

    cv2.destroyAllWindows()

