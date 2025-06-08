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


Detected_sound = False
bot = telepot.Bot('6890031926:AAGvKK-V8hJTW3NP_I8M2NN7yrsnRXZxYwA')
p_flag = False

def record():
    global Detected_sound
    global p_flag
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
            if time.time() - start_time2 >= 10:
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
                            if duration > 7:
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
                        
                        current_datetime = datetime.now()
                        formatted_datetime = current_datetime.strftime("Date:%d-%m-%Y\nTime:%I:%M:%S %p")
                        message = "New message!👇⚠️\n😱!!SCREAM/THREAT DETECTED!!😱\n" + formatted_datetime + "\nlocation identified as👇⚠️\nhttps://www.google.com/maps/place/12.970670,79.164068"

                        bot.sendMessage(6001277574, message)
                        f = open('recorded_audio.mp3', 'rb')
                        bot.sendAudio(6001277574, f, title="Last 7second Audio")
                        break
                break  # Break out of the outer loop after 2 seconds have passed since sound detection

        print("Recording finished.")

        stream.stop_stream()
        stream.close()

        if not p_flag:
            p.terminate()

        Detected_sound = False



def detect():
    global Detected_sound
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
