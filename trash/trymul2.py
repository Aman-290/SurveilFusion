from ultralytics import YOLO
import cv2
import math 
import telepot
import time
import numpy as np
import asyncio

from multiprocessing import Process, Queue
import signal

# Sentinel value for resetting the queue
RESET_QUEUE_SENTINEL = "RESET_QUEUE"

# Function to handle smartphone detection and message sending
def send_message_process(queue):
    bot = telepot.Bot('6890031926:AAGvKK-V8hJTW3NP_I8M2NN7yrsnRXZxYwA')

    while True:
        # Get the message from the queue
        message = queue.get()

        # Check if it's the sentinel value to reset the queue
        if message == RESET_QUEUE_SENTINEL:
            queue = Queue()  # Reset the queue
        else:
            # Send the message and image
            bot.sendMessage(1400097718, message)
            f = open('kang.jpg', 'rb')
            bot.sendPhoto(1400097718, f)

# Function to open and read from the webcam
def webcam_process(queue):
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    model = YOLO("yolo-Weights/yolov8n.pt")
    classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
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

    smartphone_detected = False

    while True:
        success, img = cap.read()
        results = model(img, stream=True)

        for r in results:
            boxes = r.boxes

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                confidence = math.ceil((box.conf[0] * 100)) / 100
                print("Confidence --->", confidence)

                cls = int(box.cls[0])
                print("Class name -->", classNames[cls])

                if classNames[cls] == "cell phone":
                    smartphone_detected = True

                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

        cv2.imshow('Webcam', img)

        if smartphone_detected:
            queue.put("New message👇⚠️\nsmartphone! on ")
            cv2.imwrite('kang.jpg', img)
            f = open('kang.jpg', 'rb')
            queue.put(f.read())  # Put the image data into the queue
            smartphone_detected = False

            # Put the sentinel value to reset the queue
            queue.put(RESET_QUEUE_SENTINEL)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()

# Main process to control the other two processes
if __name__ == '__main__':
    message_queue = Queue()

    # Start the webcam process
    webcam_proc = Process(target=webcam_process, args=(message_queue,))
    webcam_proc.start()

    # Start the send message process
    send_message_proc = Process(target=send_message_process, args=(message_queue,))
    send_message_proc.start()

    try:
        # Wait for processes to finish
        webcam_proc.join()
        send_message_proc.join()
    except KeyboardInterrupt:
        # Handle Ctrl+C to gracefully terminate the processes
        print("Terminating processes...")
        webcam_proc.terminate()
        send_message_proc.terminate()
        webcam_proc.join()
        send_message_proc.join()

    cv2.destroyAllWindows()
