import requests

def send_message(number, text):
    url = "http://localhost:3000/send-message"
    payload = {
        "number": number,
        "text": text
    }
    response = requests.post(url, json=payload)
    print(response.text)


for  i in range(100):
    send_message("+919746445097", "I love you anju"+i*"u")   