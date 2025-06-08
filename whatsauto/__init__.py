import asyncio
from playwright.async_api import async_playwright
import threading

isSend = False

image_path = None
phone_number = None
message = None
def confSend(image_path1, phone_number1, message1):
    global isSend,image_path, phone_number, message
    print("called confSend")
    if isSend == False:
        image_path = image_path1
        phone_number = phone_number1
        message =  message1
        isSend = True

   
async def check_message_sent(page):
    while True:
        # Check if the element exists
        msg_time_element = await page.query_selector('span[data-icon="msg-time"]')

        if msg_time_element:
            print("Message is still pending. Waiting...")
            # Wait for a short interval before checking again
            await page.wait_for_timeout(1000)
        else:
            print("Message sent!")
            break


async def send_whatsapp_message():
    global isSend,image_path, phone_number, message
    async with async_playwright() as p:
        context = await p.firefox.launch_persistent_context(user_data_dir='./my-user-data-directory', headless=False)
        page = await context.new_page()
        await page.goto("https://web.whatsapp.com/")
        while True:
            if isSend:
                await page.click('span[data-icon="new-chat-outline"]')
                await page.click('p.selectable-text.copyable-text.iq0m558w.g0rxnol2')
                await page.keyboard.type(phone_number)
                await page.wait_for_timeout(1000)
                await page.keyboard.press('Enter')
                await page.wait_for_timeout(500)
                await page.keyboard.type(message)
                await page.wait_for_timeout(1500)
                await page.keyboard.press('Enter')
                await page.wait_for_timeout(1300)

                #open uploader
                await page.click('.bo8jc6qi.p4t1lx4y.brjalhku [data-icon="attach-menu-plus"]')

                # Paste the image and send
                await page.wait_for_timeout(1500)
                file_input_selector = 'input[type="file"][accept="image/*,video/mp4,video/3gpp,video/quicktime"][multiple]'# Adjust the selector based on the actual HTML structure
                await page.set_input_files(file_input_selector, image_path)
                await page.wait_for_timeout(1700)
                await page.keyboard.press('Enter')

                await page.wait_for_timeout(2300)

                await check_message_sent(page)

                await page.wait_for_timeout(500)
                # await page.close()
                isSend=False
def funforsendmsg():
    asyncio.run(send_whatsapp_message())

def initBrowser():
    thread = threading.Thread(target=funforsendmsg)
    thread.start()



# callBroswer()
# confSend('kang.jpg', '+919946295010', "hello")




