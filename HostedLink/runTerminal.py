import subprocess
import re
import time

def GetHostedLink():
    # Start the cloudflared tunnel as a subprocess
    process = subprocess.Popen(
        ["cloudflared", "tunnel", "--url", "http://localhost:5000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )

    pattern = r"https://[a-zA-Z0-9-]+\.trycloudflare\.com"
    timeout = 30  # seconds
    start_time = time.time()

    for line in process.stdout:
        print(line.strip())  # Optional: See logs live
        match = re.search(pattern, line)
        if match:
            link = match.group()
            print(f"Hosted link: {link}")
            return link
        if time.time() - start_time > timeout:
            print("Timeout: Couldn't find the link.")
            break

    process.terminate()
    return "Couldn't get link due to network error"

# Run the function
# GetHostedLink()
