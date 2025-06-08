import subprocess
import re
import os

def run_powershell_command(command, directory):
    # Change directory
    os.chdir(directory)
    # Execute the command
    process = subprocess.Popen(["powershell", "-Command", command], stdout=subprocess.PIPE)
    result = process.communicate()[0]
    return result.decode("utf-8")

# Directory where the PowerShell command will be executed
directory = r"C:\Users\amans\Desktop\cloudflar"
# Run the PowerShell command
output = run_powershell_command(".\cloudflared-windows-amd64.exe tunnel --url http://127.0.0.1:5000/", directory)

# Use regular expressions to extract the website link
website_link = re.search(r"https://\S+", output)
if website_link:
    website_link = website_link.group()
    print("Website link:", website_link)
else:
    print("Website link not found in the output.")