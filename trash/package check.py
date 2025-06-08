import importlib.util
import pkg_resources

packages = {
    "flask": None,
    "opencv-python": None,
    "certifi": "2020.6.20",
    "chardet": "3.0.4",
    "click": "7.1.2",
    "cmake": "3.18.2.post1",
    "decorator": "4.4.2",
    "dlib": "19.18.0",
    "face-recognition": "1.3.0",
    "face-recognition-models": "0.3.0",
    "idna": "2.10",
    "imageio": "2.9.0",
    "imageio-ffmpeg": "0.4.2",
    "moviepy": "1.0.3",
    "numpy": "1.18.4",
    "opencv-python": "4.4.0.46",
    "Pillow": "8.0.1",
    "proglog": "0.1.9",
    "requests": "2.24.0",
    "tqdm": "4.51.0",
    "urllib3": "1.25.11",
    "wincertstore": "0.2"
}

installed_packages = []

for package, version in packages.items():
    try:
        spec = importlib.util.find_spec(package)
        if spec is not None:
            installed_version = pkg_resources.get_distribution(package).version
            if version is None:
                installed_packages.append((package, installed_version))
            elif installed_version == version:
                installed_packages.append((package, installed_version, "Installed with correct version"))
            else:
                installed_packages.append((package, installed_version, f"Installed with incorrect version {installed_version} (required {version})"))
    except Exception as e:
        print(f"An error occurred while checking {package}: {e}")

print("Installed packages:")
for package_info in installed_packages:
    if len(package_info) == 2:
        print(package_info[0])
    else:
        print(f"{package_info[0]} ({package_info[1]}): {package_info[2]}")
