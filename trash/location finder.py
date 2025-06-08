import geocoder

def get_location():
    try:
        # Get location based on IP address
        location = geocoder.ip('me')

        # Print the details of the location
        print("Location details:")
        print("Latitude:", location.latlng[0])
        print("Longitude:", location.latlng[1])
        print("City:", location.city)
        print("Region:", location.region)
        print("Country:", location.country)

    except Exception as e:
        print("Error:", str(e))

if __name__ == "__main__":
    get_location()