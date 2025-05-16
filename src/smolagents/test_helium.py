import helium
import os
import time
import sys
import PIL.Image
from io import BytesIO
from selenium import webdriver

WORKDIR=os.getenv('WORKDIR')
DATAPATH=os.path.join(WORKDIR, 'datasets/webarena/test_smolagents')
if not os.path.exists(DATAPATH):
    os.makedirs(DATAPATH)

def save_screenshot():
    """
    Save a screenshot of the current page.
    """
    print("Saving screenshot...")
    driver = helium.get_driver()
    # Get the current page title
    # Create a filename based on the title
    filename = f"{time.time()}.png"
    # Save the screenshot
    png_bytes = driver.get_screenshot_as_png()
    image = PIL.Image.open(BytesIO(png_bytes))
    image.save(os.path.join(DATAPATH, filename))
    print(f"Screenshot saved as {filename}")
    return image


def reset():
    print("Starting Chrome and go to login page...")
    url = "http://10.7.4.57:9083/admin"

    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--force-device-scale-factor=1")
    chrome_options.add_argument("--window-size=1000,1350")
    chrome_options.add_argument("--disable-pdf-viewer")
    chrome_options.add_argument("--window-position=0,0")

    driver = helium.start_chrome(url, headless=True, options=chrome_options)
    # save_screenshot()

    username = os.getenv('SHOPPING_ADMIN_USERNAME')
    password = os.getenv('SHOPPING_ADMIN_PASSWORD')
    print("Entering login info...")
    helium.write(username, into='Username')
    helium.write(password, into='Password')
    # save_screenshot()

    print("Clicking sign in...")
    helium.click('Sign in')
    time.sleep(3)
    image = save_screenshot()
    return driver, image

if __name__ == "__main__":
    # Check if the script is being run directly
    # If so, call the reset function
    # and save the screenshot
    # global  driver
    driver, image = reset()
    helium.click('REPORTS')
    time.sleep(3)
    image = save_screenshot()
    
