Animals Skin Disease Detection

Description:

Animals Skin Disease Detection is a web application built using Streamlit as the front end and Python as the back end. It allows users to detect skin diseases in cats and dogs using machine learning models.

Features:

- Detect skin diseases in cats and dogs.
- Upload images of diseased animals for diagnosis.
- View diagnosis results.
- Capture images from the camera for real-time diagnosis.

Installation:

To run the application locally, follow these steps:

1. Clone this repository:
   - git clone <repository_url>
   - cd <repository_name>

2. Create a new virtual environment:
   - python -m venv my_venv

3. Activate the virtual environment:
   - source my_venv/bin/activate

4. Install dependencies using pip:
   - pip install -r requirements.txt

Usage:

1. Change directory to the codes folder:
   - cd codes

2. Run the Streamlit application with image insertion:
   - streamlit run main.py

3. Alternatively, if you want to use the camera feature, run:
   - streamlit run st_main.py

4. Follow the instructions on the Streamlit web interface to upload images or capture images from the camera for diagnosis.

5. To close the server, press Ctrl + C.

Folder Structure:

- codes: Contains Python scripts for the Streamlit application.
- images_to_check: Contains test images of diseased dogs and cats.
- temp_path: Contains images to store from the user interface.
- requirements.txt: Contains required packages and versions.
