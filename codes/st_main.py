import streamlit as st
from PIL import Image
from cat import get_cat_result, get_cat_med
from dog import get_dog_result, get_dog_med  


def main():
    st.title("Animals Skin Disease Detection")

    # User selects either dog or cat
    animal_choice = st.radio("Select Animal:", ("Dog", "Cat"))

    if animal_choice == "Dog":
        img_path = take_photo("dog")
        if img_path:
            pred_class, confidence = get_dog_result(img_path)
            medicine = get_dog_med(pred_class)
            display_result(pred_class, confidence, medicine)

    elif animal_choice == "Cat":
        img_path = take_photo("cat")
        if img_path:
            pred_class, confidence = get_cat_result(img_path)
            medicine = get_cat_med(pred_class)
            display_result(pred_class, confidence, medicine)

def take_photo(animal):
    st.write(f"Take a photo of the {animal}:")
    img_file_buffer = st.camera_input(f"Take a photo of the {animal}")

    if img_file_buffer is not None:
        img_path = f"/home/pi/Desktop/disease_detection_files/temp_path/temp_image.jpg"
        img = Image.open(img_file_buffer)
        img.save(img_path)
        return img_path
    else:
        return None

def display_result(pred_class, confidence, medicine):
    st.text(f'Predicted class is: {pred_class}')
    st.text(f'Confidence level: {confidence}')
    st.text(f'Prescribed medicine: {medicine}')

if __name__ == "__main__":
    main()
