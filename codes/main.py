import streamlit as st
from PIL import Image
from cat import get_cat_result, get_cat_med
from dog import get_dog_result, get_dog_med 

def main():
    st.title("Animals Skin Disease Detection")

    # User selects either dog or cat
    animal_choice = st.radio("Select Animal:", ("Dog", "Cat"))

    uploaded_file = st.file_uploader(f"Upload a photo of the {animal_choice.lower()}")

    if uploaded_file is not None:
        img_path = save_uploaded_file(uploaded_file, animal_choice.lower())
        if animal_choice == "Dog":
            pred_class, confidence = get_dog_result(img_path)
            medicine = get_dog_med(pred_class)
        elif animal_choice == "Cat":
            pred_class, confidence = get_cat_result(img_path)
            medicine = get_cat_med(pred_class)
        
        display_result(pred_class, confidence, medicine)

def save_uploaded_file(uploaded_file, animal):
    img_path = f"/home/pi/Desktop/disease_detection_files/temp_path/temp_image.jpg"
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return img_path

def display_result(pred_class, confidence, medicine):
    st.text(f'Predicted class is: {pred_class}')
    st.text(f'Confidence level: {confidence}')
    st.text(f'Prescribed medicine: {medicine}')

if __name__ == "__main__":
    main()
