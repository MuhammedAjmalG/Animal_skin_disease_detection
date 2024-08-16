#import libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras

# load model for cat
model_path = '/home/pi/Desktop/disease_detection_files/models/model_dog_83_73'
loaded_model = keras.layers.TFSMLayer(model_path, call_endpoint='serving_default')

# initialize classes
class_names = ['Blastomycosis', 'Cellulitis', 'myisis', 'ringworm']

# initialize medicine dictionary for dog
dog_dis_med= {  
"Blastomycosis": "Amphotericin B (Abelcet®), Ketoconazole",
"Cellulitis":"Cephalexin, Moxifloxacin, Nafcillin, and Vancomycin",
"myisis": "Nitenpyram 1 mg/kg PO administered every 24 hours",
"ringworm":"Miconazole (Micaved), Terbinafine (Lamisil), or Clotrimazole (Otomax or Otibiotic)"
}

# functin to take image path input and return result
def get_dog_result(image_path):
  img = tf.keras.utils.load_img(image_path, target_size=(180, 180))
  img_array = tf.keras.utils.img_to_array(img)
  img_array = tf.expand_dims(img_array, 0) # Create a batch

  predictions = loaded_model(img_array)
  score = tf.nn.softmax(predictions['outputs'])
  predicted_class = class_names[np.argmax(score)]
  confidence = 100 * np.max(score)

  return predicted_class, confidence

# function to return the medicine for the class
def get_dog_med(pred_class):
  if pred_class in dog_dis_med.keys():
    medicine = dog_dis_med[pred_class]
  return medicine
  
if __name__ == '__main__':
  img_path = '/home/pi/Desktop/disease_detection_files/images_to_check/dog/dg_cel.jpeg'
  pred_class, confidence = get_dog_result(img_path)
  print(pred_class, confidence)
  medicine = get_dog_med(pred_class)
  print(medicine)
