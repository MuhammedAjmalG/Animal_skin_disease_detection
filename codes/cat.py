#import libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras

# load model for cat
model_path = '/home/pi/Desktop/disease_detection_files/models/model_cat_95_80'
loaded_model = keras.layers.TFSMLayer(model_path, call_endpoint='serving_default')

# initialize classes
class_names = ['Cryptococcosis', 'Cuterebra (Botfly) Infestation', 'Ringworm', 'ear mites', 'felina acne', 'feline herperhives']

# initialize dictionary for medicine
cat_dis_med= {
"Cryptococcosis":" Amphotericin B, Ketoconazole, Fluconazole and Itraconazole",
"Cuterebra (Botfly) Infestation":"Thoroughly flushed with Sterile Saline, Debrided (if necessary)",
"Ringworm":"Itraconazole, Itrafungol, Sporanox, Onmel or Terbinafine",
"ear mites":"NexGard SPECTRA",
"felina acne":"Mupirocin (Muricin)",
"feline herperhives":"Famciclovir (Famvir)"
}
 
# function to take image path input and return result
def get_cat_result(image_path):
  img = tf.keras.utils.load_img(image_path, target_size=(180, 180))
  img_array = tf.keras.utils.img_to_array(img)
  img_array = tf.expand_dims(img_array, 0) # Create a batch

  predictions = loaded_model(img_array)
  score = tf.nn.softmax(predictions['outputs'])
  pred_class = class_names[np.argmax(score)]
  confidence = 100*np.max(score)
  return pred_class, confidence

# function to return the medicine for the class
def get_cat_med(pred_class):
  if pred_class in cat_dis_med.keys():
    medicine = cat_dis_med[pred_class]
  return medicine

if __name__ == '__main__':
  img_path = '/home/pi/Desktop/disease_detection_files/images_to_check/cat/ct_acne (1).jpeg'
  pred_class, confidence = get_cat_result(img_path)
  print(pred_class, confidence)
  medicine = get_cat_med(pred_class)
  print(medicine)
