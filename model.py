from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

image_size=[224,224]
train_path='/content/drive/MyDrive/lion_tiger/animals/animals/animals'
inception=InceptionV3(input_shape=image_size+[3],weights='imagenet',include_top=False)
for layer in inception.layers:
  layer.trainable=False

folder=glob('/content/drive/MyDrive/lion_tiger/animals/animals/animals')
prediction=Dense(len(folder),activation='softmax')(x)
model=Model(inputs=inception.input,outputs=prediction)
model.summary()

from tensorflow.keras.layers import Dense

output_layer = Dense(61, activation='softmax')(x)  # ✅ Two neurons for one-hot labels
model = Model(inputs=inception.input, outputs=output_layer)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # ✅ Categorical loss


from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_data=ImageDataGenerator(rescale=1./255,
                              shear_range=0.2,
                              zoom_range=0.2,
                              horizontal_flip=True)
test_data=ImageDataGenerator(rescale=1./255)
training_set=train_data.flow_from_directory('/content/drive/MyDrive/lion_tiger/animals/animals/animals',
                                            target_size=(224,224),
                                            batch_size=16,
                                            class_mode='categorical')

testing_set=test_data.flow_from_directory('/content/drive/MyDrive/lion_tiger/animals/animals/animals',
                                            target_size=(224,224),
                                            batch_size=32,
                                            class_mode='categorical')

r = model.fit(
    training_set,
    validation_data=testing_set,
    epochs=10,
    steps_per_epoch=len(training_set),
    validation_steps=len(testing_set)
    )


from tensorflow.keras.models import load_model

model.save('model_inception_61.h5')

model = load_model('model_inception_61.h5')


# load the names in folder
import os

# Path to the dataset directory
dataset_path = r"/content/drive/MyDrive/lion_tiger/animals/animals/animals"

# Get all folder names inside the dataset path
folder_names = [f.name for f in os.scandir(dataset_path) if f.is_dir()]

# Print the folder names
print(folder_names)

# Load the model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load an image for testing
img_path = '/content/drive/MyDrive/lion_tiger/animals/animals/animals/crab/01c3fc5278.jpg'  # Replace with your image path
img = image.load_img(img_path, target_size=(224, 224))  # Resize to match InceptionV3 input size

# Convert the image to an array
img_array = image.img_to_array(img)

# Expand dimensions to match batch size format
img_array = np.expand_dims(img_array, axis=0)

# Normalize image (same as training)
from tensorflow.keras.applications.inception_v3 import preprocess_input
img_array = preprocess_input(img_array)  # Preprocess like InceptionV3

# Predict the class
predictions = model.predict(img_array)
print("Raw Predictions:", predictions)
class_names = ['seal', 'squirrel', 'turtle', 'woodpecker', 'snake', 'wolf', 'sheep', 'zebra', 'turkey', 'tiger', 
               'rhinoceros', 'panda', 'porcupine', 'raccoon', 'pig', 'otter', 'rat', 'possum', 'ox', 'orangutan', 
               'okapi', 'octopus', 'koala', 'hyena', 'lizard', 'moth', 'kangaroo', 'leopard', 'lion', 'mouse', 
               'horse', 'hornbill', 'goose', 'grasshopper', 'fox', 'hippopotamus', 'hare', 'goat', 'hedgehog', 
               'gorilla', 'hamster', 'elephant', 'eagle', 'chimpanzee', 'crab', 'deer', 'duck', 'dolphin', 'dog', 
               'donkey', 'crow', 'coyote', 'cow', 'cat', 'boar', 'beetle', 'bear', 'antelope', 'badger', 'bee', 'bison']

predicted_class = np.argmax(predictions, axis=1)[0]  # Get class index
print("Predicted Class:", class_names[predicted_class])


plt.imshow(img)
plt.axis('off')
plt.title(f'Predicted: {class_names[predicted_class]}')
plt.show()

# # wrong prediction because of overfitting or underfitting
img_array = preprocess_input(img_array)
training_set.class_indices  # Use this to get the correct mapping
import json

# Save during training
with open('class_indices.json', 'w') as f:
    json.dump(training_set.class_indices, f)

# Load during prediction
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)

# Reverse the mapping
idx_to_class = {v: k for k, v in class_indices.items()}
print("Predicted Class:", idx_to_class[predicted_class])
model = load_model('model_inception_61.h5')
from tensorflow.keras.preprocessing import image
import numpy as np

# Load an image for testing
img_path = '/content/drive/MyDrive/lion_tiger/animals/animals/animals/hedgehog/037fda7d6a.jpg'  # Replace with your image path
img = image.load_img(img_path, target_size=(224, 224))  # Resize to match InceptionV3 input size

# Convert the image to an array
img_array = image.img_to_array(img)

# Expand dimensions to match batch size format
img_array = np.expand_dims(img_array, axis=0)

# Normalize image (same as training)
from tensorflow.keras.applications.inception_v3 import preprocess_input
img_array = preprocess_input(img_array)  # Preprocess like InceptionV3

# Predict the class
predictions = model.predict(img_array)
print("Raw Predictions:", predictions)
class_names = ['seal', 'squirrel', 'turtle', 'woodpecker', 'snake', 'wolf', 'sheep', 'zebra', 'turkey', 'tiger', 
               'rhinoceros', 'panda', 'porcupine', 'raccoon', 'pig', 'otter', 'rat', 'possum', 'ox', 'orangutan', 
               'okapi', 'octopus', 'koala', 'hyena', 'lizard', 'moth', 'kangaroo', 'leopard', 'lion', 'mouse', 
               'horse', 'hornbill', 'goose', 'grasshopper', 'fox', 'hippopotamus', 'hare', 'goat', 'hedgehog', 
               'gorilla', 'hamster', 'elephant', 'eagle', 'chimpanzee', 'crab', 'deer', 'duck', 'dolphin', 'dog', 
               'donkey', 'crow', 'coyote', 'cow', 'cat', 'boar', 'beetle', 'bear', 'antelope', 'badger', 'bee', 'bison']

idx_to_class = {v: k for k, v in class_indices.items()}

predicted_class = np.argmax(predictions, axis=1)[0]
print("Predicted Class:", idx_to_class[predicted_class])


plt.imshow(img)
plt.axis('off')
plt.title(f'Predicted: {idx_to_class[predicted_class]}')
plt.show()

## correct answer