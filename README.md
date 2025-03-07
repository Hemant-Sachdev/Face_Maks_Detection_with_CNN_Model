# Face Mask Detection
  This repository contains a project that detects whether a person is wearing a mask or not using a Convolutional Neural Network (CNN) built with TensorFlow and 
  Keras. The dataset is sourced from Kaggle.
# Test Images output
  ![image](https://github.com/user-attachments/assets/4c3b2d5b-3a96-4792-8fac-598fb87e7b9c)
  ![image](https://github.com/user-attachments/assets/c9b2600e-3735-484c-b9fa-c6d38879f946)

## Getting Started

## Open in Colab
You can run this project in Google Colab. To do this, click the link below:
    
 ## Prerequisites
    # Make sure you have kaggle installed. You can install it using the following command:
    !pip install kaggle

 ## Setting Up Kaggle API
    # Create a directory for the Kaggle API key:
    !mkdir -p ~/.kaggle
    
    # Copy your kaggle.json file to the newly created directory:
    !cp kaggle.json ~/.kaggle/
    
    # Set the permissions of the file:
    !chmod 600 ~/.kaggle/kaggle.json
    
 ## Downloading the Dataset
    # Download the face mask dataset from Kaggle:
    !kaggle datasets download -d omkargurav/face-mask-dataset
    
 ## Extracting the Dataset
    # Unzip the downloaded dataset:
    from zipfile import ZipFile
    data = '/content/face-mask-dataset.zip'

    with ZipFile(data, 'r') as zip:
      zip.extractall()
      print("The dataset has been extracted")
      
 ## Listing Files
    # List the contents of the current directory:
    !ls
  
 ## Importing Necessary Libraries
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import cv2
    from google.colab.patches import cv2_imshow
    from PIL import Image
    from sklearn.model_selection import train_test_split
    import tensorflow as tf
    from tensorflow import keras
  
 ## Preparing the Dataset
    # List and count images with and without masks:
    
    files_with_mask = os.listdir('/content/data/with_mask')
    files_without_mask = os.listdir('/content/data/without_mask')

    print(len(files_with_mask), ": Are with Mask")
    print(len(files_without_mask), ": Are without Mask")
  
 ## Labeling the Data

    labels_with_mask = [1]*len(files_with_mask)
    labels_without_mask = [0]*len(files_without_mask)

    final_labels = labels_with_mask + labels_without_mask

 ## Image Processing
    # Resize images and convert them to numpy arrays:
    new_data = []
    path_with_mask = '/content/data/with_mask/'

    for img_file in files_with_mask:
      image = Image.open(path_with_mask + img_file)
      image = image.resize((130,130))
      image = image.convert('RGB')
      image = np.array(image)
      new_data.append(image)

    path_without_mask = '/content/data/without_mask/'

    for img_file in files_without_mask:
      image = Image.open(path_without_mask + img_file)
      image = image.resize((130,130))
      image = image.convert('RGB')
      image = np.array(image)
      new_data.append(image)

    X = np.array(new_data)
    y = np.array(final_labels)

 ## Splitting the Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

 ## Scaling the Data
    scaled_X_train = X_train / 255
    scaled_X_test = X_test / 255

 ## Building the Model
    model = keras.Sequential([
      keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(130, 130, 3)),
      keras.layers.MaxPooling2D(pool_size=(2,2)),
      keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu'),
      keras.layers.MaxPooling2D(pool_size=(2,2)),
      keras.layers.Flatten(),
      keras.layers.Dense(256, activation='relu'),
      keras.layers.Dropout(0.5),
      keras.layers.Dense(128, activation='relu'),
      keras.layers.Dropout(0.5),
      keras.layers.Dense(64, activation='relu'),
      keras.layers.Dropout(0.5),
      keras.layers.Dense(2, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

 ## Training the Model

    his = model.fit(scaled_X_train, y_train, validation_split=0.1, epochs=5)

 ## Evaluating the Model
    loss, accuracy = model.evaluate(scaled_X_test, y_test)
    print('Test Accuracy : ', accuracy)

 ## Plotting the Results
    h = his
    plt.plot(h.history['loss'], label='Train Loss')
    plt.plot(h.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.show()

    plt.plot(h.history['accuracy'], label='Train Accuracy')
    plt.plot(h.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.show()

 ## Prediction
    path_of_input_image = input('Provide the image path for Prediction: ')
    in_image = cv2.imread(path_of_input_image)
    cv2_imshow(in_image)

    in_image_resize = cv2.resize(in_image, (130,130))
    scaled_in_image = in_image_resize / 255
    in_image_reshape = np.reshape(scaled_in_image, [1, 130, 130, 3])

    in_pred = model.predict(in_image_reshape)
    in_pred_label = np.argmax(in_pred)

    if in_pred_label == 1:
      print("The Person is Wearing a Mask")
    else:
      print("The person is not Wearing a Mask")

 ## License
  This project is licensed under the MIT License - see the LICENSE file for details.

 ## Acknowledgments
  Kaggle for providing the dataset.
  Google Colab for providing the environment to run this project.


