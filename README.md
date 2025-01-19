# Smart Waste Management System

This project implements a Smart Waste Management System using Convolutional Neural Networks (CNN) and Autoencoders to classify and detect anomalies in waste images.

## Project Structure

## Requirements

- Python 3.x
- TensorFlow
- scikit-learn
- matplotlib
- transformers
- Pillow

## Installation

1. Clone the repository:

   ```sh
   git clone https://github.com/yourusername/Smart_Waste_Management_System.git
   cd Smart_Waste_Management_System
   ```

2. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

1. **Dataset Preparation**: Place your dataset in the specified directory. Update the `dataset_dir` variable in the notebook with the path to your dataset.

2. **Training the CNN Model**: The CNN model is built and trained using the training data. The model is then saved to `cnn_model.h5`.

3. **Feature Extraction**: Features are extracted from the trained CNN model for both training and validation data.

4. **Training the Autoencoder**: An autoencoder is built and trained using the extracted features.

5. **Anomaly Detection**: The trained autoencoder is used to detect anomalies in new data.

6. **Evaluation**: The model's performance is evaluated using confusion matrices and classification reports.

## Notebook Overview

- **Imports**: The necessary libraries and modules are imported.
- **Dataset Path and Image Size**: The dataset path and categories are defined.
- **Model Training**: The CNN model is built and trained.
- **Feature Extraction**: Features are extracted using the trained CNN model.
- **Autoencoder Training**: The autoencoder is built and trained.
- **Anomaly Detection**: Anomalies are detected using the trained autoencoder.
- **Evaluation**: The model's performance is evaluated using confusion matrices and classification reports.

## Example Code

```python
# Train CNN model
cnn_model = build_cnn_model()
cnn_history = cnn_model.fit(
    train_generator,
    epochs=50,
    validation_data=validation_generator,
    callbacks=[lr_scheduler, early_stopping]
)
cnn_model.save('cnn_model.h5')

# Extract features
train_features = prepare_autoencoder_data(train_generator, cnn_model)
validation_features = prepare_autoencoder_data(validation_generator, cnn_model)

# Build and train the autoencoder
autoencoder = build_autoencoder(train_features.shape[1])
autoencoder_history = train_autoencoder(autoencoder, train_features, validation_features)

# Detect anomalies on new data
test_generator = datagen.flow_from_directory(
    'path/to/test/data',
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='categorical'
)

Make sure to update the repository URL and any other specific details as needed.
```
