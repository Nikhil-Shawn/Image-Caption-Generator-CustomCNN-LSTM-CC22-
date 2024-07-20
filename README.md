# Image Caption Generator CustomCNN+LSTM (CC22)

This project focuses on creating a custom model for image captioning. It includes steps for downloading datasets, training the model, extracting image features, and generating captions for both dataset images and external images.

## Step-by-Step Summary

### 1. Downloading Datasets

- **Flickr8k Dataset**: [Kaggle Link](https://www.kaggle.com/datasets/adityajn105/flickr8k)
- **STL-10 Dataset**: [Kaggle Link](https://www.kaggle.com/datasets/jessicali9530/stl10)

### 2. Extracting Datasets

Extract the datasets to specific directories for further processing.

### 3. Model Creation and Compilation

- Created a custom CNN model for feature extraction.
- Compiled the model using the Adam optimizer and SparseCategoricalCrossentropy loss.

### 4. Training the Model

- Loaded the STL-10 dataset.
- Preprocessed the images.
- Trained the model using Early Stopping and Learning Rate Reduction callbacks.

### 5. Feature Extraction

- Extracted features using the custom model.
- Saved the extracted features in a pickle file.

### 6. Caption Processing

- Loaded captions from the Flickr8k dataset and created mappings.
- Cleaned the captions and tokenized the text.

### 7. Model for Image Captioning

- Created an encoder-decoder model using LSTM and CNN features.
- Trained the model with image-caption pairs from the Flickr8k dataset.

### 8. Caption Generation

- Generated captions for test images.
- Calculated BLEU scores to evaluate the model.

### 9. Download and Generate Captions for External Images

- Downloaded images from URLs.
- Extracted features for these images.
- Generated captions for the downloaded images.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Nikhil-Shawn/Image-Caption-Generator-CustomCNN-Model-With-LSTM.git
   cd image-captioning
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Download and Extract Datasets

Download the Flickr8k and STL-10 datasets from the provided Kaggle links and extract them into the `data/` directory.

### 2. Training the Model

Use the STL-10 dataset to train the model for recognizing the 10 classes. This model will be used for feature extraction.

```python
# Train the model on STL-10 dataset
python train_stl10.py
```

### 3. Feature Extraction

Extract features from the images using the trained model.

```python
# Extract features using the trained model
python extract_features.py
```

### 4. Image Captioning Model Training

Train the encoder-decoder model for image captioning using the Flickr8k dataset.

```python
# Train the image captioning model
python train_caption_model.py
```

### 5. Generate Captions

Generate captions for test images and evaluate the model.

```python
# Generate captions for test images
python generate_captions.py
```

### 6. Caption External Images

Download images from URLs, extract features, and generate captions.

```python
# Generate captions for external images
python caption_external_images.py --urls "url1,url2,url3"
```

## Results

- **Feature Extraction**: Saved in `features.pkl`.
- **Generated Captions**: Displayed in the console and saved in `captions.txt`.
- **Evaluation**: BLEU scores printed in the console.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.
