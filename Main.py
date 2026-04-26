from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.applications import VGG16, InceptionV3, ResNet50
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalAveragePooling2D
import numpy as np
import os
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load models (do this once when starting the server)
def load_models():
    global final_model, vgg_model, inc_model, resnet_model, rbm1, rbm2, scaler, class_names
    
    # Load the final classifier model
    final_model = load_model("brain_tumor_final_model.h5")
    
    # Load the DBN pipeline components
    rbm1, rbm2, scaler = joblib.load("dbn_scaler_pipeline.pkl")
    
    # Load feature extractors
    def create_feature_model(base_model):
        output = GlobalAveragePooling2D()(base_model.output)
        return Model(inputs=base_model.input, outputs=output)

    vgg_model = create_feature_model(VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3)))
    inc_model = create_feature_model(InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3)))
    resnet_model = create_feature_model(ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3)))
    
    # Update with your actual class names
    class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']  

load_models()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path):
    """Load and preprocess an image for prediction"""
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_tumor(img_path):
    """Make a prediction on a single image"""
    # Preprocess the image
    img_array = preprocess_image(img_path)
    
    # Extract features
    vgg_features = vgg_model.predict(img_array)
    inc_features = inc_model.predict(img_array)
    resnet_features = resnet_model.predict(img_array)
    combined_features = np.concatenate([vgg_features, inc_features, resnet_features], axis=1)
    
    # Scale features
    scaled_features = scaler.transform(combined_features)
    
    # Apply DBN transformations
    rbm1_features = rbm1.transform(scaled_features)
    rbm2_features = rbm2.transform(rbm1_features)
    
    # Make prediction
    predictions = final_model.predict(rbm2_features)
    predicted_class = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0]))
    
    return class_names[predicted_class], confidence

@app.route('/predict', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            prediction, confidence = predict_tumor(filepath)
            return jsonify({
                'prediction': prediction,
                'confidence': confidence,
                'image_url': f'/uploads/{filename}'
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        finally:
            # Clean up (optional)
            pass
    
    return jsonify({'error': 'File type not allowed'}), 400

if __name__ == '__main__':
    # Create upload folder if it doesn't exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True, port=5000)
