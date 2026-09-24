# Enhanced Prediction Script for Cattle Breed Classification
# Supports ensemble predictions for higher accuracy

import os
import argparse
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as kimage
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_preprocess
import glob

IMG_SIZE = (224, 224)

# Each model was trained with a different preprocess_input (see cow_sih.py).
# Feeding the wrong range into a model silently degrades its predictions.
MODEL_PREPROCESS_FNS = {
    'mobilenet': mobilenet_preprocess,
    'efficientnet': lambda x: x,  # EfficientNetV2 has built-in rescaling
    'resnet': resnet_preprocess,
}

def load_labels(path="class_labels.pkl", fallback=None):
    """Load class labels from pickle file"""
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return fallback or ["Gir", "Sahiwal", "Red Sindhi", "Tharparkar"]

def load_single_model(model_path):
    """Load a single trained model"""
    if os.path.exists(model_path):
        print(f"✅ Loading model: {model_path}")
        try:
            # Try loading with custom objects for focal loss
            model = tf.keras.models.load_model(model_path, compile=False)
            return model
        except Exception as e:
            print(f"❌ Error loading {model_path}: {e}")
            return None
    else:
        print(f"❌ Model file not found: {model_path}")
        return None

def load_ensemble_models():
    """Load all available trained models for ensemble prediction"""
    model_files = {
        'mobilenet': 'best_mobilenet.h5',
        'efficientnet': 'best_efficientnet.h5',
        'resnet': 'best_resnet.h5'
    }
    
    models = {}
    for name, path in model_files.items():
        model = load_single_model(path)
        if model is not None:
            models[name] = model
    
    print(f"✅ Loaded {len(models)} models for ensemble prediction")
    return models

def preprocess_image_enhanced(img_path, model_name='mobilenet'):
    """Enhanced image preprocessing matching the transform each model was trained with"""
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"❌ File not found: {img_path}")

    # Load and resize image
    img = kimage.load_img(img_path, target_size=IMG_SIZE)
    x = kimage.img_to_array(img)

    # Enhanced preprocessing
    x = x.astype("float32")

    # Contrast enhancement (similar to training)
    x = tf.image.adjust_contrast(x, contrast_factor=1.2)

    # Model-specific normalization
    preprocess_fn = MODEL_PREPROCESS_FNS.get(model_name, lambda a: a / 255.0)
    x = preprocess_fn(x)

    # Add batch dimension
    x = np.expand_dims(x, axis=0)

    return x

def predict_single_model(model, img_path, class_labels, model_name='mobilenet'):
    """Make prediction using a single model"""
    try:
        x = preprocess_image_enhanced(img_path, model_name)
        probs = model.predict(x, verbose=0)
        idx = int(np.argmax(probs))
        conf = float(np.max(probs)) * 100.0
        return class_labels[idx], conf, probs[0]
    except Exception as e:
        print(f"❌ Error in single model prediction: {e}")
        return None, 0.0, None

def predict_ensemble(models, img_path, class_labels):
    """Make ensemble prediction using multiple models"""
    if len(models) == 0:
        raise ValueError("No models available for ensemble prediction")

    all_predictions = []
    individual_results = {}

    print(f"🔍 Making predictions with {len(models)} models...")

    for name, model in models.items():
        try:
            x = preprocess_image_enhanced(img_path, name)
            probs = model.predict(x, verbose=0)
            pred_idx = int(np.argmax(probs))
            conf = float(np.max(probs)) * 100.0
            pred_breed = class_labels[pred_idx]
            
            all_predictions.append(probs[0])
            individual_results[name] = {
                'breed': pred_breed,
                'confidence': conf,
                'probabilities': probs[0]
            }
            
            print(f"   {name:12}: {pred_breed:12} ({conf:.1f}%)")
            
        except Exception as e:
            print(f"❌ Error with model {name}: {e}")
            continue
    
    if len(all_predictions) == 0:
        raise ValueError("All models failed to make predictions")
    
    # Ensemble prediction - average probabilities (soft voting)
    ensemble_probs = np.mean(all_predictions, axis=0)
    ensemble_idx = int(np.argmax(ensemble_probs))
    ensemble_conf = float(np.max(ensemble_probs)) * 100.0
    ensemble_breed = class_labels[ensemble_idx]
    
    # Calculate confidence metrics
    prediction_variance = np.var([result['confidence'] for result in individual_results.values()])
    agreement_score = len([1 for result in individual_results.values() 
                          if result['breed'] == ensemble_breed]) / len(individual_results)
    
    return {
        'ensemble_breed': ensemble_breed,
        'ensemble_confidence': ensemble_conf,
        'ensemble_probabilities': ensemble_probs,
        'individual_results': individual_results,
        'prediction_variance': prediction_variance,
        'agreement_score': agreement_score * 100.0,
        'num_models': len(all_predictions)
    }

def predict_breed_enhanced(img_path, use_ensemble=True):
    """Enhanced breed prediction with ensemble support"""
    
    # Load class labels
    class_labels = load_labels()
    
    if use_ensemble:
        # Try ensemble prediction first
        models = load_ensemble_models()
        
        if len(models) > 0:
            try:
                result = predict_ensemble(models, img_path, class_labels)
                return result
            except Exception as e:
                print(f"❌ Ensemble prediction failed: {e}")
                print("🔄 Falling back to single model prediction...")
    
    # Fallback to single model prediction
    single_model_files = [('mobilenet', 'best_mobilenet.h5'), ('efficientnet', 'best_efficientnet.h5'),
                           ('resnet', 'best_resnet.h5')]

    for model_name, model_file in single_model_files:
        model = load_single_model(model_file)
        if model is not None:
            breed, conf, probs = predict_single_model(model, img_path, class_labels, model_name)
            if breed:
                return {
                    'ensemble_breed': breed,
                    'ensemble_confidence': conf,
                    'ensemble_probabilities': probs,
                    'individual_results': {'single_model': {'breed': breed, 'confidence': conf}},
                    'prediction_variance': 0.0,
                    'agreement_score': 100.0,
                    'num_models': 1
                }
    
    raise ValueError("No working models found for prediction")

def display_results(result):
    """Display prediction results in a formatted way"""
    print("\n" + "="*60)
    print("🐄 CATTLE BREED PREDICTION RESULTS")
    print("="*60)
    
    print(f"🎯 ENSEMBLE PREDICTION: {result['ensemble_breed']}")
    print(f"📊 CONFIDENCE: {result['ensemble_confidence']:.2f}%")
    print(f"🤝 MODEL AGREEMENT: {result['agreement_score']:.1f}%")
    print(f"📈 PREDICTION VARIANCE: {result['prediction_variance']:.2f}")
    print(f"🔢 MODELS USED: {result['num_models']}")
    
    print(f"\n📋 PROBABILITY DISTRIBUTION:")
    class_labels = load_labels()
    for i, (label, prob) in enumerate(zip(class_labels, result['ensemble_probabilities'])):
        bar_length = int(prob * 20)  # Scale to 20 characters
        bar = "█" * bar_length + "░" * (20 - bar_length)
        print(f"   {label:12}: {bar} {prob*100:.1f}%")
    
    if result['num_models'] > 1:
        print(f"\n🔍 INDIVIDUAL MODEL RESULTS:")
        for model_name, res in result['individual_results'].items():
            print(f"   {model_name:12}: {res['breed']:12} ({res['confidence']:.1f}%)")
    
    # Confidence interpretation
    conf = result['ensemble_confidence']
    agreement = result['agreement_score']
    
    print(f"\n💡 PREDICTION RELIABILITY:")
    if conf >= 90 and agreement >= 80:
        print("   🟢 VERY HIGH - Strong confidence with good model agreement")
    elif conf >= 75 and agreement >= 60:
        print("   🟡 HIGH - Good confidence with reasonable model agreement")
    elif conf >= 60:
        print("   🟠 MODERATE - Moderate confidence, consider multiple predictions")
    else:
        print("   🔴 LOW - Low confidence, prediction may be uncertain")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced cattle breed prediction with ensemble support")
    parser.add_argument("image_path", type=str, nargs="?", default=None,
                       help="Path to the cattle image")
    parser.add_argument("--ensemble", action="store_true", default=True,
                       help="Use ensemble prediction (default: True)")
    parser.add_argument("--single", action="store_true", default=False,
                       help="Force single model prediction")
    parser.add_argument("--batch", type=str, default=None,
                       help="Predict on batch of images in directory")
    
    args = parser.parse_args()
    
    if args.batch:
        # Batch prediction mode
        print(f"🔄 Running batch predictions on directory: {args.batch}")
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(args.batch, ext)))
        
        print(f"📁 Found {len(image_files)} images")
        
        for img_path in image_files:
            print(f"\n📸 Processing: {os.path.basename(img_path)}")
            try:
                result = predict_breed_enhanced(img_path, use_ensemble=not args.single)
                print(f"✅ Predicted: {result['ensemble_breed']} ({result['ensemble_confidence']:.1f}%)")
            except Exception as e:
                print(f"❌ Error: {e}")
    
    elif args.image_path:
        # Single image prediction
        try:
            result = predict_breed_enhanced(args.image_path, use_ensemble=not args.single)
            display_results(result)
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
    
    else:
        print("❌ Please provide an image path or use --batch for directory processing")
        print("Example: python enhanced_predict.py path/to/cattle/image.jpg")
        print("Example: python enhanced_predict.py --batch path/to/images/directory/")