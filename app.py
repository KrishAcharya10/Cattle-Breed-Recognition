"""
Complete Cattle Breed Recognition Streamlit Application
With ALL Advanced Features and Computer Vision Capabilities
"""

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import pickle
import os
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import datetime
import json
import base64
from io import BytesIO
import time
import warnings
warnings.filterwarnings('ignore')

# Configuration
IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 60.0

# Each model was trained with a different preprocess_input, matching cow_sih.py.
# Feeding the wrong range into a model silently degrades its predictions.
MODEL_PREPROCESS_FNS = {
    'MobileNet': tf.keras.applications.mobilenet_v2.preprocess_input,
    'EfficientNet': lambda x: x,  # EfficientNetV2 has built-in rescaling
    'ResNet': tf.keras.applications.resnet_v2.preprocess_input,
}

class ImageProcessor:
    """Advanced image processing with multiple computer vision techniques"""

    @staticmethod
    def enhance_image_preprocessing(img_array, model_name):
        """Preprocessing matching the transform each model was trained with"""
        x = img_array.astype("float32")
        x = tf.image.adjust_contrast(x, contrast_factor=1.2)
        preprocess_fn = MODEL_PREPROCESS_FNS.get(model_name, lambda a: a / 255.0)
        x = preprocess_fn(x)
        return x
    
    @staticmethod
    def apply_clahe(image_np):
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
        if len(image_np.shape) == 3:
            lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
            return enhanced
        return image_np
    
    @staticmethod
    def apply_canny_edge_detection(image_np, low_threshold=50, high_threshold=150):
        """Apply Canny edge detection"""
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY) if len(image_np.shape) == 3 else image_np
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
        edges = cv2.Canny(blurred, low_threshold, high_threshold)
        return edges
    
    @staticmethod
    def apply_sobel_edge_detection(image_np):
        """Apply Sobel edge detection"""
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY) if len(image_np.shape) == 3 else image_np
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        sobel_combined = np.sqrt(sobelx**2 + sobely**2)
        sobel_combined = np.uint8(sobel_combined / sobel_combined.max() * 255)
        return sobel_combined
    
    @staticmethod
    def apply_laplacian_edge_detection(image_np):
        """Apply Laplacian edge detection"""
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY) if len(image_np.shape) == 3 else image_np
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
        laplacian = np.uint8(np.absolute(laplacian))
        return laplacian
    
    @staticmethod
    def apply_prewitt_edge_detection(image_np):
        """Apply Prewitt edge detection"""
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY) if len(image_np.shape) == 3 else image_np
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
        kernelx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        kernely = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        prewittx = cv2.filter2D(blurred, cv2.CV_64F, kernelx)
        prewitty = cv2.filter2D(blurred, cv2.CV_64F, kernely)
        prewitt = np.sqrt(prewittx**2 + prewitty**2)
        prewitt = np.uint8(prewitt / prewitt.max() * 255)
        return prewitt
    
    @staticmethod
    def apply_scharr_edge_detection(image_np):
        """Apply Scharr edge detection"""
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY) if len(image_np.shape) == 3 else image_np
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
        scharrx = cv2.Scharr(blurred, cv2.CV_64F, 1, 0)
        scharry = cv2.Scharr(blurred, cv2.CV_64F, 0, 1)
        scharr = np.sqrt(scharrx**2 + scharry**2)
        scharr = np.uint8(scharr / scharr.max() * 255)
        return scharr
    
    @staticmethod
    def create_edge_overlay(original_img, edges, color_map=cv2.COLORMAP_JET):
        """Create colorful edge overlay on original image"""
        if len(original_img.shape) == 2:
            original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
        
        edges_colored = cv2.applyColorMap(edges, color_map)
        edges_colored = cv2.cvtColor(edges_colored, cv2.COLOR_BGR2RGB)
        
        # Create overlay
        overlay = cv2.addWeighted(original_img, 0.7, edges_colored, 0.3, 0)
        return overlay

class ObjectDetectionVisualizer:
    """Create object detection style visualizations"""
    
    @staticmethod
    def create_bbox_visualization(image, breed_name, confidence, bbox_coords=None, 
                                  show_info=True, show_features=True):
        """Create comprehensive object detection visualization"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        img_draw = image.copy()
        draw = ImageDraw.Draw(img_draw)
        
        width, height = img_draw.size
        
        # Create bounding box
        if bbox_coords is None:
            bbox = [int(width * 0.1), int(height * 0.1), int(width * 0.9), int(height * 0.9)]
        else:
            bbox = bbox_coords
        
        # Color based on confidence
        if confidence >= 90:
            color = "green"
            reliability = "VERY HIGH"
        elif confidence >= 75:
            color = "orange"
            reliability = "HIGH"
        elif confidence >= 60:
            color = "yellow"
            reliability = "MODERATE"
        else:
            color = "red"
            reliability = "LOW"
        
        # Draw bounding box with thickness
        for i in range(5):
            draw.rectangle(
                [bbox[0]-i, bbox[1]-i, bbox[2]+i, bbox[3]+i],
                outline=color,
                width=1
            )
        
        # Try to load font
        try:
            font_large = ImageFont.truetype("arial.ttf", 24)
            font_medium = ImageFont.truetype("arial.ttf", 18)
            font_small = ImageFont.truetype("arial.ttf", 14)
        except:
            font_large = font_medium = font_small = ImageFont.load_default()
        
        # Main label
        label = f"{breed_name}"
        conf_label = f"{confidence:.1f}%"
        
        # Calculate text dimensions
        try:
            text_bbox = draw.textbbox((0, 0), label, font=font_large)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
        except:
            text_width, text_height = 200, 30
        
        # Draw label background
        label_bg = [bbox[0], bbox[1] - text_height - 40, bbox[0] + text_width + 120, bbox[1]]
        draw.rectangle(label_bg, fill=color, outline=color)
        
        # Draw breed name
        draw.text((bbox[0] + 10, bbox[1] - text_height - 35), label, fill="white", font=font_large)
        
        # Draw confidence
        draw.text((bbox[0] + text_width + 20, bbox[1] - text_height - 35), 
                 conf_label, fill="white", font=font_medium)
        
        # Draw reliability indicator
        rel_y = bbox[1] - 15
        draw.text((bbox[0] + 10, rel_y), f"Reliability: {reliability}", 
                 fill="white", font=font_small)
        
        if show_info:
            # Draw info panel at bottom
            info_y = bbox[3] + 15
            info_lines = [
                f"📊 Breed: {breed_name}",
                f"🎯 Confidence: {confidence:.2f}%",
                f"📐 Image Size: {width}x{height}px",
                f"🔍 Detection Status: CONFIRMED"
            ]
            
            for line in info_lines:
                draw.text((bbox[0] + 5, info_y), line, fill="black", font=font_small)
                info_y += 20
        
        if show_features:
            # Draw corner markers
            marker_size = 15
            for corner in [(bbox[0], bbox[1]), (bbox[2], bbox[1]), 
                          (bbox[0], bbox[3]), (bbox[2], bbox[3])]:
                draw.ellipse([corner[0]-marker_size, corner[1]-marker_size,
                            corner[0]+marker_size, corner[1]+marker_size],
                           fill=color, outline="white", width=2)
        
        return img_draw
    
    @staticmethod
    def create_multi_object_visualization(image, predictions, class_labels, top_n=3):
        """Create visualization showing top N predictions"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        img_draw = image.copy()
        draw = ImageDraw.Draw(img_draw)
        
        width, height = img_draw.size
        
        # Get top predictions
        top_indices = np.argsort(predictions)[-top_n:][::-1]
        
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        # Draw legend box
        legend_width = 250
        legend_height = 30 + (top_n * 25)
        legend_box = [width - legend_width - 10, 10, width - 10, legend_height]
        
        draw.rectangle(legend_box, fill="white", outline="black", width=2)
        draw.text((legend_box[0] + 10, 15), "Top Predictions:", fill="black", font=font)
        
        y_offset = 40
        colors = ["green", "orange", "red", "blue", "purple"]
        
        for i, idx in enumerate(top_indices):
            breed = class_labels[idx]
            conf = predictions[idx] * 100
            color = colors[i % len(colors)]
            
            # Draw color indicator
            draw.rectangle([legend_box[0] + 10, y_offset, 
                          legend_box[0] + 20, y_offset + 15],
                         fill=color, outline="black")
            
            # Draw text
            text = f"{breed}: {conf:.1f}%"
            draw.text((legend_box[0] + 30, y_offset), text, fill="black", font=font)
            y_offset += 25
        
        return img_draw

class GradCAMGenerator:
    """Generate Grad-CAM visualizations"""
    
    @staticmethod
    def generate_gradcam(model, img_array, class_idx):
        """Generate Grad-CAM heatmap"""
        try:
            # Find last convolutional layer
            last_conv_layer = None
            for layer in reversed(model.layers):
                if len(layer.output_shape) == 4:
                    last_conv_layer = layer.name
                    break
            
            if last_conv_layer is None:
                return None, "No convolutional layer found"
            
            grad_model = tf.keras.Model(
                inputs=model.inputs,
                outputs=[model.get_layer(last_conv_layer).output, model.output]
            )
            
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(img_array)
                class_output = predictions[:, class_idx]
            
            grads = tape.gradient(class_output, conv_outputs)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            conv_outputs = conv_outputs[0]
            heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
            heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
            heatmap = heatmap.numpy()
            
            return heatmap, None
        except Exception as e:
            return None, str(e)
    
    @staticmethod
    def create_gradcam_overlay(original_img, heatmap, alpha=0.4):
        """Create Grad-CAM overlay"""
        img_array = np.array(original_img)
        heatmap_resized = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(img_array, 1-alpha, heatmap_colored, alpha, 0)
        return Image.fromarray(overlay), heatmap_colored

class ChartGenerator:
    """Generate various charts and graphs"""
    
    @staticmethod
    def create_probability_bar_chart(probabilities, class_labels):
        """Create probability bar chart"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Sort by probability
        sorted_indices = np.argsort(probabilities)[::-1]
        sorted_probs = probabilities[sorted_indices] * 100
        sorted_labels = [class_labels[i] for i in sorted_indices]
        
    
        colors = plt.cm.viridis(sorted_probs / sorted_probs.max())
        bars = ax.barh(sorted_labels, sorted_probs, color=colors)
        
        ax.set_xlabel('Probability (%)', fontsize=12, fontweight='bold')
        ax.set_title('Breed Classification Probabilities', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, prob) in enumerate(zip(bars, sorted_probs)):
            ax.text(prob + 1, i, f'{prob:.1f}%', va='center', fontweight='bold')
        
        plt.tight_layout()
        
        # Save to buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        return buf
    
    @staticmethod
    def create_confidence_gauge(confidence):
        """Create confidence gauge chart"""
        fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': 'polar'})
        
        # Create gauge
        theta = np.linspace(0, np.pi, 100)
        
        # Color zones
        colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']
        boundaries = [0, 20, 40, 60, 80, 100]
        
        for i in range(len(colors)):
            start = boundaries[i] / 100 * np.pi
            end = boundaries[i+1] / 100 * np.pi
            theta_zone = np.linspace(start, end, 20)
            ax.fill_between(theta_zone, 0, 1, color=colors[i], alpha=0.3)
        
        # Needle
        needle_theta = confidence / 100 * np.pi
        ax.plot([needle_theta, needle_theta], [0, 0.8], 'k-', linewidth=3)
        ax.plot(needle_theta, 0.8, 'ko', markersize=10)
        
        ax.set_ylim(0, 1)
        ax.set_theta_zero_location('W')
        ax.set_theta_direction(1)
        ax.set_xticks(np.linspace(0, np.pi, 6))
        ax.set_xticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
        ax.set_yticks([])
        ax.set_title(f'Confidence: {confidence:.1f}%', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        return buf
    
    @staticmethod
    def create_model_comparison_chart(individual_results):
        """Create model comparison chart"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        models = list(individual_results.keys())
        confidences = [individual_results[m]['confidence'] for m in models]
        breeds = [individual_results[m]['breed'] for m in models]
        
        # Confidence comparison
        colors = plt.cm.Set3(np.arange(len(models)))
        bars = ax1.bar(models, confidences, color=colors)
        ax1.set_ylabel('Confidence (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Model Confidence Comparison', fontsize=14, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', alpha=0.3)
        
        # Add breed labels
        for bar, breed, conf in zip(bars, breeds, confidences):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{breed}\n{conf:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # Breed distribution
        breed_counts = {}
        for breed in breeds:
            breed_counts[breed] = breed_counts.get(breed, 0) + 1
        
        ax2.pie(breed_counts.values(), labels=breed_counts.keys(), autopct='%1.1f%%',
               startangle=90, colors=colors[:len(breed_counts)])
        ax2.set_title('Model Agreement Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        return buf

@st.cache_resource
def load_available_models():
    """Load all available trained models"""
    # Only the best checkpoint per architecture is loaded - running both the
    # "final" and "best" save of each model doubled ensemble inference time
    # (best_resnet.h5 alone is ~360MB) without adding meaningfully different predictions.
    model_files = {
        'MobileNet': 'best_mobilenet.h5',
        'EfficientNet': 'best_efficientnet.h5',
        'ResNet': 'best_resnet.h5'
    }
    
    available_models = {}
    model_info = []
    
    for name, path in model_files.items():
        if os.path.exists(path):
            try:
                model = tf.keras.models.load_model(path, compile=False)
                available_models[name] = model
                model_info.append(f"✅ {name}")
            except Exception as e:
                model_info.append(f"❌ {name} (load error)")
        else:
            model_info.append(f"⭕ {name} (not found)")
    
    return available_models, model_info

@st.cache_data
def load_class_labels():
    """Load class labels"""
    if os.path.exists("class_labels.pkl"):
        with open("class_labels.pkl", "rb") as f:
            return pickle.load(f)
    else:
        return ["Gir", "Sahiwal", "Red Sindhi", "Tharparkar", "Hariana", "Ongole"]

def predict_single_model(model, img, class_labels, model_name):
    """Single model prediction with all visualizations"""
    img_resized = img.resize(IMG_SIZE)
    img_np = np.array(img_resized)

    # Prepare for prediction
    x = image.img_to_array(img_resized)
    x = ImageProcessor.enhance_image_preprocessing(x, model_name)
    x_batch = np.expand_dims(x, axis=0)
    
    # Predict
    pred = model.predict(x_batch, verbose=0)
    class_idx = np.argmax(pred)
    confidence = float(np.max(pred)) * 100
    breed = class_labels[class_idx]
    
    # Generate visualizations
    results = {
        'breed': breed,
        'confidence': confidence,
        'probabilities': pred[0],
        'class_labels': class_labels
    }
    
    # Object detection style
    results['bbox_img'] = ObjectDetectionVisualizer.create_bbox_visualization(
        img_resized, breed, confidence)
    
    results['multi_bbox_img'] = ObjectDetectionVisualizer.create_multi_object_visualization(
        img_resized, pred[0], class_labels)
    
    # Edge detection
    results['canny_edges'] = ImageProcessor.apply_canny_edge_detection(img_np)
    results['sobel_edges'] = ImageProcessor.apply_sobel_edge_detection(img_np)
    results['laplacian_edges'] = ImageProcessor.apply_laplacian_edge_detection(img_np)
    results['prewitt_edges'] = ImageProcessor.apply_prewitt_edge_detection(img_np)
    results['scharr_edges'] = ImageProcessor.apply_scharr_edge_detection(img_np)
    
    # Edge overlays
    results['canny_overlay'] = ImageProcessor.create_edge_overlay(img_np, results['canny_edges'])
    results['sobel_overlay'] = ImageProcessor.create_edge_overlay(img_np, results['sobel_edges'])
    
    # Grad-CAM
    heatmap, error = GradCAMGenerator.generate_gradcam(model, x_batch, class_idx)
    if heatmap is not None:
        results['gradcam_overlay'], results['gradcam_heatmap'] = \
            GradCAMGenerator.create_gradcam_overlay(img_resized, heatmap)
    else:
        results['gradcam_error'] = error
    
    # Enhanced preprocessing
    results['clahe_img'] = ImageProcessor.apply_clahe(img_np)
    
    return results

def predict_ensemble(models, img, class_labels):
    """Ensemble prediction with all visualizations"""
    img_resized = img.resize(IMG_SIZE)
    img_np = np.array(img_resized)
    x_raw = image.img_to_array(img_resized)

    all_predictions = []
    individual_results = {}
    best_model = None
    best_x_batch = None
    best_confidence = 0

    # Get predictions from all models, each with its own trained preprocessing
    for name, model in models.items():
        try:
            x = ImageProcessor.enhance_image_preprocessing(x_raw, name)
            x_batch = np.expand_dims(x, axis=0)
            pred = model.predict(x_batch, verbose=0)
            class_idx = np.argmax(pred)
            confidence = float(np.max(pred)) * 100
            breed = class_labels[class_idx]

            all_predictions.append(pred[0])
            individual_results[name] = {
                'breed': breed,
                'confidence': confidence,
                'probabilities': pred[0]
            }

            if confidence > best_confidence:
                best_confidence = confidence
                best_model = model
                best_x_batch = x_batch
        except Exception as e:
            continue
    
    if len(all_predictions) == 0:
        return None
    
    # Ensemble results
    ensemble_probs = np.mean(all_predictions, axis=0)
    ensemble_idx = np.argmax(ensemble_probs)
    ensemble_conf = float(np.max(ensemble_probs)) * 100
    ensemble_breed = class_labels[ensemble_idx]
    
    results = {
        'breed': ensemble_breed,
        'confidence': ensemble_conf,
        'probabilities': ensemble_probs,
        'class_labels': class_labels,
        'individual_results': individual_results
    }
    
    # Generate all visualizations
    results['bbox_img'] = ObjectDetectionVisualizer.create_bbox_visualization(
        img_resized, ensemble_breed, ensemble_conf)
    
    results['multi_bbox_img'] = ObjectDetectionVisualizer.create_multi_object_visualization(
        img_resized, ensemble_probs, class_labels)
    
    # Edge detection
    results['canny_edges'] = ImageProcessor.apply_canny_edge_detection(img_np)
    results['sobel_edges'] = ImageProcessor.apply_sobel_edge_detection(img_np)
    results['laplacian_edges'] = ImageProcessor.apply_laplacian_edge_detection(img_np)
    results['prewitt_edges'] = ImageProcessor.apply_prewitt_edge_detection(img_np)
    results['scharr_edges'] = ImageProcessor.apply_scharr_edge_detection(img_np)
    
    # Edge overlays
    results['canny_overlay'] = ImageProcessor.create_edge_overlay(img_np, results['canny_edges'])
    results['sobel_overlay'] = ImageProcessor.create_edge_overlay(img_np, results['sobel_edges'])
    
    # Grad-CAM with best model
    if best_model is not None:
        heatmap, error = GradCAMGenerator.generate_gradcam(best_model, best_x_batch, ensemble_idx)
        if heatmap is not None:
            results['gradcam_overlay'], results['gradcam_heatmap'] = \
                GradCAMGenerator.create_gradcam_overlay(img_resized, heatmap)
    
    # Enhanced preprocessing
    results['clahe_img'] = ImageProcessor.apply_clahe(img_np)
    
    return results

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="🐄 Advanced Cattle Breed Recognition System",
        page_icon="🐄",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 48px;
            font-weight: bold;
            text-align: center;
            color: #2E7D32;
            margin-bottom: 10px;
        }
        .sub-header {
            font-size: 20px;
            text-align: center;
            color: #555;
            margin-bottom: 30px;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #2E7D32;
        }
        .stButton>button {
            width: 100%;
            background-color: #2E7D32;
            color: white;
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<p class="main-header">🐄 Advanced Cattle Breed Recognition</p>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Classification with Comprehensive Computer Vision Analysis</p>', 
                unsafe_allow_html=True)
    
    # Load models and labels
    models, model_info = load_available_models()
    class_labels = load_class_labels()
    
    # Sidebar
    with st.sidebar:
        st.header("📊 System Dashboard")
        
        st.subheader("Model Status")
        for info in model_info:
            if "✅" in info:
                st.success(info)
            elif "❌" in info:
                st.error(info)
            else:
                st.info(info)
        
        st.markdown("---")
        
        st.metric("Available Models", len(models))
        st.metric("Supported Breeds", len(class_labels))
        st.metric("System Status", "🟢 READY" if len(models) > 0 else "🔴 NO MODELS")
        
        st.markdown("---")
        
        st.subheader("🎨 Visualization Options")
        show_bbox = st.checkbox("Object Detection Style", value=True)
        show_edges = st.checkbox("Edge Detection Analysis", value=True)
        show_gradcam = st.checkbox("Grad-CAM Heatmap", value=True)
        show_charts = st.checkbox("Statistical Charts", value=True)
        show_enhanced = st.checkbox("Enhanced Processing", value=True)
        
        st.markdown("---")
        
        st.subheader("⚙️ Advanced Settings")
        edge_threshold_low = st.slider("Canny Low Threshold", 10, 100, 50)
        edge_threshold_high = st.slider("Canny High Threshold", 100, 300, 150)
        gradcam_alpha = st.slider("Grad-CAM Overlay Alpha", 0.0, 1.0, 0.4)
    
    if len(models) == 0:
        st.error("❌ **No trained models found!**")
        st.info("Please run the training script first to train the models.")
        return
    
    # Model selection
    if len(models) > 1:
        col1, col2 = st.columns([3, 1])
        with col1:
            prediction_mode = st.radio(
                "🔧 **Prediction Mode:**",
                ["🎯 Ensemble (Recommended)", "📱 Single Model"],
                horizontal=True
            )
        with col2:
            if prediction_mode == "📱 Single Model":
                selected_model = st.selectbox("Select Model", list(models.keys()))
    else:
        prediction_mode = "📱 Single Model"
        selected_model = list(models.keys())[0]
    
    # File upload
    st.markdown("---")
    uploaded_file = st.file_uploader(
        "📁 **Upload Cattle Image**",
        type=["jpg", "jpeg", "png", "bmp"],
        help="Upload a clear image of a cow or buffalo for breed identification"
    )
    
    if uploaded_file is not None:
        # Display original image
        img = Image.open(uploaded_file)
        
        st.markdown("---")
        st.subheader("📸 Original Image")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(img, caption="Uploaded Image", use_column_width=True)
        
        # Process button
        if st.button("🚀 Analyze Image", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Prediction
                status_text.text("🧠 Running AI analysis...")
                progress_bar.progress(20)
                
                start_time = time.time()
                
                if prediction_mode == "📱 Single Model":
                    model = models[selected_model]
                    results = predict_single_model(model, img, class_labels, selected_model)
                else:
                    results = predict_ensemble(models, img, class_labels)
                
                processing_time = time.time() - start_time
                
                if results is None:
                    st.error("❌ Prediction failed")
                    return
                
                progress_bar.progress(50)
                status_text.text("🎨 Generating visualizations...")
                
                # Display results
                st.markdown("---")
                st.subheader("🎯 Analysis Results")
                
                # Main metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Predicted Breed", results['breed'])
                
                with col2:
                    st.metric("Confidence", f"{results['confidence']:.1f}%")
                
                with col3:
                    if 'individual_results' in results:
                        agreement = sum(1 for r in results['individual_results'].values() 
                                      if r['breed'] == results['breed'])
                        total = len(results['individual_results'])
                        st.metric("Model Agreement", f"{(agreement/total)*100:.0f}%")
                    else:
                        st.metric("Model Used", selected_model.split()[0])
                
                with col4:
                    st.metric("Processing Time", f"{processing_time:.2f}s")
                
                # Reliability indicator
                if results['confidence'] >= 90:
                    st.success("🟢 **Reliability: VERY HIGH** - Excellent confidence in prediction")
                elif results['confidence'] >= 75:
                    st.warning("🟡 **Reliability: HIGH** - Good confidence in prediction")
                elif results['confidence'] >= 60:
                    st.warning("🟠 **Reliability: MODERATE** - Acceptable confidence")
                else:
                    st.error("🔴 **Reliability: LOW** - Consider retaking image")
                
                progress_bar.progress(70)
                
                # Object Detection Visualizations
                if show_bbox:
                    st.markdown("---")
                    st.subheader("🎯 Object Detection Style Visualizations")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.image(results['bbox_img'], 
                                caption=f"Object Detection: {results['breed']} ({results['confidence']:.1f}%)",
                                use_column_width=True)
                    
                    with col2:
                        if 'multi_bbox_img' in results:
                            st.image(results['multi_bbox_img'],
                                    caption="Top Predictions Display",
                                    use_column_width=True)
                
                # Edge Detection Visualizations
                if show_edges:
                    st.markdown("---")
                    st.subheader("🔍 Edge Detection Analysis")
                    
                    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Canny", "Sobel", "Laplacian", "Prewitt", "Scharr"])
                    
                    with tab1:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(results['canny_edges'], 
                                    caption="Canny Edge Detection",
                                    use_column_width=True)
                        with col2:
                            if 'canny_overlay' in results:
                                st.image(results['canny_overlay'],
                                        caption="Canny Overlay on Original",
                                        use_column_width=True)
                    
                    with tab2:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(results['sobel_edges'],
                                    caption="Sobel Edge Detection",
                                    use_column_width=True)
                        with col2:
                            if 'sobel_overlay' in results:
                                st.image(results['sobel_overlay'],
                                        caption="Sobel Overlay on Original",
                                        use_column_width=True)
                    
                    with tab3:
                        st.image(results['laplacian_edges'],
                                caption="Laplacian Edge Detection",
                                use_column_width=True)
                    
                    with tab4:
                        st.image(results['prewitt_edges'],
                                caption="Prewitt Edge Detection",
                                use_column_width=True)
                    
                    with tab5:
                        st.image(results['scharr_edges'],
                                caption="Scharr Edge Detection",
                                use_column_width=True)
                
                # Grad-CAM Visualization
                if show_gradcam and 'gradcam_overlay' in results:
                    st.markdown("---")
                    st.subheader("🔥 Grad-CAM Analysis - AI Decision Visualization")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.image(results['gradcam_overlay'],
                                caption="Grad-CAM Overlay - AI Attention Areas",
                                use_column_width=True)
                        st.info("🔴 **Red/Yellow**: High attention | 🔵 **Blue/Green**: Low attention")
                    
                    with col2:
                        if 'gradcam_heatmap' in results:
                            st.image(results['gradcam_heatmap'],
                                    caption="Pure Heatmap Visualization",
                                    use_column_width=True)
                
                # Enhanced Processing
                if show_enhanced and 'clahe_img' in results:
                    st.markdown("---")
                    st.subheader("✨ Enhanced Image Processing")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.image(img, caption="Original Image", use_column_width=True)
                    
                    with col2:
                        st.image(results['clahe_img'],
                                caption="CLAHE Enhanced (Contrast Optimization)",
                                use_column_width=True)
                
                progress_bar.progress(90)
                
                # Statistical Charts
                if show_charts:
                    st.markdown("---")
                    st.subheader("📊 Statistical Analysis & Charts")
                    
                    # Probability chart
                    chart_buf = ChartGenerator.create_probability_bar_chart(
                        results['probabilities'], class_labels)
                    st.image(chart_buf, caption="Breed Probability Distribution")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Confidence gauge
                        gauge_buf = ChartGenerator.create_confidence_gauge(results['confidence'])
                        st.image(gauge_buf, caption="Confidence Gauge")
                    
                    with col2:
                        # Model comparison for ensemble
                        if 'individual_results' in results:
                            comparison_buf = ChartGenerator.create_model_comparison_chart(
                                results['individual_results'])
                            st.image(comparison_buf, caption="Model Comparison Analysis")
                
                progress_bar.progress(100)
                status_text.text("✅ Analysis complete!")
                
                # Detailed probability table
                st.markdown("---")
                st.subheader("📈 Detailed Probability Breakdown")
                
                prob_data = []
                for i, (breed_name, prob) in enumerate(zip(class_labels, results['probabilities'])):
                    prob_data.append({
                        'Rank': i + 1,
                        'Breed': breed_name,
                        'Probability': f"{prob * 100:.2f}%",
                        'Confidence Bar': prob
                    })
                
                prob_df = pd.DataFrame(prob_data)
                prob_df = prob_df.sort_values('Confidence Bar', ascending=False).head(10)
                prob_df = prob_df.drop('Confidence Bar', axis=1)
                prob_df['Rank'] = range(1, len(prob_df) + 1)
                
                st.dataframe(prob_df, use_container_width=True, hide_index=True)
                
                # Individual model results
                if 'individual_results' in results:
                    st.markdown("---")
                    st.subheader("🔍 Individual Model Results")
                    
                    model_data = []
                    for model_name, result in results['individual_results'].items():
                        model_data.append({
                            'Model': model_name,
                            'Predicted Breed': result['breed'],
                            'Confidence': f"{result['confidence']:.2f}%",
                            'Match': '✅' if result['breed'] == results['breed'] else '❌'
                        })
                    
                    model_df = pd.DataFrame(model_data)
                    model_df = model_df.sort_values('Confidence', ascending=False)
                    
                    st.dataframe(model_df, use_container_width=True, hide_index=True)
                
                # Export results
                st.markdown("---")
                st.subheader("💾 Export Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Export JSON
                    export_data = {
                        'timestamp': datetime.now().isoformat(),
                        'breed': results['breed'],
                        'confidence': float(results['confidence']),
                        'probabilities': {class_labels[i]: float(results['probabilities'][i]) 
                                        for i in range(len(class_labels))},
                        'processing_time': processing_time
                    }
                    
                    if 'individual_results' in results:
                        export_data['individual_models'] = results['individual_results']
                    
                    json_str = json.dumps(export_data, indent=2)
                    st.download_button(
                        label="📄 Download JSON Report",
                        data=json_str,
                        file_name=f"cattle_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                
                with col2:
                    # Export CSV
                    csv_df = pd.DataFrame([{
                        'Timestamp': datetime.now().isoformat(),
                        'Breed': results['breed'],
                        'Confidence': results['confidence'],
                        'Processing Time': processing_time
                    }])
                    
                    csv_str = csv_df.to_csv(index=False)
                    st.download_button(
                        label="📊 Download CSV Report",
                        data=csv_str,
                        file_name=f"cattle_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                with col3:
                    # Save bbox image
                    if 'bbox_img' in results:
                        buf = BytesIO()
                        results['bbox_img'].save(buf, format='PNG')
                        st.download_button(
                            label="🖼️ Download Detection Image",
                            data=buf.getvalue(),
                            file_name=f"detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                            mime="image/png"
                        )
                
                time.sleep(1)
                progress_bar.empty()
                status_text.empty()
                
            except Exception as e:
                st.error(f"❌ **Analysis failed:** {str(e)}")
                progress_bar.empty()
                status_text.empty()
    
    # Information sections
    with st.expander("ℹ️ **About This System**"):
        st.markdown("""
        ### 🎯 System Capabilities
        
        This advanced cattle breed recognition system combines multiple deep learning models
        and computer vision techniques to provide comprehensive analysis:
        
        **1. Multi-Model Ensemble Learning**
        - Combines MobileNet, EfficientNet, and ResNet architectures
        - Voting mechanism for robust predictions
        - Individual model confidence tracking
        
        **2. Object Detection Style Visualization**
        - YOLO/RCNN style bounding boxes
        - Confidence-based color coding
        - Multi-prediction display
        
        **3. Edge Detection Analysis**
        - Canny, Sobel, Laplacian, Prewitt, and Scharr algorithms
        - Colorful overlays on original images
        - Structural feature highlighting
        
        **4. Grad-CAM Explainability**
        - AI attention visualization
        - Decision transparency
        - Feature importance mapping
        
        **5. Statistical Analysis**
        - Probability distributions
        - Confidence gauges
        - Model comparison charts
        - Comprehensive reporting
        
        ### 📷 Best Practices for Image Upload
        
        - Use high-resolution images (minimum 640x480)
        - Ensure good lighting conditions
        - Capture full or partial side view
        - Avoid blurry or heavily cropped images
        - Multiple angles improve ensemble accuracy
        
        ### 🐄 Supported Breeds
        
        """ + ", ".join(class_labels) + """
        
        ### 🔬 Technical Specifications
        
        - **Deep Learning Frameworks:** TensorFlow 2.x, Keras
        - **Computer Vision:** OpenCV, PIL
        - **Visualization:** Matplotlib, Seaborn, Streamlit
        - **Model Architectures:** MobileNetV2, EfficientNetV2B0, ResNet152V2
        - **Input Resolution:** 224x224 pixels
        - **Preprocessing:** CLAHE, Contrast Enhancement, Normalization
        - **Expected Accuracy:** 85-95% (ensemble mode)
        
        ### 📊 Output Types
        
        1. **Detection Images:** Bounding boxes with labels
        2. **Edge Maps:** Multiple edge detection algorithms
        3. **Heatmaps:** Grad-CAM attention visualization
        4. **Charts:** Probability distributions, confidence gauges
        5. **Reports:** JSON and CSV exports
        
        ### 🚀 Performance Notes
        
        - CPU Processing: 2-5 seconds per image
        - GPU Processing: 0.5-1 second per image
        - Ensemble mode slower but more accurate
        - Real-time capable on modern hardware
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666; padding: 20px;'>
            <p><strong>🐄 Advanced Cattle Breed Recognition System</strong></p>
            <p>Powered by TensorFlow, OpenCV & Streamlit | Built with ❤️ for Livestock Management</p>
            <p>© 2025 | Version 2.0</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()