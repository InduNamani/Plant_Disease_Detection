import cv2
import numpy as np
from PIL import Image

def extract_features(image_path):
    """
    Extract numerical features from an image
    
    Returns: 1D array of numbers representing the image
    """
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))  
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    features = []
    features.extend([
        np.mean(img_rgb[:,:,0]),  
        np.mean(img_rgb[:,:,1]),  
        np.mean(img_rgb[:,:,2])  
    ])
    features.extend([
        np.std(img_rgb[:,:,0]),
        np.std(img_rgb[:,:,1]),
        np.std(img_rgb[:,:,2])
    ])
    features.extend([
        np.mean(img_hsv[:,:,0]),  
        np.mean(img_hsv[:,:,1]),  
        np.mean(img_hsv[:,:,2])   
    ])
    
    # FEATURE 3: Texture Features
    # How much variation is in the image (texture = rough/smooth)
    features.append(np.std(img_gray))
    
    # Edge detection - diseases create more edges/boundaries
    edges = cv2.Canny(img_gray, 100, 200)
    features.append(np.sum(edges) / (128 * 128))  # Edge density
    
    # FEATURE 4: Color Histogram (distribution)
    # How many pixels of each color intensity
    hist_r = cv2.calcHist([img_rgb], [0], None, [32], [0, 256])
    hist_g = cv2.calcHist([img_rgb], [1], None, [32], [0, 256])
    hist_b = cv2.calcHist([img_rgb], [2], None, [32], [0, 256])
    
    # Flatten and normalize
    features.extend(hist_r.flatten() / (128 * 128))
    features.extend(hist_g.flatten() / (128 * 128))
    features.extend(hist_b.flatten() / (128 * 128))
    
    return np.array(features)


def preprocess_image_for_display(uploaded_file):
    """
    Convert uploaded Streamlit file to displayable format
    """
    image = Image.open(uploaded_file)
    return image
