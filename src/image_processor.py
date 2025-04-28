import cv2
import os

def split_coin_image(image_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load {image_path}")
        return
    
    h, w, _ = img.shape

    offset = 70
    mid = (w - offset) // 2
    left = img[:, :mid]    
    right = img[:, mid:w-offset]  

    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # Save front and back images
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_front.jpg"), left)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_back.jpg"), right)

    print(f"Split {image_path} into front and back images.")

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Warning: Failed to load {image_path}")
        return None
    resized = cv2.resize(img, (256, 256))
    return resized

def preprocess_local_dir(input_dir, split_dir):
    images = {}
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(input_dir, filename)
            split_coin_image(path, split_dir)

    for filename in os.listdir(split_dir):
            img = preprocess_image(path)
            if img is not None:
                images[filename] = img
    return images

def preprocess_missing_dir(input_dir):
    images = {}
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(input_dir, filename)
            img = preprocess_image(path)
            if img is not None:
                images[filename] = img
    return images 