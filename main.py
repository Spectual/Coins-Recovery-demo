from src.data_fetcher import fetch_coin_images
from src.image_processor import preprocess_directory
from src.feature_matcher import match_images
from src.save_result import init_db, save_match
from src.visualizer import visualize_matches

import os

API_KEY = "89ef3102-9109-435b-87a8-ec51bf7e171d"  
DB_PATH = "coin_matches.db"
API_IMAGES_DIR = "data/api_images"
SPLIT_IMAGES_DIR = "data/split_images"
MISSING_IMAGES_DIR = "data/missing_images"
MATCH_THRESHOLD = 30

def main():
    init_db(DB_PATH)

    # Fetch coin images from API
    fetch_coin_images(API_KEY, API_IMAGES_DIR, max_items=20)

    # Split local images fetched from API
    api_images = preprocess_local_dir(API_IMAGES_DIR, SPLIT_IMAGES_DIR)
    missing_images = preprocess_missing_dir(MISSING_IMAGES_DIR)

    # Match images
    # for api_name, api_img in api_images.items():
    #     for missing_name, missing_img in missing_images.items():
    #         score = match_images(api_img, missing_img, threshold=MATCH_THRESHOLD)
    #         print(f"[Match] {api_name} <--> {missing_name} : {score}")
    #         save_match(DB_PATH, api_name, missing_name, score)

    # # Step 5: 可视化匹配结果
    # visualize_matches(DB_PATH)

if __name__ == "__main__":
    main()