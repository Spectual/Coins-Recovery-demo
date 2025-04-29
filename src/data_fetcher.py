import os
import requests
import time
from tqdm import tqdm
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_coin_images(
    api_key: str,
    save_dir: str = "data/api_images",
    max_items: int = 20,
    retry_attempts: int = 3,
    delay_between_retries: int = 5
) -> int:
    """
    Fetch coin images from Harvard Art Museums API.
    
    Args:
        api_key: API key for authentication
        save_dir: Directory to save downloaded images
        max_items: Maximum number of images to download
        retry_attempts: Number of retry attempts for failed requests
        delay_between_retries: Delay between retry attempts in seconds
        
    Returns:
        Number of successfully downloaded images
    """
    os.makedirs(save_dir, exist_ok=True)
    url = "https://api.harvardartmuseums.org/object"
    
    count = 0
    page = 1
    
    with tqdm(total=max_items, desc="Downloading images") as pbar:
        while count < max_items:
            params = {
                "apikey": api_key,
                "classification": "Coins",
                "size": 10,
                "hasimage": 1,
                "page": page
            }
            
            for attempt in range(retry_attempts):
                try:
                    response = requests.get(url, params=params, timeout=30)
                    response.raise_for_status()
                    break
                except (requests.RequestException, requests.Timeout) as e:
                    if attempt == retry_attempts - 1:
                        logger.error(f"Failed to fetch page {page} after {retry_attempts} attempts: {e}")
                        return count
                    logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay_between_retries} seconds...")
                    time.sleep(delay_between_retries)
            
            data = response.json()
            records = data.get("records", [])
            
            if not records:
                logger.info("No more records available")
                break
                
            for record in records:
                if count >= max_items:
                    break
                    
                img_url = record.get("primaryimageurl")
                if not img_url:
                    continue
                    
                try:
                    img_response = requests.get(img_url, timeout=30)
                    img_response.raise_for_status()
                    
                    # Save image with metadata
                    filename = f"coin_{count}_{record.get('objectnumber', 'unknown')}.jpg"
                    filepath = os.path.join(save_dir, filename)
                    
                    with open(filepath, 'wb') as f:
                        f.write(img_response.content)
                    
                    # Save metadata
                    metadata = {
                        'object_number': record.get('objectnumber'),
                        'title': record.get('title'),
                        'date': record.get('dated'),
                        'culture': record.get('culture'),
                        'period': record.get('period')
                    }
                    
                    with open(os.path.join(save_dir, f"{filename}.json"), 'w') as f:
                        import json
                        json.dump(metadata, f, indent=2)
                    
                    count += 1
                    pbar.update(1)
                    
                except Exception as e:
                    logger.error(f"Error downloading image {img_url}: {e}")
                    continue
            
            if not data.get("info", {}).get("next"):
                break
                
            page += 1
            time.sleep(1)  # Polite crawling
            
    logger.info(f"Successfully downloaded {count} images")
    return count