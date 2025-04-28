import os
import requests
import time

def fetch_coin_images(api_key, save_dir="data/api_images", max_items=20):
    os.makedirs(save_dir, exist_ok=True)
    url = "https://api.harvardartmuseums.org/object"
    params = {
        "apikey": api_key,
        "classification": "Coins",
        "size": 10,
        "hasimage": 1,
        "page": 1
    }
    
    count = 0
    while count < max_items:
        response = requests.get(url, params=params)
        if response.status_code != 200:
            print(f"Failed to fetch page {params['page']}")
            break
        
        data = response.json()
        for record in data.get("records", []):
            img_url = record.get("primaryimageurl")
            if img_url:
                try:
                    img_data = requests.get(img_url).content
                    with open(os.path.join(save_dir, f"coin_{count}.jpg"), 'wb') as f:
                        f.write(img_data)
                    count += 1
                    if count >= max_items:
                        break
                except Exception as e:
                    print(f"Download error: {e}")
        
        if not data.get("info", {}).get("next"):
            break  # 没有下一页了
        params["page"] += 1
        time.sleep(1)  # polite crawling
    print(f"Total downloaded images: {count}")