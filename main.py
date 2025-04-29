import argparse
import logging
from pathlib import Path
from typing import Optional
import sys
import json
import sqlite3

from src.data_fetcher import fetch_coin_images
from src.image_processor import preprocess_directory
from src.feature_matcher import FeatureMatcher
from src.save_result import MatchDatabase
from src.visualizer import MatchVisualizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CoinMatcher:
    def __init__(
        self,
        api_key: str,
        db_path: str = "coin_matches.db",
        api_images_dir: str = "data/api_images",
        missing_images_dir: str = "data/missing_images",
        output_dir: str = "results",
        max_items: int = 20,
        match_threshold: float = 0.3
    ):
        """
        Initialize the coin matcher.
        
        Args:
            api_key: Harvard Art Museums API key
            db_path: Path to SQLite database
            api_images_dir: Directory for API images
            missing_images_dir: Directory for missing images
            output_dir: Directory for results
            max_items: Maximum number of items to fetch
            match_threshold: Minimum match score threshold
        """
        self.api_key = api_key
        self.db_path = Path(db_path)
        self.api_images_dir = Path(api_images_dir)
        self.missing_images_dir = Path(missing_images_dir)
        self.output_dir = Path(output_dir)
        self.max_items = max_items
        self.match_threshold = match_threshold
        
        # Create directories
        self.api_images_dir.mkdir(parents=True, exist_ok=True)
        self.missing_images_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self.db = MatchDatabase(str(self.db_path))
        
    def fetch_images(self):
        """Fetch coin images from the API."""
        logger.info("Fetching images from Harvard Art Museums API...")
        count = fetch_coin_images(
            self.api_key,
            str(self.api_images_dir),
            self.max_items
        )
        logger.info(f"Successfully fetched {count} images")
        
    def process_images(self):
        """Process and match images."""
        logger.info("Processing API images...")
        api_images = preprocess_directory(
            str(self.api_images_dir),
            str(self.output_dir / "processed_api"),
            is_api_images=True
        )
        
        logger.info("Processing missing images...")
        missing_images = preprocess_directory(
            str(self.missing_images_dir),
            str(self.output_dir / "processed_missing"),
            is_api_images=False
        )
        
        if not api_images or not missing_images:
            logger.error("No images to process")
            return
        
        logger.info("Matching images...")
        matcher = FeatureMatcher()
        
        # Track best matches for each missing image
        best_matches = {}
        
        # For each missing image, find the best matching API image
        for missing_name, missing_img in missing_images.items():
            best_score = 0.0
            best_api_name = None
            
            for api_name, api_img in api_images.items():
                try:
                    result = matcher.match_features(api_img, missing_img)
                    score = result.score
                    
                    if score > best_score and score >= self.match_threshold:
                        best_score = score
                        best_api_name = api_name
                        
                except Exception as e:
                    logger.error(f"Error matching {api_name} with {missing_name}: {e}")
                    continue
            
            if best_api_name is not None:
                logger.info(f"Best match for {missing_name}: {best_api_name} (score: {best_score:.3f})")
                best_matches[missing_name] = (best_api_name, best_score)
        
        # Save the best matches
        for missing_name, (api_name, score) in best_matches.items():
            self.db.save_match(api_name, missing_name, score)
        
        logger.info(f"Found {len(best_matches)} matches above threshold {self.match_threshold}")
    
    def visualize_results(self):
        """Visualize matching results."""
        logger.info("Generating visualizations...")
        visualizer = MatchVisualizer(
            str(self.db_path),
            str(self.api_images_dir),
            str(self.missing_images_dir),
            str(self.output_dir / "visualizations")
        )
        
        visualizer.visualize_best_matches()
        visualizer.plot_score_distribution()
        
        # Export results to JSON
        self.db.export_results(str(self.output_dir / "results.json"))

def main():
    parser = argparse.ArgumentParser(description="Coin Image Matching System")
    
    parser.add_argument(
        "--api-key",
        default="89ef3102-9109-435b-87a8-ec51bf7e171d",
        help="Harvard Art Museums API key"
    )
    parser.add_argument(
        "--db-path",
        default="coin_matches.db",
        help="Path to SQLite database"
    )
    parser.add_argument(
        "--api-images-dir",
        default="data/api_images",
        help="Directory for API images"
    )
    parser.add_argument(
        "--missing-images-dir",
        default="data/missing_images",
        help="Directory for missing images"
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory for results"
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=20,
        help="Maximum number of items to fetch"
    )
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=0.3,
        help="Minimum match score threshold"
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip fetching images from API"
    )
    
    args = parser.parse_args()
    
    try:
        # Create necessary directories
        Path(args.api_images_dir).mkdir(parents=True, exist_ok=True)
        Path(args.missing_images_dir).mkdir(parents=True, exist_ok=True)
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Check if missing images directory is not empty
        if not any(Path(args.missing_images_dir).iterdir()):
            logger.warning(f"No images found in {args.missing_images_dir}")
            logger.info("Please place your missing coin images in this directory")
            return
        
        matcher = CoinMatcher(
            api_key=args.api_key,
            db_path=args.db_path,
            api_images_dir=args.api_images_dir,
            missing_images_dir=args.missing_images_dir,
            output_dir=args.output_dir,
            max_items=args.max_items,
            match_threshold=args.match_threshold
        )
        
        if not args.skip_fetch:
            matcher.fetch_images()
        
        matcher.process_images()
        matcher.visualize_results()
        
        logger.info("Processing completed successfully")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()