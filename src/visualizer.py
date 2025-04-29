import cv2
import numpy as np
from typing import List, Tuple, Optional
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from src.save_result import MatchDatabase
from src.feature_matcher import FeatureMatcher
import sqlite3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MatchVisualizer:
    def __init__(
        self,
        db_path: str,
        api_images_dir: str,
        missing_images_dir: str,
        output_dir: str = "results/visualizations"
    ):
        """
        Initialize the match visualizer.
        
        Args:
            db_path: Path to the SQLite database
            api_images_dir: Directory containing API images
            missing_images_dir: Directory containing missing images
            output_dir: Directory to save visualizations
        """
        self.db = MatchDatabase(db_path)
        self.api_images_dir = Path(api_images_dir)
        self.missing_images_dir = Path(missing_images_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def draw_matches(
        self,
        img1: np.ndarray,
        kp1: List[cv2.KeyPoint],
        img2: np.ndarray,
        kp2: List[cv2.KeyPoint],
        matches: List[cv2.DMatch],
        max_matches: int = 30
    ) -> np.ndarray:
        """
        Draw matching keypoints between two images.
        
        Args:
            img1: First image
            kp1: Keypoints from first image
            img2: Second image
            kp2: Keypoints from second image
            matches: List of matches
            max_matches: Maximum number of matches to draw
            
        Returns:
            Image with drawn matches
        """
        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)
        matches = matches[:max_matches]
        
        # Draw matches
        img_matches = cv2.drawMatches(
            img1, kp1, img2, kp2, matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        
        return img_matches
    
    def visualize_match(
        self,
        api_image: str,
        missing_image: str,
        save_path: Optional[str] = None
    ):
        """
        Visualize a single match between two images.
        
        Args:
            api_image: Name of the API image
            missing_image: Name of the missing image
            save_path: Path to save the visualization
        """
        try:
            # Load images from processed directories
            processed_api_dir = self.output_dir.parent / "processed_api"
            processed_missing_dir = self.output_dir.parent / "processed_missing"
            
            # For API images, we need to handle both front and back
            if api_image.endswith('_front') or api_image.endswith('_back'):
                img1_path = processed_api_dir / f"{api_image}.jpg"
            else:
                img1_path = processed_api_dir / f"{api_image}_front.jpg"
                
            # For missing images, we just use the original name
            img2_path = processed_missing_dir / f"{missing_image}.jpg"
            
            logger.info(f"Loading images: {img1_path}, {img2_path}")
            
            # Check if files exist
            if not img1_path.exists():
                logger.error(f"API image not found: {img1_path}")
                return
            if not img2_path.exists():
                logger.error(f"Missing image not found: {img2_path}")
                return
            
            img1 = cv2.imread(str(img1_path))
            img2 = cv2.imread(str(img2_path))
            
            if img1 is None or img2 is None:
                logger.error(f"Failed to load one or both images: {img1_path}, {img2_path}")
                return
            
            # Match features
            matcher = FeatureMatcher()
            result = matcher.match_features(img1, img2)
            
            # Draw matches
            vis_img = self.draw_matches(
                img1, result.keypoints1,
                img2, result.keypoints2,
                result.matches
            )
            
            # Add score text
            text = f"Match Score: {result.score:.3f}"
            cv2.putText(
                vis_img, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )
            
            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(save_path), vis_img)
                logger.info(f"Saved visualization to {save_path}")
            
            return vis_img
            
        except Exception as e:
            logger.error(f"Error visualizing match: {e}")
            return None
    
    def visualize_best_matches(
        self,
        top_k: int = 5,
        min_score: float = 0.3
    ):
        """
        Visualize the top-k best matches.
        
        Args:
            top_k: Number of top matches to visualize
            min_score: Minimum score threshold
        """
        matches = self.db.get_best_matches(top_k, min_score)
        
        for i, (api_img, missing_img, score) in enumerate(matches):
            save_path = self.output_dir / f"match_{i+1}_{score:.3f}.jpg"
            self.visualize_match(api_img, missing_img, str(save_path))
    
    def plot_score_distribution(self):
        """Plot the distribution of match scores."""
        try:
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT score FROM matches")
                scores = [row[0] for row in cursor.fetchall()]
            
            if not scores:
                logger.warning("No scores found in database")
                return
            
            plt.figure(figsize=(10, 6))
            plt.hist(scores, bins=30, edgecolor='black')
            plt.title('Distribution of Match Scores')
            plt.xlabel('Score')
            plt.ylabel('Frequency')
            
            save_path = self.output_dir / "score_distribution.png"
            plt.savefig(save_path)
            plt.close()
            
            logger.info(f"Saved score distribution plot to {save_path}")
            
        except Exception as e:
            logger.error(f"Error plotting score distribution: {e}")

def visualize_matches(
    db_path: str,
    api_images_dir: str = "data/api_images",
    missing_images_dir: str = "data/missing_images",
    output_dir: str = "results/visualizations",
    top_k: int = 5
):
    """
    Visualize matches between API and missing images.
    
    Args:
        db_path: Path to the SQLite database
        api_images_dir: Directory containing API images
        missing_images_dir: Directory containing missing images
        output_dir: Directory to save visualizations
        top_k: Number of top matches to visualize
    """
    visualizer = MatchVisualizer(db_path, api_images_dir, missing_images_dir, output_dir)
    visualizer.visualize_best_matches(top_k)
    visualizer.plot_score_distribution()