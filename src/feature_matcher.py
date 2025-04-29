import cv2
import numpy as np
from typing import Tuple, Dict, List, Optional
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MatchResult:
    score: float
    keypoints1: List[cv2.KeyPoint]
    keypoints2: List[cv2.KeyPoint]
    matches: List[cv2.DMatch]
    homography: Optional[np.ndarray] = None

class FeatureMatcher:
    def __init__(
        self,
        detector_type: str = "ORB",
        matcher_type: str = "FLANN",
        ratio_threshold: float = 0.75,
        min_matches: int = 10
    ):
        """
        Initialize the feature matcher.
        
        Args:
            detector_type: Type of feature detector ("ORB", "SIFT", or "SURF")
            matcher_type: Type of matcher ("FLANN" or "BF")
            ratio_threshold: Ratio test threshold for good matches
            min_matches: Minimum number of matches required
        """
        self.detector_type = detector_type
        self.matcher_type = matcher_type
        self.ratio_threshold = ratio_threshold
        self.min_matches = min_matches
        
        # Initialize detector
        if detector_type == "ORB":
            self.detector = cv2.ORB_create(
                nfeatures=1000,
                scaleFactor=1.2,
                nlevels=8,
                edgeThreshold=31
            )
        elif detector_type == "SIFT":
            self.detector = cv2.SIFT_create()
        elif detector_type == "SURF":
            self.detector = cv2.xfeatures2d.SURF_create()
        else:
            raise ValueError(f"Unsupported detector type: {detector_type}")
        
        # Initialize matcher
        if matcher_type == "FLANN":
            if detector_type == "ORB":
                index_params = dict(
                    algorithm=6,  # FLANN_INDEX_LSH
                    table_number=6,
                    key_size=12,
                    multi_probe_level=1
                )
            else:
                index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE
            search_params = dict(checks=50)
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:  # BF
            if detector_type == "ORB":
                self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            else:
                self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    
    def extract_features(self, image: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """
        Extract features from an image with improved robustness to lighting changes.
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (keypoints, descriptors)
        """
        try:
            # Ensure image is in correct format
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive histogram equalization
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            image = clahe.apply(image)
            
            # Apply gamma correction
            gamma = 1.5
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            image = cv2.LUT(image, table)
            
            # Apply bilateral filtering to reduce noise while preserving edges
            image = cv2.bilateralFilter(image, 9, 75, 75)
            
            # Extract features with increased number of keypoints
            if self.detector_type == "ORB":
                self.detector = cv2.ORB_create(
                    nfeatures=2000,  # Increased from 1000
                    scaleFactor=1.2,
                    nlevels=8,
                    edgeThreshold=31,
                    patchSize=31,    # Increased from default
                    fastThreshold=20  # Lowered to detect more keypoints
                )
            elif self.detector_type == "SIFT":
                self.detector = cv2.SIFT_create(
                    nfeatures=2000,  # Increased from default
                    contrastThreshold=0.03,  # Lowered to detect more keypoints
                    edgeThreshold=10
                )
            
            keypoints, descriptors = self.detector.detectAndCompute(image, None)
            
            if keypoints is None or descriptors is None or len(keypoints) == 0:
                logger.warning("No features could be extracted from the image")
                return [], np.array([])
            
            return keypoints, descriptors
        
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return [], np.array([])
    
    def match_features(
        self,
        img1: np.ndarray,
        img2: np.ndarray
    ) -> MatchResult:
        """
        Match features between two images with improved robustness.
        
        Args:
            img1: First image
            img2: Second image
            
        Returns:
            MatchResult containing matching information
        """
        # Extract features
        kp1, des1 = self.extract_features(img1)
        kp2, des2 = self.extract_features(img2)
        
        if len(kp1) == 0 or len(kp2) == 0 or des1.size == 0 or des2.size == 0:
            logger.warning("Could not extract features from one or both images")
            return MatchResult(0.0, kp1, kp2, [])
        
        # Match features with improved parameters
        if self.matcher_type == "FLANN":
            try:
                matches = self.matcher.knnMatch(des1, des2, k=2)
                
                # Apply ratio test with relaxed threshold
                good_matches = []
                for m, n in matches:
                    if m.distance < 0.8 * n.distance:  # Relaxed from 0.75
                        good_matches.append(m)
            except Exception as e:
                logger.error(f"Error in FLANN matching: {e}")
                good_matches = []
        else:
            try:
                good_matches = self.matcher.match(des1, des2)
            except Exception as e:
                logger.error(f"Error in BF matching: {e}")
                good_matches = []
        
        # Calculate score with improved weighting
        if len(good_matches) < self.min_matches:
            return MatchResult(0.0, kp1, kp2, good_matches)
        
        try:
            # Calculate homography with improved parameters
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            # Use RANSAC with increased iterations and relaxed threshold
            homography, mask = cv2.findHomography(
                src_pts, dst_pts,
                cv2.RANSAC,
                ransacReprojThreshold=5.0,  # Increased from default
                maxIters=2000,              # Increased from default
                confidence=0.995            # Increased from default
            )
            
            inliers = np.sum(mask)
            
            # Calculate final score with improved weighting
            match_ratio = len(good_matches) / min(len(kp1), len(kp2))
            inlier_ratio = inliers / len(good_matches)
            score = (0.6 * inlier_ratio + 0.4 * match_ratio) * 100  # Scale to 0-100 range
            
            return MatchResult(score, kp1, kp2, good_matches, homography)
        except cv2.error as e:
            logger.warning(f"Could not compute homography: {e}")
            return MatchResult(0.0, kp1, kp2, good_matches)

def match_images(
    img1: np.ndarray,
    img2: np.ndarray,
    threshold: float = 0.5,
    detector_type: str = "ORB",
    matcher_type: str = "FLANN"
) -> float:
    """
    Match two images and return a similarity score.
    
    Args:
        img1: First image
        img2: Second image
        threshold: Minimum score threshold
        detector_type: Type of feature detector
        matcher_type: Type of matcher
        
    Returns:
        Similarity score between 0 and 1
    """
    matcher = FeatureMatcher(detector_type, matcher_type)
    result = matcher.match_features(img1, img2)
    return result.score if result.score >= threshold else 0.0