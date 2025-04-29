import sqlite3
from typing import List, Tuple, Optional
import json
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MatchDatabase:
    def __init__(self, db_path: str):
        """
        Initialize the match database.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Initialize the database with required tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create matches table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS matches (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        api_image_name TEXT NOT NULL,
                        missing_image_name TEXT NOT NULL,
                        score REAL NOT NULL,
                        match_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        metadata TEXT
                    )
                """)
                
                # Create index for faster queries
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_matches_score 
                    ON matches(score DESC)
                """)
                
                conn.commit()
                logger.info(f"Database initialized at {self.db_path}")
                
        except sqlite3.Error as e:
            logger.error(f"Database initialization error: {e}")
            raise
    
    def save_match(
        self,
        api_image: str,
        missing_image: str,
        score: float,
        metadata: Optional[dict] = None
    ):
        """
        Save a match result to the database.
        
        Args:
            api_image: Name of the API image
            missing_image: Name of the missing image
            score: Match score
            metadata: Additional metadata to save
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if this match already exists
                cursor.execute("""
                    SELECT id FROM matches 
                    WHERE api_image_name = ? AND missing_image_name = ?
                """, (api_image, missing_image))
                
                existing = cursor.fetchone()
                
                if existing:
                    # Update existing match
                    cursor.execute("""
                        UPDATE matches 
                        SET score = ?, match_time = CURRENT_TIMESTAMP, metadata = ?
                        WHERE id = ?
                    """, (score, json.dumps(metadata) if metadata else None, existing[0]))
                    logger.debug(f"Updated match: {api_image} <-> {missing_image} (score: {score})")
                else:
                    # Insert new match
                    cursor.execute("""
                        INSERT INTO matches 
                        (api_image_name, missing_image_name, score, metadata)
                        VALUES (?, ?, ?, ?)
                    """, (
                        api_image,
                        missing_image,
                        score,
                        json.dumps(metadata) if metadata else None
                    ))
                    logger.debug(f"Saved new match: {api_image} <-> {missing_image} (score: {score})")
                
                conn.commit()
                
        except sqlite3.Error as e:
            logger.error(f"Error saving match to database: {e}")
            raise
    
    def get_best_matches(
        self,
        limit: int = 10,
        min_score: float = 0.0
    ) -> List[Tuple[str, str, float]]:
        """
        Get the best matches from the database.
        
        Args:
            limit: Maximum number of matches to return
            min_score: Minimum score threshold
            
        Returns:
            List of (api_image, missing_image, score) tuples
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT api_image_name, missing_image_name, score
                    FROM matches
                    WHERE score >= ?
                    ORDER BY score DESC
                    LIMIT ?
                """, (min_score, limit))
                
                return cursor.fetchall()
                
        except sqlite3.Error as e:
            logger.error(f"Error retrieving matches from database: {e}")
            return []
    
    def export_results(self, output_path: str):
        """
        Export all match results to a JSON file.
        
        Args:
            output_path: Path to save the JSON file
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT 
                        api_image_name,
                        missing_image_name,
                        score,
                        match_time,
                        metadata
                    FROM matches
                    ORDER BY score DESC
                """)
                
                results = []
                for row in cursor.fetchall():
                    result = {
                        'api_image': row[0],
                        'missing_image': row[1],
                        'score': row[2],
                        'match_time': row[3],
                        'metadata': json.loads(row[4]) if row[4] else None
                    }
                    results.append(result)
                
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2)
                
                logger.info(f"Exported results to {output_path}")
                
        except (sqlite3.Error, IOError) as e:
            logger.error(f"Error exporting results: {e}")
            raise

def init_db(db_path: str) -> MatchDatabase:
    """
    Initialize and return a database connection.
    
    Args:
        db_path: Path to the SQLite database file
        
    Returns:
        Initialized MatchDatabase instance
    """
    return MatchDatabase(db_path)

def save_match(
    db_path: str,
    api_image: str,
    missing_image: str,
    score: float,
    metadata: Optional[dict] = None
):
    """
    Save a match result to the database.
    
    Args:
        db_path: Path to the SQLite database file
        api_image: Name of the API image
        missing_image: Name of the missing image
        score: Match score
        metadata: Additional metadata to save
    """
    db = MatchDatabase(db_path)
    db.save_match(api_image, missing_image, score, metadata)