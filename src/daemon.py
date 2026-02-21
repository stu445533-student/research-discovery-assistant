"""
Background daemon for auto-updates
"""
import time
import logging
import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.main import ResearchAssistantSystem
import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/daemon.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def run_daemon():
    """Run background update service"""
    
    logger.info("="*80)
    logger.info("STARTING AUTO-UPDATE DAEMON")
    logger.info("="*80)
    logger.info(f"Update interval: {config.ARXIV_POLL_INTERVAL} seconds")
    
    system = ResearchAssistantSystem()
    
    # Load existing state
    system.load_state()
    
    update_count = 0
    
    try:
        while True:
            update_count += 1
            logger.info(f"\n[UPDATE #{update_count}] Starting at {datetime.now()}")
            
            try:
                system.update_paper_database()
                logger.info(f"✓ Update #{update_count} complete")
            except Exception as e:
                logger.error(f"✗ Update #{update_count} failed: {e}")
            
            # Wait 5 minutes
            logger.info(f"Sleeping for {config.ARXIV_POLL_INTERVAL} seconds...")
            time.sleep(config.ARXIV_POLL_INTERVAL)
            
    except KeyboardInterrupt:
        logger.info("\n\nDaemon stopped by user")
        logger.info(f"Total updates performed: {update_count}")
        system.save_state()

if __name__ == "__main__":
    run_daemon()
