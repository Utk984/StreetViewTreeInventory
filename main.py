import logging
import time
import os
from src.pipeline.inventorize import inventorize
from src.pipeline.triangulate import triangulate_tree_detections
from config import Config
from ultralytics import YOLO
import asyncio
from src.utils.system_resources import calculate_optimal_concurrency

logger = logging.getLogger(__name__)

def load_models(config: Config):
    """Load tree segmentation and calibration models with logging."""
    logger.info("Starting model loading process")
    
    try:
        # Load tree segmentation model from config
        logger.info(f"Loading tree segmentation model from: {config.TREE_MODEL_PATH}")
        tree_model = YOLO(config.TREE_MODEL_PATH)
        logger.info("✅ Tree segmentation model loaded successfully")
        return tree_model
        
    except Exception as e:
        logger.error(f"❌ Failed to load models: {str(e)}")
        raise

def main():
    """Main pipeline execution with comprehensive logging."""
    pipeline_start_time = time.time()
    logger.info("=" * 60)
    logger.info("🌳 Tree Detection Pipeline Started")
    logger.info("=" * 60)
    
    try:
        config = Config()
        
        # Load models
        tree_model = load_models(config)
        
        # Calculate optimal concurrency based on system resources
        try:
            optimal_concurrent = calculate_optimal_concurrency() + 4
            logger.info(f"🔧 Auto-calculated optimal concurrency: {optimal_concurrent}")
        except Exception as e:
            logger.error(f"💥 Failed to calculate optimal concurrency: {str(e)}")
            optimal_concurrent = config.MAX_CONCURRENT
            logger.info(f"🔧 Using user-specified concurrency: {optimal_concurrent}")
        
        # Run streaming pipeline
        logger.info("🔄 Starting streaming panorama processing pipeline")
        asyncio.run(inventorize(config, tree_model, max_concurrent=optimal_concurrent, chunk_size=optimal_concurrent))
        
        # Run MRF triangulation post-processing if enabled
        if config.ENABLE_TRIANGULATION:
            logger.info("🔺 Starting MRF triangulation post-processing")
            triangulation_start_time = time.time()
            
            try:
                # Check if we have detection results to triangulate
                if not os.path.exists(config.OUTPUT_CSV) or os.path.getsize(config.OUTPUT_CSV) <= 100:
                    logger.warning("⚠️ No detection results found for triangulation, skipping...")
                else:
                    triangulated_trees = triangulate_tree_detections(
                        config.OUTPUT_CSV,
                        config.TRIANGULATED_OUTPUT_CSV,
                        max_object_dist=config.TRIANGULATION_MAX_OBJECT_DIST,
                        max_cluster_dist=config.TRIANGULATION_MAX_CLUSTER_DIST,
                        icm_iterations=config.TRIANGULATION_ICM_ITERATIONS,
                        depth_weight=config.TRIANGULATION_DEPTH_WEIGHT,
                        multiview_weight=config.TRIANGULATION_MULTIVIEW_WEIGHT
                    )
                    
                    triangulation_time = time.time() - triangulation_start_time
                    logger.info(f"✅ Triangulation completed in {triangulation_time:.2f} seconds")
                    logger.info(f"🌳 Final triangulated trees: {len(triangulated_trees)}")
                    logger.info(f"📄 Triangulated results saved to: {config.TRIANGULATED_OUTPUT_CSV}")
                    
            except Exception as e:
                logger.error(f"❌ Triangulation failed: {str(e)}")
                logger.warning("⚠️ Continuing with original detection results...")
        else:
            logger.info("🔺 Triangulation disabled in configuration")
        
        total_time = time.time() - pipeline_start_time
        logger.info("=" * 60)
        logger.info(f"🎉 Pipeline completed successfully in {total_time:.2f} seconds")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"💥 Pipeline failed with error: {str(e)}")
        logger.exception("Full traceback:")
        raise

if __name__ == "__main__":
    main()