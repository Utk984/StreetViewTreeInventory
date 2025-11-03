import os
import cv2
import numpy as np
import pandas as pd
import asyncio
import logging
import time
import re
from concurrent.futures import ThreadPoolExecutor

from config import Config
from src.inference.segment import detect_trunks
from src.utils.unwrap import divide_panorama
from src.utils.masks import add_masks, remove_duplicates, make_image, serialize_ultralytics_mask, save_panorama_masks
from src.utils.transformation import get_point
from src.utils.geodesic import localize_pixel_with_depth, get_depth_at_pixel

logger = logging.getLogger(__name__)

IO_EXECUTOR = ThreadPoolExecutor(max_workers=16)

def process_view(config: Config, view, tree_data, pano, image, theta, i):
    """Process a single view with detailed logging."""
    logger.debug(f"🔄 Processing view {i} at theta={theta}°")
    trees = []
    tree_masks = []
    
    try:
        for j, tree in enumerate(tree_data):
            masks = tree.masks
            boxes = tree.boxes
            
            if masks is not None:
                logger.debug(f"🌳 Processing {len(masks)} trees in view {i}")
                
                for k, mask in enumerate(masks):
                    image_path = os.path.join(config.VIEW_DIR, f"{pano.id}_view{i}_tree{j}_box{k}.jpg")
                    conf = boxes[k].conf.item()
                    
                    logger.debug(f"Processing tree {j}-{k} with confidence {conf:.3f}")

                    try:
                        orig_point, _ = get_point(mask, theta, pano, config.HEIGHT, config.WIDTH, config.FOV)
                        
                        # Get distance from depth map
                        W, H = image.shape[1], image.shape[0]
                        u, v = orig_point[0], orig_point[1]

                        # Filter out trees at extremes
                        if u < 350 or u > W - 350 or v < 150 or v > H - 150:
                            logger.warning(f"⚠️ Tree {j}-{k} is at extremes length wise")
                            continue
                        
                        distance_pano = get_depth_at_pixel(pano.depth, u, v, W, H, flipped=True)
                        if distance_pano is None:
                            logger.warning(f"⚠️ No depth map for {pano.id} at {u}, {v}")
                            continue
                        if distance_pano > 15:
                            logger.warning(f"⚠️ Distance too far for {pano.id} at {u}, {v}")
                            continue
                        
                        # Create local frame for coordinate transformation
                        lat_pano, lon_pano = localize_pixel_with_depth(pano, u, v, W, H, distance_pano)                        
                        logger.info(f"Pano distance: {distance_pano:.2f}m")
                        
                        # Submit image creation to thread pool
                        if config.SAVE_VIEWS:
                            IO_EXECUTOR.submit(
                                make_image, view, boxes[k], mask, image_path
                            )
                        
                    except Exception as e:
                        logger.error(f"❌ Error processing tree {k} in view {i}: {e}")
                        continue

                    tree = {
                        "image_path": image_path,
                        "pano_id": pano.id,
                        "stview_lat": pano.lat,
                        "stview_lng": pano.lon,
                        "tree_lat": lat_pano,
                        "tree_lng": lon_pano,
                        "image_x": float(u),
                        "image_y": float(v),
                        "theta": theta,
                        "conf": conf,
                        "distance_pano": distance_pano,
                    }
                    trees.append(tree)
                    tree_masks.append(mask)
        
        return trees, tree_masks
        
    except Exception as e:
        logger.error(f"❌ Error processing view {i}: {str(e)}")
        return [], []

async def process_single_panorama(pano_id: str, pano, image, config: Config, tree_model, semaphore: asyncio.Semaphore):
    """Process a single panorama with pre-fetched data."""
    async with semaphore:
        pano_start_time = time.time()
        
        try:
            logger.info(f"🔄 Processing panorama: {pano.id}")
            
            # Generate perspective views in parallel
            logger.info("🔄 Generating perspective views")
            view_start_time = time.time()
            
            # Run perspective generation in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            views = await loop.run_in_executor(
                None, divide_panorama, image, config.HEIGHT, config.WIDTH, config.FOV
            )
            
            view_time = time.time() - view_start_time
            logger.info(f"✅ Perspective Generation completed in {view_time:.3f}s")

            # Process each view (keeping sequential for YOLO model safety)
            trees = []
            all_masks = []
            view_processing_start = time.time()
            for i, (view, theta) in enumerate(views):
                logger.debug(f"🔍 Detecting trunks in view {i}")
                tree_data = detect_trunks(view, tree_model, config.DEVICE)
                view_trees, tree_masks = process_view(config, view, tree_data, pano, image, theta, i)
                trees.extend(view_trees)
                all_masks.extend(tree_masks)
            
            view_processing_time = time.time() - view_processing_start
            logger.info(f"✅ View Processing completed in {view_processing_time:.3f}s")

            if not trees:
                logger.info(f"🌳 No trunks found in panorama {pano.id}")
                return None

            logger.info(f"🌳 Found {len(trees)} trunks in panorama {pano.id}")
            
            # Remove duplicates
            logger.info("🔄 Removing duplicate detections")
            dedup_start_time = time.time()
            
            df_part, masks_part = remove_duplicates(
                pd.DataFrame(trees),
                all_masks,
                image.shape[1], image.shape[0],
                config.HEIGHT, config.WIDTH, config.FOV
            )
            dedup_time = time.time() - dedup_start_time
            logger.info(f"✅ Deduplication completed in {dedup_time:.3f}s - {len(df_part)} trees remaining")

            # Serialize masks for JSON storage
            mask_json_path = None
            if masks_part:
                logger.info(f"💾 Serializing {len(masks_part)} kept masks for JSON storage")
                serialization_start = time.time()
                
                # Group masks by view
                view_masks = {}
                for idx, mask in enumerate(masks_part):
                    if mask is not None and idx < len(df_part):
                        row = df_part.iloc[idx]
                        image_path = row.get('image_path', '')
                        
                        # Extract view number from image_path
                        view_num = None
                        if 'view' in image_path:
                            try:
                                match = re.search(r'view(\d+)', image_path)
                                if match:
                                    view_num = int(match.group(1))
                            except:
                                pass
                        
                        if view_num is not None:
                            view_name = f"view_{view_num}"
                            
                            if view_name not in view_masks:
                                view_masks[view_name] = []
                            
                            serialized_mask = serialize_ultralytics_mask(mask)
                            view_masks[view_name].append({
                                "tree_index": f"{idx}-0",
                                "image_path": image_path,
                                "confidence": float(row.get('conf', 0.0)),
                                "mask_data": serialized_mask
                            })
                        else:
                            logger.warning(f"⚠️ Could not extract view number from path: {image_path}")
                
                if view_masks:
                    mask_json_path = save_panorama_masks(pano.id, view_masks, config)
                    serialization_time = time.time() - serialization_start
                    logger.info(f"✅ Masks Serialization completed in {serialization_time:.3f}s to {mask_json_path}")

            # Save full panorama with masks (non-blocking)
            def _save_full():
                logger.debug(f"💾 Saving full panorama with masks: {pano.id}")
                full = add_masks(image.copy(), df_part.copy(), config.HEIGHT, config.WIDTH, config.FOV, mask_json_path)
                full_path = os.path.join(config.FULL_DIR, f"{pano.id}.jpg")
                # Convert RGB to BGR for OpenCV imwrite
                full_bgr = cv2.cvtColor(full, cv2.COLOR_RGB2BGR)
                cv2.imwrite(full_path, full_bgr)
                logger.debug(f"✅ Full Panorama saved: {full_path}")
                
                if not config.SAVE_MASK_JSON and mask_json_path:
                    os.remove(mask_json_path)

            if config.SAVE_FULL:
                IO_EXECUTOR.submit(_save_full)

            # Remove exact duplicates before saving
            df_part = df_part.drop_duplicates()
            logger.info(f"📊 After deduplication: {len(df_part)} unique detections")
            
            # Save results to CSV (non-blocking)
            logger.info("💾 Saving results to CSV")
            save_start = time.time()
            
            def _save_csv():
                df_part.to_csv(config.OUTPUT_CSV, mode='a', index=False, header=False, lineterminator="\n")
            
            IO_EXECUTOR.submit(_save_csv)
            
            save_time = time.time() - save_start
            logger.info(f"✅ Results CSV queued for saving in {save_time:.3f}s")

            pano_time = time.time() - pano_start_time
            logger.info(f"✅ Panorama {pano.id} Processing completed in {pano_time:.3f}s")
            
            return {
                'pano_id': pano.id,
                'trees_found': len(df_part),
                'processing_time': pano_time
            }
            
        except Exception as e:
            logger.error(f"❌ Error processing panorama {pano_id}: {str(e)}")
            logger.exception("Full traceback:")
            return None
