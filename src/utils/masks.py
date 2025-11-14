import cv2
import numpy as np
import logging
import time
import os
from src.utils.transformation import map_perspective_point_to_original
from shapely.geometry import Polygon
from shapely.strtree import STRtree
import json
import torch
from typing import Dict, Any, List

# Configure logger for mask processing
logger = logging.getLogger(__name__)

try:
    from pycocotools import mask as maskUtils
    HAS_PYCOCOTOOLS = True
except ImportError:
    HAS_PYCOCOTOOLS = False
    logger.warning("pycocotools not installed. Install with: pip install pycocotools")

def serialize_ultralytics_mask(mask_obj) -> Dict[str, Any]:
    """
    Serialize an Ultralytics mask object to a JSON-serializable dictionary using RLE encoding.
    This reduces storage size by 100-1000x compared to storing polygon coordinates.
    
    Args:
        mask_obj: Ultralytics mask object
        
    Returns:
        Dict containing serialized mask data with RLE encoding
    """
    try:
        mask_data = {
            "orig_shape": mask_obj.orig_shape if hasattr(mask_obj, 'orig_shape') else None,
            "encoding": "rle" if HAS_PYCOCOTOOLS else "polygon",
        }
        
        if HAS_PYCOCOTOOLS and hasattr(mask_obj, 'xy') and mask_obj.xy is not None and len(mask_obj.xy) > 0:
            # Use RLE encoding for compact storage
            try:
                # Get the mask's bounding box and dimensions
                polygon = mask_obj.xy[0]
                if isinstance(polygon, torch.Tensor):
                    polygon = polygon.detach().cpu().numpy()
                
                # Get image dimensions from orig_shape or estimate from polygon
                if hasattr(mask_obj, 'orig_shape') and mask_obj.orig_shape:
                    h, w = mask_obj.orig_shape[:2]
                else:
                    # Estimate from polygon bounds
                    h = int(np.max(polygon[:, 1])) + 1
                    w = int(np.max(polygon[:, 0])) + 1
                
                # Create binary mask from polygon
                binary_mask = np.zeros((h, w), dtype=np.uint8)
                pts = polygon.astype(np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(binary_mask, [pts], 1)
                
                # Encode to RLE (COCO format)
                rle = maskUtils.encode(np.asfortranarray(binary_mask))
                rle['counts'] = rle['counts'].decode('utf-8')  # Convert bytes to string for JSON
                
                mask_data["rle"] = rle
                mask_data["bbox"] = [
                    float(np.min(polygon[:, 0])),  # x_min
                    float(np.min(polygon[:, 1])),  # y_min
                    float(np.max(polygon[:, 0])),  # x_max
                    float(np.max(polygon[:, 1]))   # y_max
                ]
                
                logger.debug(f"✅ RLE encoded mask: shape={h}x{w}, RLE size={len(rle['counts'])} chars")
                
            except Exception as e:
                # Fallback to polygon storage if RLE fails
                logger.warning(f"⚠️ RLE encoding failed, falling back to polygon: {str(e)}")
                mask_data["encoding"] = "polygon"
                mask_data["xy"] = [polygon.tolist() if isinstance(polygon, np.ndarray) else polygon 
                                   for polygon in mask_obj.xy]
        else:
            # Fallback: store polygon coordinates (original method)
            if hasattr(mask_obj, 'xy') and mask_obj.xy is not None:
                xy_list = []
                for polygon in mask_obj.xy:
                    if isinstance(polygon, np.ndarray):
                        xy_list.append(polygon.tolist())
                    elif isinstance(polygon, torch.Tensor):
                        xy_list.append(polygon.detach().cpu().numpy().tolist())
                    else:
                        xy_list.append(polygon)
                mask_data["xy"] = xy_list
        
        return mask_data
        
    except Exception as e:
        logger.error(f"❌ Error serializing mask: {str(e)}")
        raise

def deserialize_ultralytics_mask(mask_data: Dict[str, Any]):
    """
    Deserialize a mask from JSON data back to a format compatible with Ultralytics.
    Handles both RLE encoded masks and polygon format.
    
    Args:
        mask_data: Dictionary containing serialized mask data
        
    Returns:
        Dictionary with mask attributes including polygon coordinates
    """
    try:
        mask_obj = {}
        
        if mask_data.get("orig_shape"):
            mask_obj["orig_shape"] = tuple(mask_data["orig_shape"])
        
        # Handle RLE encoded masks
        if mask_data.get("encoding") == "rle" and mask_data.get("rle") and HAS_PYCOCOTOOLS:
            try:
                rle = mask_data["rle"]
                # Decode RLE to binary mask
                if isinstance(rle['counts'], str):
                    rle['counts'] = rle['counts'].encode('utf-8')
                binary_mask = maskUtils.decode(rle)
                
                # Find contours to get polygon coordinates
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # Get the largest contour (main mask)
                    largest_contour = max(contours, key=cv2.contourArea)
                    # Convert to xy format
                    polygon = largest_contour.squeeze().astype(np.float32)
                    if polygon.ndim == 2 and polygon.shape[0] > 2:
                        mask_obj["xy"] = [polygon]
                    else:
                        # Fallback if contour is too small
                        logger.warning("⚠️ Contour too small, using bbox")
                        if mask_data.get("bbox"):
                            x1, y1, x2, y2 = mask_data["bbox"]
                            mask_obj["xy"] = [np.array([
                                [x1, y1], [x2, y1], [x2, y2], [x1, y2]
                            ], dtype=np.float32)]
                
                # Store binary mask as well
                mask_obj["data"] = binary_mask.astype(np.float32)
                
                logger.debug(f"✅ Decoded RLE mask to polygon with {len(mask_obj.get('xy', [[]])[0])} points")
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to decode RLE, trying polygon fallback: {str(e)}")
                # Try polygon fallback
                if mask_data.get("xy"):
                    mask_obj["xy"] = [np.array(polygon, dtype=np.float32) for polygon in mask_data["xy"]]
        
        # Handle polygon format (fallback or original)
        elif mask_data.get("xy"):
            mask_obj["xy"] = [np.array(polygon, dtype=np.float32) for polygon in mask_data["xy"]]
        
        if mask_data.get("xyn"):
            mask_obj["xyn"] = [np.array(polygon, dtype=np.float32) for polygon in mask_data["xyn"]]
        
        if mask_data.get("data") and "data" not in mask_obj:
            mask_obj["data"] = np.array(mask_data["data"], dtype=np.float32)
        
        return mask_obj
        
    except Exception as e:
        logger.error(f"❌ Error deserializing mask: {str(e)}")
        raise

def save_panorama_masks(pano_id: str, view_masks: Dict[str, List[Dict[str, Any]]], config) -> str:
    """
    Save masks for a panorama in JSON format.
    
    Args:
        pano_id: Panorama ID
        view_masks: Dictionary with view paths as keys and list of mask data as values
        config: Configuration object with MASK_DIR
        
    Returns:
        Path to the saved JSON file
    """
    try:
        # Create mask directory if it doesn't exist
        os.makedirs(config.MASK_DIR, exist_ok=True)
        
        # Create JSON file path
        json_path = os.path.join(config.MASK_DIR, f"{pano_id}_masks.json")
        
        # Prepare data structure
        mask_data = {
            "pano_id": pano_id,
            "views": view_masks,
            "metadata": {
                "total_views": len(view_masks),
                "total_masks": sum(len(masks) for masks in view_masks.values())
            }
        }
        
        # Save to JSON file
        with open(json_path, 'w') as f:
            json.dump(mask_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Saved {mask_data['metadata']['total_masks']} masks for panorama {pano_id} to {json_path}")
        return json_path
        
    except Exception as e:
        logger.error(f"❌ Error saving panorama masks for {pano_id}: {str(e)}")
        raise

def load_panorama_masks(json_path: str) -> Dict[str, Any]:
    """
    Load masks for a panorama from JSON file.
    
    Args:
        json_path: Path to the JSON file
        
    Returns:
        Dictionary containing mask data
    """
    try:
        with open(json_path, 'r') as f:
            mask_data = json.load(f)
        
        logger.debug(f"✅ Loaded masks from {json_path}")
        return mask_data
        
    except Exception as e:
        logger.error(f"❌ Error loading masks from {json_path}: {str(e)}")
        raise

def make_image(view, box, mask, image_path):
    """
    Create and save a tree image with bounding box, confidence, and mask overlay.
    """
    logger.debug(f"💾 Creating tree image: {image_path}")
    start_time = time.time()
    
    try:
        xyxy = [int(coord) for coord in box.xyxy[0].tolist()]
        conf = box.conf.item()
        
        logger.debug(f"📦 Bounding box: {xyxy}, confidence: {conf:.3f}")

        im = view.copy()
        cv2.rectangle(im, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
        cv2.putText(
            im,
            f"{conf:.2f}",
            (xyxy[0], max(xyxy[1] - 5, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            2,
        )

        mask_points = mask.xy[0].astype(np.int32)
        overlay = view.copy()
        cv2.fillPoly(overlay, [mask_points], (0, 255, 0))
        cv2.addWeighted(overlay, 0.3, im, 0.7, 0, im)
        cv2.polylines(im, [mask_points], isClosed=True, color=(0, 255, 0), thickness=1)

        # Convert RGB to BGR for OpenCV imwrite
        im_bgr = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        cv2.imwrite(image_path, im_bgr)
        
        save_time = time.time() - start_time
        logger.debug(f"✅ Tree image saved in {save_time:.3f}s: {image_path}")
        
    except Exception as e:
        logger.error(f"❌ Error creating tree image {image_path}: {str(e)}")
        raise

def remove_duplicates(df, masks, image_x, image_y, height, width, fov, iou_threshold=0.4):
    """
    Remove duplicate detections with comprehensive logging.
    * Converts each mask to a Shapely Polygon (no giant binary images)
    * Uses an STRtree to query only potentially overlapping pairs
    
    Args:
        df: DataFrame with detection data (without mask column)
        masks: List of Ultralytics mask objects corresponding to each row in df
        image_x, image_y: Image dimensions
        height, width, fov: Perspective view parameters
        iou_threshold: IoU threshold for duplicate removal
        
    Returns:
        tuple: (filtered_df, filtered_masks) - DataFrame and masks list with duplicates removed
    """
    logger.info(f"🔄 Starting duplicate removal - {len(df)} initial detections")
    start_time = time.time()
    
    try:
        polys, areas, idx_map = [], [], [] 

        # Convert masks to polygons
        conversion_start = time.time()
        valid_masks = 0
        
        for idx, (mask, th) in enumerate(zip(masks, df["theta"])):
            if mask is None or not hasattr(mask, 'xy') or not mask.xy:
                continue

            try:
                pts = mask.xy[0].astype(np.float32)
                pano_pts = np.vstack(
                    [
                        map_perspective_point_to_original(
                            x, y, th, (image_x, image_y), height, width, fov
                        )
                        for x, y in pts
                    ]
                )
                poly = Polygon(pano_pts).buffer(0)  
                if poly.is_empty or not poly.is_valid:
                    continue

                polys.append(poly)
                areas.append(poly.area)
                idx_map.append(idx)
                valid_masks += 1
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to convert mask {idx} to polygon: {str(e)}")
                continue

        conversion_time = time.time() - conversion_start
        logger.debug(f"📐 Converted {valid_masks}/{len(df)} masks to polygons in {conversion_time:.3f}s")

        if not polys:
            logger.warning("⚠️ No valid polygons found, returning empty DataFrame and masks")
            return df.iloc[0:0], []        
        
        # Build spatial index and find overlaps
        dedup_start = time.time()
        tree = STRtree(polys)
        keep = set(range(len(polys)))
        removed_count = 0

        for i in range(len(polys)):
            if i not in keep:
                continue
                
            candidates = tree.query(polys[i])
            
            for j in candidates:
                if j <= i or j not in keep:
                    continue
                    
                try:
                    intersection = polys[i].intersection(polys[j])
                    inter_area = intersection.area if hasattr(intersection, 'area') else 0
                    
                    if inter_area == 0:
                        continue
                        
                    smaller_area = min(areas[i], areas[j])
                    iou = inter_area / smaller_area
                    
                    if iou > iou_threshold:
                        remove_idx = i if areas[i] < areas[j] else j
                        if remove_idx in keep:
                            keep.discard(remove_idx)
                            removed_count += 1
                            logger.debug(f"🗑️ Removed duplicate: IoU={iou:.3f} > {iou_threshold}")
                            
                except Exception as e:
                    logger.warning(f"⚠️ Error computing intersection for polygons {i}, {j}: {str(e)}")
                    continue

        dedup_time = time.time() - dedup_start
        keep_df_idx = [idx_map[k] for k in keep]
        result_df = df.iloc[keep_df_idx].reset_index(drop=True)
        
        # Filter masks to match the kept DataFrame rows
        result_masks = [masks[idx] for idx in keep_df_idx]
        
        total_time = time.time() - start_time
        logger.info(f"✅ Duplicate removal completed in {total_time:.3f}s")
        logger.info(f"📊 Removed {removed_count} duplicates, kept {len(result_df)}/{len(df)} detections")
        
        return result_df, result_masks
        
    except Exception as e:
        logger.error(f"❌ Error during duplicate removal: {str(e)}")
        raise

def add_masks(image, df, height, width, FOV, mask_json_path=None):
    """
    Add masks to an image by overlaying Ultralytics masks with logging.
    Now works with masks stored in JSON files instead of DataFrame.

    Args:
        image (numpy.ndarray): The original image.
        df (pandas.DataFrame): DataFrame containing position data (no mask data).
        height (int): Height of perspective views.
        width (int): Width of perspective views.
        FOV (int): Field of view in degrees.
        mask_json_path (str, optional): Path to JSON file containing mask data.

    Returns:
        numpy.ndarray: The image with overlaid masks.
    """
    logger.debug(f"🎨 Adding masks to image shape: {image.shape}")
    start_time = time.time()
    
    try:
        overlay = image.copy()
        img_shape = (image.shape[1], image.shape[0])
        processed_masks = 0
        
        # If no mask JSON path provided, just draw center points
        if mask_json_path is None or not os.path.exists(mask_json_path):
            logger.warning("⚠️ No mask JSON file provided or file doesn't exist, drawing center points only")
            for idx, row in df.iterrows():
                try:
                    # Draw center point
                    point = (int(row["image_x"]), int(row["image_y"]))
                    cv2.circle(image, point, 25, (0, 0, 255), -1)
                    processed_masks += 1
                except Exception as e:
                    logger.warning(f"⚠️ Error drawing center point {idx}: {str(e)}")
                    continue
        else:
            # Load mask data from JSON
            with open(mask_json_path, 'r') as f:
                mask_data = json.load(f)
            
            # Process each view's masks
            for view_name, view_masks in mask_data.get("views", {}).items():
                for mask_info in view_masks:
                    try:
                        mask_data_obj = mask_info["mask_data"]
                        tree_index = mask_info["tree_index"]
                        
                        # Find corresponding row in DataFrame by matching image_path
                        matching_row = None
                        if len(df) > 0:
                            mask_image_path = mask_info.get("image_path", "")
                            for idx, row in df.iterrows():
                                if row.get("image_path") == mask_image_path:
                                    matching_row = row
                                    break
                        
                        # Fallback to first row if no match found
                        if matching_row is None and len(df) > 0:
                            matching_row = df.iloc[0]
                            logger.warning(f"⚠️ No matching row found for mask {tree_index}, using first row")
                        
                        if matching_row is not None:
                            
                            # Deserialize mask (handles both RLE and polygon formats)
                            deserialized_mask = deserialize_ultralytics_mask(mask_data_obj)
                            
                            # Reconstruct mask points from deserialized data
                            if deserialized_mask.get("xy") and len(deserialized_mask["xy"]) > 0:
                                mask_points = deserialized_mask["xy"][0]
                                if isinstance(mask_points, np.ndarray) and len(mask_points) > 0:
                                    original_points = []
                                    
                                    for point in mask_points:
                                        orig_point = map_perspective_point_to_original(
                                            point[0], point[1], matching_row["theta"], img_shape, height, width, FOV
                                        )
                                        orig_point = tuple(map(int, orig_point))
                                        original_points.append(orig_point)

                                # Draw mask outline and fill
                                cv2.polylines(
                                    overlay,
                                    [np.array(original_points, np.int32)],
                                    isClosed=True,
                                    color=(0, 255, 0),
                                    thickness=5,
                                )
                                cv2.fillPoly(overlay, [np.array(original_points, np.int32)], color=(0, 255, 0))
                                cv2.addWeighted(overlay, 0.2, image, 0.8, 0, image)

                                # Draw center point
                                point = (int(matching_row["image_x"]), int(matching_row["image_y"]))
                                cv2.circle(image, point, 25, (0, 0, 255), -1)
                                
                                processed_masks += 1
                                
                    except Exception as e:
                        logger.warning(f"⚠️ Error processing mask {mask_info.get('tree_index', 'unknown')}: {str(e)}")
                        continue

        overlay_time = time.time() - start_time
        logger.debug(f"✅ Added {processed_masks} masks in {overlay_time:.3f}s")
        
        return image
        
    except Exception as e:
        logger.error(f"❌ Error adding masks to image: {str(e)}")
        raise