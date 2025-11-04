#!/usr/bin/env python3
"""
Debug version of MRF triangulation following Krylov et al. exactly.
Adds extensive logging to verify inputs match expected format.
"""

import numpy as np
import pandas as pd
import logging
from math import radians, pi, cos, sin, asin, sqrt, atan2, tan, atan, exp, log, degrees
from scipy.cluster.hierarchy import linkage, fcluster
import time

logger = logging.getLogger(__name__)

def lat_lon_to_meters(lat, lon):
    """Converts given lat/lon in WGS84 Datum to XY in Spherical Mercator EPSG:4326"""
    origin_shift = 2 * pi * 6378137 / 2.0
    mx = lon * origin_shift / 180.0
    my = log(tan((90 + lat) * pi / 360.0)) / (pi / 180.0)
    my = my * origin_shift / 180.0
    return mx, my

def meters_to_lat_lon(mx, my):
    """Converts XY point from Spherical Mercator EPSG:4326 to lat/lon in WGS84 Datum"""
    origin_shift = 2 * pi * 6378137 / 2.0
    lon = (mx / origin_shift) * 180.0
    lat = (my / origin_shift) * 180.0
    lat = 180 / pi * (2 * atan(exp(lat * pi / 180.0)) - pi / 2.0)
    return lat, lon

def calculate_bearing(lat1, lon1, lat2, lon2):
    """Calculate bearing from point 1 to point 2 in degrees (clockwise from North)"""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    y = sin(dlon) * cos(lat2)
    x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
    bearing = atan2(y, x)
    return (degrees(bearing) + 360) % 360

def prepare_mrf_input(tree_df):
    """
    Prepare MRF input following original Krylov et al. format.
    
    Original input: camera_lat, camera_lon, bearing, depth
    
    Returns: List of tuples (latp1, lonp1, bearing, depth, 0, camera_lat, camera_lon, latp, lonp)
    where:
        - latp1, lonp1: normalized position (1m from camera in bearing direction)
        - bearing: bearing from camera to tree (degrees clockwise from North)
        - depth: estimated distance from camera to tree (meters)
        - camera_lat, camera_lon: camera GPS position
        - latp, lonp: depth-based tree position
    """
    logger.info("=" * 60)
    logger.info("STEP 1: PREPARING MRF INPUT DATA")
    logger.info("=" * 60)
    
    objects_base = []
    
    for idx, row in tree_df.iterrows():
        # Extract inputs
        camera_lat = row['stview_lat']
        camera_lon = row['stview_lng']
        tree_lat = row['tree_lat']
        tree_lng = row['tree_lng']
        depth = row['distance_pano']
        
        # Calculate bearing from camera to tree
        bearing = calculate_bearing(camera_lat, camera_lon, tree_lat, tree_lng)
        
        # Debug: log first few entries
        if idx < 3:
            logger.info(f"\nEntry {idx}:")
            logger.info(f"  Camera: ({camera_lat:.6f}, {camera_lon:.6f})")
            logger.info(f"  Tree:   ({tree_lat:.6f}, {tree_lng:.6f})")
            logger.info(f"  Bearing: {bearing:.2f}° (clockwise from North)")
            logger.info(f"  Depth: {depth:.2f}m")
        
        # --- FIX: Convert all points to meters FIRST ---
        mx_cam, my_cam = lat_lon_to_meters(camera_lat, camera_lon)
        mx_tree, my_tree = lat_lon_to_meters(tree_lat, tree_lng)

        # Store METER coordinates for the ray and camera
        # (We use 'my' for lat-like and 'mx' for lon-like coordinates)
        latp1, lonp1 = my_tree, mx_tree   # Ray direction point (in meters)
        latp, lonp = my_tree, mx_tree     # Depth-based pos (in meters)
        cam_lat, cam_lon = my_cam, mx_cam # Camera pos (in meters)
        
        if idx < 3:
            logger.info(f"  Camera (meters): ({mx_cam:.2f}, {my_cam:.2f})")
            logger.info(f"  Tree (meters):   ({mx_tree:.2f}, {my_tree:.2f})")
        
        # Create object tuple (it now contains METERS, despite old var names)
        # (tree_my, tree_mx, bearing, depth, 0, cam_my, cam_mx, tree_my_dup, tree_mx_dup)
        objects_base.append((latp1, lonp1, bearing, depth, 0, cam_lat, cam_lon, latp, lonp))
    
    logger.info(f"\n✅ Prepared {len(objects_base)} objects in MRF format")
    logger.info(f"   Bearing range: {min(obj[2] for obj in objects_base):.1f}° - {max(obj[2] for obj in objects_base):.1f}°")
    logger.info(f"   Depth range: {min(obj[3] for obj in objects_base):.1f}m - {max(obj[3] for obj in objects_base):.1f}m")
    
    return objects_base

def haversine(lon1, lat1, lon2, lat2):
    """Calculate great circle distance between two points (decimal degrees) -> meters"""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return 6367000.0 * c

def intersect(object1, object2, max_object_dist):
    """Calculate intersection point between two rays (camera + bearing)"""
    # Camera positions (now in METERS)
    my_c1, mx_c1 = object1[5], object1[6] # my = lat-like, mx = lon-like
    my_c2, mx_c2 = object2[5], object2[6]
    
    # Object positions (now in METERS)
    my_p1, mx_p1 = object1[0], object1[1]
    my_p2, mx_p2 = object2[0], object2[1]
    
    # Ray direction vectors (using 'y' for lat/my, 'x' for lon/mx)
    a1 = my_p1 - my_c1  # dy1
    b1 = my_p2 - my_c2  # dy2
    c1 = my_c2 - my_c1

    a2 = mx_p1 - mx_c1  # dx1
    b2 = mx_p2 - mx_c2  # dx2
    c2 = mx_c2 - mx_c1
    
    denominator = a2*b1 - b2*a1
    if abs(denominator) < 1e-10:
        return -1, -1, 0, 0  # Parallel rays
    
    y = (a1*c2 - a2*c1) / denominator # This 'y' is the scaling factor for ray 2
    
    if abs(a1) > 1e-10:
        x = (b1*y + c1) / a1  # This 'x' is the scaling factor for ray 1
    else:
        x = (b2*y + c2) / a2
    
    if x < 0 or y < 0:
        return -2, -2, 0, 0  # Invalid intersection (behind cameras)
    
    # Intersection point (in METERS)
    intersect_my = a1*x + my_c1
    intersect_mx = a2*x + mx_c1
    
    # Check real distances
    dist1 = sqrt((intersect_mx - mx_c1)**2 + (intersect_my - my_c1)**2)
    dist2 = sqrt((intersect_mx - mx_c2)**2 + (intersect_my - my_c2)**2)

    if dist1 > max_object_dist or dist2 > max_object_dist:
         return -3, -3, 0, 0  # Too far

    # Return distances (dist1, dist2) and intersection (mx, my)
    return dist1, dist2, intersect_mx, intersect_my

def debug_triangulate(input_csv, output_csv,
                     max_object_dist=15.0,
                     max_cluster_dist=2.0,
                     icm_iterations=15,
                     depth_weight=0.3,
                     multiview_weight=0.2):
    """
    MRF triangulation with extensive debugging following original Krylov et al. algorithm.
    """
    
    logger.info("=" * 80)
    logger.info("KRYLOV ET AL. MRF TRIANGULATION - DEBUG MODE")
    logger.info("=" * 80)
    
    start_time = time.time()
    
    # Load data
    tree_df = pd.read_csv(input_csv)
    logger.info(f"Loaded {len(tree_df)} tree detections from {len(tree_df['pano_id'].unique())} panoramas")
    
    # Prepare MRF input
    objects_base = prepare_mrf_input(tree_df)
    
    # Calculate standalone price
    standalone_price = max(1 - depth_weight - multiview_weight, 0)
    logger.info(f"\nMRF Parameters:")
    logger.info(f"  Max object distance: {max_object_dist}m")
    logger.info(f"  Max cluster distance: {max_cluster_dist}m")
    logger.info(f"  ICM iterations: {icm_iterations}")
    logger.info(f"  Depth weight (α): {depth_weight}")
    logger.info(f"  Multiview weight (β): {multiview_weight}")
    logger.info(f"  Standalone price (1-α-β): {standalone_price}")
    
    # Find admissible intersections
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: FINDING ADMISSIBLE RAY INTERSECTIONS")
    logger.info("=" * 60)
    
    max_cam_dist = 1.5 * max_object_dist
    objects_dist = np.zeros((len(objects_base), len(objects_base)))
    intersects = np.zeros((len(objects_base), len(objects_base), 2))
    num_intersects = 0
    
    logger.info(f"Max camera distance: {max_cam_dist}m")
    
    for i in range(len(objects_base)):
        if i % 100 == 0 and i > 0:
            logger.info(f'Processed {i}/{len(objects_base)} objects ({100.0*i/len(objects_base):.1f}%)')
        
        objects_dist[i, i] = -5
        
        for j in range(i+1, len(objects_base)):
            # Check camera distance
            # --- FIX: haversine needs lat/lon, get it from original df ---
            cam_dist = haversine(tree_df.iloc[i]['stview_lng'], tree_df.iloc[i]['stview_lat'],
                               tree_df.iloc[j]['stview_lng'], tree_df.iloc[j]['stview_lat'])
            
            # Filter: cameras too close or too far
            if cam_dist < 0.5 or cam_dist > max_cam_dist:
                objects_dist[i, j] = objects_dist[j, i] = -4
                continue
            
            # Calculate ray intersection (now in METERS)
            dist1, dist2, mx, my = intersect(objects_base[i], objects_base[j], max_object_dist)
            
            # Store distance to intersection (dist1)
            objects_dist[i, j] = objects_dist[j, i] = dist1
            
            # intersects array now stores METER coordinates (mx, my)
            intersects[i, j, :] = intersects[j, i, :] = [mx, my]
            
            if dist1 > 0:
                num_intersects += 1
                if num_intersects <= 3:  # Log first few
                    logger.info(f"\nIntersection {num_intersects}:")
                    logger.info(f"  Objects: {i} ↔ {j}")
                    logger.info(f"  Camera distance: {cam_dist:.2f}m")
                    logger.info(f"  Ray distances: cam1→tree={dist1:.2f}m, cam2→tree={dist2:.2f}m")
                    logger.info(f"  Intersection point (meters): ({mx:.2f}, {my:.2f})")
    
    logger.info(f"\n✅ Found {num_intersects} admissible intersections out of {len(objects_base)*(len(objects_base)-1)//2} possible pairs")
    logger.info(f"   Intersection rate: {100.0*num_intersects/(len(objects_base)*(len(objects_base)-1)//2):.2f}%")
    
    # Initialize connectivity
    connectivity = np.zeros((len(objects_base), len(objects_base)), dtype=np.uint8)
    viable_options = np.array([np.count_nonzero(objects_dist[i, :] > 0) 
                               for i in range(len(objects_base))])
    
    objects_with_connections = np.sum(viable_options > 0)
    logger.info(f"   Objects with possible connections: {objects_with_connections}/{len(objects_base)}")
    
    # ICM Optimization
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: ICM OPTIMIZATION")
    logger.info("=" * 60)
    
    np.random.seed(int(100000.0 * time.time()) % 1000000000)
    total_changes = 0
    
    for iteration in range(icm_iterations * len(objects_base)):
        if (iteration + 1) % len(objects_base) == 0:
            epoch = (iteration + 1) // len(objects_base)
            logger.info(f'ICM Iteration #{epoch}/{icm_iterations}: accepted {total_changes} changes so far')
        
        test_obj = np.random.randint(0, len(objects_base))
        if viable_options[test_obj] == 0:
            continue
        
        # Select random viable pair
        rand_num = 1 + np.random.randint(0, viable_options[test_obj])
        count = 0
        for i in range(len(objects_base)):
            if objects_dist[test_obj, i] > 0:
                count += 1
                if count == rand_num:
                    test_pair = i
                    break
        
        # Calculate energy before toggle
        energy_old = (calculate_energy(objects_dist, objects_base, connectivity, test_obj,
                                      depth_weight, multiview_weight, standalone_price) +
                     calculate_energy(objects_dist, objects_base, connectivity, test_pair,
                                    depth_weight, multiview_weight, standalone_price))
        
        # Toggle connection
        connectivity[test_obj, test_pair] = 1 - connectivity[test_obj, test_pair]
        connectivity[test_pair, test_obj] = 1 - connectivity[test_pair, test_obj]
        
        # Calculate energy after toggle
        energy_new = (calculate_energy(objects_dist, objects_base, connectivity, test_obj,
                                      depth_weight, multiview_weight, standalone_price) +
                     calculate_energy(objects_dist, objects_base, connectivity, test_pair,
                                    depth_weight, multiview_weight, standalone_price))
        
        # Accept or revert
        if energy_new <= energy_old:
            total_changes += 1
        else:
            # Revert
            connectivity[test_obj, test_pair] = 1 - connectivity[test_obj, test_pair]
            connectivity[test_pair, test_obj] = 1 - connectivity[test_pair, test_obj]
    
    logger.info(f"✅ ICM completed: {total_changes} connections established")
    
    # Extract final groups and triangulated positions
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: EXTRACTING TRIANGULATED GROUPS")
    logger.info("=" * 60)
    
    triangulated_groups = []
    triangulated_objects = set()
    processed_objects = set()
    
    for i in range(len(objects_base)):
        if i in processed_objects:
            continue
        
        # Find all connected objects for this group
        group = [i]
        processed_objects.add(i)
        
        for j in range(len(objects_base)):
            if connectivity[i, j] and j not in processed_objects:
                group.append(j)
                processed_objects.add(j)
        
        if len(group) >= 2:
            triangulated_groups.append(group)
            for obj_idx in group:
                triangulated_objects.add(obj_idx)
    
    logger.info(f"Found {len(triangulated_groups)} unique groups")
    logger.info(f"Objects involved in groups: {len(triangulated_objects)}")
    
    for i, group in enumerate(triangulated_groups):
        logger.info(f"  Group {i+1}: {len(group)} objects (indices: {group})")
    
    # Calculate triangulated positions by averaging intersections within each group
    logger.info("\n" + "=" * 60)
    logger.info("STEP 5: AVERAGING INTERSECTIONS PER GROUP")
    logger.info("=" * 60)
    
    final_trees = []
    for group_idx, group in enumerate(triangulated_groups):
        # Get all pairwise intersections within this group
        intersection_points = []
        for i in group:
            for j in group:
                if i < j and connectivity[i, j]:
                    int_lat, int_lon = intersects[i, j, :]
                    if int_lat != 0 and int_lon != 0:
                        intersection_points.append((int_lat, int_lon))
        
        if not intersection_points:
            logger.warning(f"Group {group_idx+1}: No valid intersections")
            continue
        
        # --- FIX: Average METER coordinates ---
        # Average all intersection points (which are mx, my)
        avg_mx = np.mean([p[0] for p in intersection_points])
        avg_my = np.mean([p[1] for p in intersection_points])
        score = len(group)
        
        # --- FIX: Convert FINAL average back to lat/lon ---
        avg_lat, avg_lon = meters_to_lat_lon(avg_mx, avg_my)
        
        logger.info(f"Group {group_idx+1}: {len(group)} detections → {len(intersection_points)} intersections")
        logger.info(f"  Triangulated (meters): ({avg_mx:.2f}, {avg_my:.2f})")
        logger.info(f"  Triangulated (lat/lon): ({avg_lat:.6f}, {avg_lon:.6f}), Score: {score}")
        
        final_trees.append({
            'lat': avg_lat,
            'lon': avg_lon,
            'score': score
        })
    
    result_df = pd.DataFrame(final_trees)
    result_df.to_csv(output_csv, index=False)
    
    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 80)
    logger.info("FINAL RESULTS")
    logger.info("=" * 80)
    logger.info(f"Input detections: {len(objects_base)}")
    logger.info(f"Admissible intersections: {num_intersects}")
    logger.info(f"ICM connections: {total_changes}")
    logger.info(f"Triangulated groups: {len(triangulated_groups)}")
    logger.info(f"Objects in groups: {len(triangulated_objects)}")
    logger.info(f"Final multi-view trees: {len(final_trees)}")
    logger.info(f"Single-view trees (not in output): {len(objects_base) - len(triangulated_objects)}")
    logger.info(f"Processing time: {elapsed:.2f}s")
    logger.info("=" * 80)
    
    return result_df

def calculate_energy(objects_dist, objects_base, connectivity, obj_idx,
                    depth_weight, multiview_weight, standalone_price):
    """Calculate MRF energy for an object"""
    connections = np.count_nonzero(connectivity[obj_idx, :])
    if connections == 0:
        return standalone_price
    
    energy = 0
    depth_min, depth_max = 1000, 0
    
    for i in range(len(objects_base)):
        if connectivity[obj_idx, i]:
            # Depth consistency penalty
            depth_penalty = depth_weight * abs(objects_dist[obj_idx, i] - objects_base[obj_idx][3])
            energy += depth_penalty
            
            depth = objects_dist[obj_idx, i]
            depth_min = min(depth_min, depth)
            depth_max = max(depth_max, depth)
    
    return energy + multiview_weight * (depth_max - depth_min)

def calculate_avg_position(intersects, connectivity, obj_idx):
    """Calculate averaged object location"""
    result = np.zeros(2)
    count = 0
    
    for i in range(intersects.shape[0]):
        if connectivity[obj_idx, i]:
            result += intersects[obj_idx, i, :]
            count += 1
    
    return result / count if count > 0 else result


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )
    
    # Run debug triangulation
    input_csv = "outputs/chandigarh_trees.csv"
    output_csv = "outputs/chandigarh_trees_debug.csv"
    
    result = debug_triangulate(
        input_csv, output_csv,
        max_object_dist=20.0,
        max_cluster_dist=3.0,
        icm_iterations=20, 
        depth_weight=0.3,
        multiview_weight=0.2
    )
    
    print(f"\n✅ Debug triangulation complete!")
    print(f"Results saved to: {output_csv}")
    print(f"Found {len(result)} high-confidence trees")

