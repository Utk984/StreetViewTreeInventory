#!/usr/bin/env python3

import sys
import os
import time
import numpy as np
import pandas as pd
from math import radians, pi, cos, sin, asin, sqrt, atan2, tan, atan, exp, log, degrees
from scipy.cluster.hierarchy import linkage, fcluster
import logging

logger = logging.getLogger(__name__)

class MRFTreeTriangulator:
    """
    MRF-based triangulation for tree detection results.
    Adapted from Krylov et al. "Automatic Discovery and Geotagging of Objects from Street View Imagery"
    """
    
    def __init__(self, max_object_dist=15, max_cluster_dist=2.0, icm_iterations=15, 
                 depth_weight=0.3, multiview_weight=0.2):
        """
        Initialize MRF triangulator with parameters tuned for trees.
        
        Args:
            max_object_dist: Max distance from camera to trees (meters)
            max_cluster_dist: Max size of clusters (meters) 
            icm_iterations: Number of ICM optimization iterations
            depth_weight: Weight for depth consistency (alpha in paper)
            multiview_weight: Weight for multi-view preference (beta in paper)
        """
        self.max_object_dist = max_object_dist
        self.max_cluster_dist = max_cluster_dist
        self.icm_iterations = icm_iterations
        self.depth_weight = depth_weight
        self.multiview_weight = multiview_weight
        self.standalone_price = max(1 - depth_weight - multiview_weight, 0)
        
    def calculate_bearing(self, lat1, lon1, lat2, lon2):
        """Calculate bearing from point 1 to point 2 in degrees."""
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlon = lon2 - lon1
        y = sin(dlon) * cos(lat2)
        x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
        bearing = atan2(y, x)
        return (degrees(bearing) + 360) % 360
    
    def convert_tree_data_to_mrf_format(self, tree_df):
        """Convert TreeInventorization CSV to MRF input format."""
        logger.info("Converting tree detection data to MRF format")
        
        mrf_objects = []
        for _, row in tree_df.iterrows():
            camera_lat = row['stview_lat']
            camera_lon = row['stview_lng']
            tree_lat = row['tree_lat']
            tree_lng = row['tree_lng']
            depth = row['distance_pano']
            
            # Calculate bearing from street view to detected tree
            bearing = self.calculate_bearing(camera_lat, camera_lon, tree_lat, tree_lng)
            
            # Convert all points to meters FIRST for accurate ray intersection
            mx_cam, my_cam = self.lat_lon_to_meters(camera_lat, camera_lon)
            mx_tree, my_tree = self.lat_lon_to_meters(tree_lat, tree_lng)
            
            # Store METER coordinates for the ray and camera
            # (We use 'my' for lat-like and 'mx' for lon-like coordinates)
            latp1, lonp1 = my_tree, mx_tree   # Ray direction point (in meters)
            latp, lonp = my_tree, mx_tree     # Depth-based pos (in meters)
            cam_lat, cam_lon = my_cam, mx_cam # Camera pos (in meters)
            
            # Create MRF object entry (now contains METERS, despite old var names)
            # Format: (tree_my, tree_mx, bearing, depth, 0, cam_my, cam_mx, tree_my_dup, tree_mx_dup)
            mrf_objects.append((
                latp1,      # tree position y (meters)
                lonp1,      # tree position x (meters)
                bearing,    # bearing from camera to tree
                depth,      # depth estimate
                0,          # unused field
                cam_lat,    # camera position y (meters)
                cam_lon,    # camera position x (meters)
                latp,       # tree position y duplicate (meters)
                lonp        # tree position x duplicate (meters)
            ))
        
        logger.info(f"Converted {len(mrf_objects)} tree detections to MRF format")
        return mrf_objects

    def lat_lon_to_meters(self, lat, lon):
        """Convert lat/lon to meters using Spherical Mercator projection."""
        origin_shift = 2 * pi * 6378137 / 2.0
        mx = lon * origin_shift / 180.0
        my = log(tan((90 + lat) * pi / 360.0)) / (pi / 180.0)
        my = my * origin_shift / 180.0
        return mx, my

    def meters_to_lat_lon(self, mx, my):
        """Convert meters to lat/lon using Spherical Mercator projection."""
        origin_shift = 2 * pi * 6378137 / 2.0
        lon = (mx / origin_shift) * 180.0
        lat = (my / origin_shift) * 180.0
        lat = 180 / pi * (2 * atan(exp(lat * pi / 180.0)) - pi / 2.0)
        return lat, lon

    def haversine_distance(self, lon1, lat1, lon2, lat2):
        """Calculate haversine distance between two GPS points in meters."""
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        return 6367000.0 * c

    def calculate_ray_intersection(self, object1, object2):
        """Calculate intersection point between two observation rays (now in METERS)."""
        # Camera positions (now in METERS)
        my_c1, mx_c1 = object1[5], object1[6]  # my = lat-like, mx = lon-like
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
        
        denominator = a2 * b1 - b2 * a1
        if abs(denominator) < 1e-10:
            return -1, -1, 0, 0  # Parallel rays
        
        y = (a1 * c2 - a2 * c1) / denominator  # This 'y' is the scaling factor for ray 2
        
        if abs(a1) > 1e-10:
            x = (b1 * y + c1) / a1  # This 'x' is the scaling factor for ray 1
        else:
            x = (b2 * y + c2) / a2
        
        if x < 0 or y < 0:
            return -2, -2, 0, 0  # Invalid intersection (behind cameras)
        
        # Intersection point (in METERS)
        intersect_my = a1 * x + my_c1
        intersect_mx = a2 * x + mx_c1
        
        # Check real distances
        dist1 = sqrt((intersect_mx - mx_c1)**2 + (intersect_my - my_c1)**2)
        dist2 = sqrt((intersect_mx - mx_c2)**2 + (intersect_my - my_c2)**2)
        
        if dist1 > self.max_object_dist or dist2 > self.max_object_dist:
            return -3, -3, 0, 0  # Too far
        
        # Return distances (dist1, dist2) and intersection (mx, my)
        return dist1, dist2, intersect_mx, intersect_my

    def calculate_mrf_energy(self, objects_dist, objects_base, connectivity, obj_idx):
        """Calculate MRF energy for an object configuration."""
        intersections = np.count_nonzero(connectivity[obj_idx, :])
        
        if intersections == 0:
            return self.standalone_price
        
        energy = 0
        depth_min, depth_max = 1000, 0
        
        for i in range(len(objects_base)):
            if connectivity[obj_idx, i]:
                # Depth consistency penalty
                depth_penalty = self.depth_weight * abs(objects_dist[obj_idx, i] - objects_base[obj_idx][3])
                energy += depth_penalty
                
                depth = objects_dist[obj_idx, i]
                depth_min = min(depth_min, depth)
                depth_max = max(depth_max, depth)
        
        # Multi-view consistency penalty
        energy += self.multiview_weight * (depth_max - depth_min)
        return energy

    def calculate_average_position(self, intersects, connectivity, obj_idx):
        """Calculate averaged object location after clustering."""
        result = np.zeros(2)
        count = 0
        
        for i in range(intersects.shape[0]):
            if connectivity[obj_idx, i]:
                result += intersects[obj_idx, i, :]
                count += 1
        
        return result / count if count > 0 else result

    def hierarchical_clustering(self, intersects, max_distance):
        """Perform hierarchical clustering on intersection points."""
        if len(intersects) == 0:
            return np.array([]).reshape(0, 3)
            
        Z = linkage(np.asarray(intersects))
        clusters = fcluster(Z, max_distance, criterion='distance') - 1
        num_clusters = max(clusters) + 1
        
        cluster_results = np.zeros((num_clusters, 3))
        for i, cluster_id in enumerate(clusters):
            cluster_results[cluster_id, 0] += intersects[i][0]
            cluster_results[cluster_id, 1] += intersects[i][1]
            cluster_results[cluster_id, 2] += 1
            
        return cluster_results

    def triangulate_trees(self, tree_df):
        """Main triangulation function using MRF optimization."""
        logger.info("Starting MRF-based tree triangulation")
        start_time = time.time()
        
        # Store original dataframe for preserving columns and for camera distance calculations
        original_df = tree_df.copy().reset_index(drop=True)
        self._original_df = original_df  # Store for camera distance calculations
        
        # Convert data format
        objects_base = self.convert_tree_data_to_mrf_format(tree_df)
        logger.info(f"Processing {len(objects_base)} tree detections")
        
        if len(objects_base) < 2:
            logger.warning("Need at least 2 detections for triangulation")
            # Just add score=1 to all trees and return
            result_df = original_df.copy()
            result_df['score'] = 1
            return result_df
        
        # Calculate pairwise intersections
        max_cam_dist = 1.5 * self.max_object_dist
        objects_dist = np.zeros((len(objects_base), len(objects_base)))
        intersects = np.zeros((len(objects_base), len(objects_base), 2))
        num_intersects = 0
        
        logger.info("Calculating ray intersections...")
        for i in range(len(objects_base)):
            if i % 100 == 0 and i > 0:
                logger.info(f'Processed {i} objects ({100.0*i/len(objects_base):.1f}%)')
                
            objects_dist[i, i] = -5
            
            for j in range(i + 1, len(objects_base)):
                # Check camera distance using original lat/lon from dataframe
                cam_dist = self.haversine_distance(
                    self._original_df.iloc[i]['stview_lng'], self._original_df.iloc[i]['stview_lat'],
                    self._original_df.iloc[j]['stview_lng'], self._original_df.iloc[j]['stview_lat']
                )
                
                if cam_dist < 0.5 or cam_dist > max_cam_dist:
                    objects_dist[i, j] = objects_dist[j, i] = -4
                    continue
                
                # Calculate ray intersection (now returns dist1, dist2, mx, my in METERS)
                dist1, dist2, mx, my = self.calculate_ray_intersection(objects_base[i], objects_base[j])
                objects_dist[i, j] = objects_dist[j, i] = dist1
                # intersects array now stores METER coordinates (mx, my)
                intersects[i, j, :] = intersects[j, i, :] = [mx, my]
                
                if dist1 > 0:
                    num_intersects += 1
        
        logger.info(f"Found {num_intersects} valid intersections")
        
        # Initialize connectivity matrix
        connectivity = np.zeros((len(objects_base), len(objects_base)), dtype=np.uint8)
        viable_options = np.array([np.count_nonzero(objects_dist[i, :] > 0) 
                                  for i in range(len(objects_base))])
        
        # ICM optimization
        logger.info("Running ICM optimization...")
        np.random.seed(int(1000000.0 * time.time()) % 1000000000)
        changes = 0
        
        total_iterations = self.icm_iterations * len(objects_base)
        for iteration in range(total_iterations):
            if (iteration + 1) % len(objects_base) == 0:
                epoch = (iteration + 1) // len(objects_base)
                logger.info(f'ICM Iteration #{epoch}: accepted {changes} changes')
                changes = 0
            
            test_obj = np.random.randint(0, len(objects_base))
            if viable_options[test_obj] == 0:
                continue
            
            # Select random viable pairing
            rand_num = 1 + np.random.randint(0, viable_options[test_obj])
            count = 0
            for i in range(len(objects_base)):
                if objects_dist[test_obj, i] > 0:
                    count += 1
                    if count == rand_num:
                        test_pair = i
                        break
            
            # Calculate energy before change
            energy_old = (self.calculate_mrf_energy(objects_dist, objects_base, connectivity, test_obj) +
                         self.calculate_mrf_energy(objects_dist, objects_base, connectivity, test_pair))
            
            # Toggle connection
            connectivity[test_obj, test_pair] = 1 - connectivity[test_obj, test_pair]
            connectivity[test_pair, test_obj] = 1 - connectivity[test_pair, test_obj]
            
            # Calculate energy after change
            energy_new = (self.calculate_mrf_energy(objects_dist, objects_base, connectivity, test_obj) +
                         self.calculate_mrf_energy(objects_dist, objects_base, connectivity, test_pair))
            
            # Accept or reject change
            if energy_new <= energy_old:
                changes += 1
            else:
                # Revert change
                connectivity[test_obj, test_pair] = 1 - connectivity[test_obj, test_pair]
                connectivity[test_pair, test_obj] = 1 - connectivity[test_pair, test_obj]
        
        # Extract final intersections and find unique groups of connected objects
        logger.info("Extracting final triangulated positions...")
        icm_intersections = []
        triangulated_groups = []  # Track unique groups of connected objects
        triangulated_objects = set()
        processed_objects = set()
        
        for i in range(len(objects_base)):
            if i in processed_objects:
                continue  # Skip if already processed as part of another group
                
            avg_pos = self.calculate_average_position(intersects, connectivity, i)
            if avg_pos[0] != 0 and avg_pos[1] != 0:
                icm_intersections.append((avg_pos[0], avg_pos[1]))
                # Find all connected objects for this group (including i itself)
                group = [i]
                processed_objects.add(i)
                triangulated_objects.add(i)
                
                for j in range(len(objects_base)):
                    if connectivity[i, j] and j not in processed_objects:
                        group.append(j)
                        processed_objects.add(j)
                        triangulated_objects.add(j)
                
                if len(group) >= 2:  # Only keep groups with 2+ objects
                    triangulated_groups.append(group)
                    logger.debug(f"Group {len(triangulated_groups)}: {group} objects")
        
        logger.info(f"Found {len(icm_intersections)} triangulated positions")
        logger.info(f"Found {len(triangulated_groups)} unique triangulated groups")
        logger.info(f"Objects involved in triangulations: {len(triangulated_objects)}")
        
        # Log group details
        for i, group in enumerate(triangulated_groups):
            logger.info(f"Group {i+1}: {len(group)} objects (indices: {group})")
        
        # Start with original dataframe and modify it
        result_df = original_df.copy()
        result_df['score'] = 1  # Default score for all trees
        rows_to_remove = []  # Track rows to remove (duplicates)
        
        # Process each triangulated group (no hierarchical clustering needed)
        logger.info("Processing triangulated groups...")
        for group_idx, group in enumerate(triangulated_groups):
            if len(group) < 2:
                continue  # Skip single-object groups
            
            # Calculate average position from all pairwise intersections within this group
            intersection_points = []
            for i in group:
                for j in group:
                    if i < j and connectivity[i, j]:
                        # Get the intersection point between objects i and j (in METERS)
                        int_mx, int_my = intersects[i, j, :]
                        if int_mx != 0 and int_my != 0:
                            intersection_points.append((int_mx, int_my))
            
            if not intersection_points:
                logger.warning(f"Group {group_idx+1}: No valid intersections found")
                continue
            
            # Average all intersection points (in METERS)
            avg_mx = np.mean([p[0] for p in intersection_points])
            avg_my = np.mean([p[1] for p in intersection_points])
            
            # Convert FINAL average back to lat/lon
            avg_lat, avg_lon = self.meters_to_lat_lon(avg_mx, avg_my)
            score = len(group)
            
            # Find the best representative row (highest confidence)
            best_idx = max(group, key=lambda idx: original_df.loc[idx, 'conf'])
            
            logger.debug(f"Group {group_idx+1}: {len(group)} detections, {len(intersection_points)} intersections")
            logger.debug(f"  Best detection: idx={best_idx} (conf={original_df.loc[best_idx, 'conf']:.3f})")
            logger.debug(f"  Triangulated position: ({avg_lat:.6f}, {avg_lon:.6f})")
            
            # Update the best row with triangulated coordinates
            result_df.loc[best_idx, 'tree_lat'] = avg_lat
            result_df.loc[best_idx, 'tree_lng'] = avg_lon
            result_df.loc[best_idx, 'score'] = score
            
            # Mark other rows in group for removal
            for idx in group:
                if idx != best_idx:
                    rows_to_remove.append(idx)
                    logger.debug(f"  Marking duplicate {idx} for removal")
        
        # Remove duplicate rows
        if rows_to_remove:
            result_df = result_df.drop(rows_to_remove).reset_index(drop=True)
            logger.info(f"✅ Removed {len(rows_to_remove)} duplicate detections")
        else:
            logger.warning("⚠️ No duplicate detections removed!")
        
        # Count final statistics
        multi_view_trees = len(result_df[result_df['score'] >= 2])
        single_view_trees = len(result_df[result_df['score'] == 1])
        original_count = len(original_df)
        reduction = original_count - len(result_df)
        
        elapsed_time = time.time() - start_time
        logger.info(f"MRF triangulation completed in {elapsed_time:.2f} seconds")
        logger.info(f"Results: {original_count} → {len(result_df)} trees (removed {reduction} duplicates)")
        logger.info(f"  - {multi_view_trees} multi-view trees (triangulated, score ≥ 2)")
        logger.info(f"  - {single_view_trees} single-view trees (original positions, score = 1)")
        
        # Validation check
        expected_duplicates = sum(len(group) - 1 for group in triangulated_groups)
        if reduction != expected_duplicates:
            logger.warning(f"⚠️ Expected to remove {expected_duplicates} duplicates, but removed {reduction}")
        
        return result_df


def triangulate_tree_detections(input_csv_path, output_csv_path, **kwargs):
    """
    Main function to perform MRF triangulation on tree detection results.
    
    This function merges duplicate detections while preserving the original CSV format:
    - Multi-view trees (seen from 2+ panoramas): Best detection kept with triangulated tree_lat/tree_lng (score ≥ 2)
    - Single-view trees: Kept with original data (score = 1) 
    - Output has same columns as input + 'score' column
    - Duplicate rows are removed
    
    Args:
        input_csv_path: Path to TreeInventorization output CSV
        output_csv_path: Path to save triangulated results
        **kwargs: Parameters for MRFTreeTriangulator
    """
    logger.info(f"Starting tree triangulation: {input_csv_path} -> {output_csv_path}")
    
    # Load tree detection data
    tree_df = pd.read_csv(input_csv_path)
    logger.info(f"Loaded {len(tree_df)} tree detections from {len(tree_df['pano_id'].unique())} panoramas")
    
    # Initialize triangulator
    triangulator = MRFTreeTriangulator(**kwargs)
    
    # Perform triangulation
    triangulated_trees = triangulator.triangulate_trees(tree_df)
    
    # Save results
    triangulated_trees.to_csv(output_csv_path, index=False)
    logger.info(f"Saved {len(triangulated_trees)} triangulated trees to {output_csv_path}")
    
    return triangulated_trees


if __name__ == "__main__":
    # Example usage
    input_csv = "outputs/chandigarh_trees.csv"
    output_csv = "outputs/chandigarh_trees_triangulated.csv"
    
    triangulated = triangulate_tree_detections(
        input_csv, output_csv,
        max_object_dist=15,
        max_cluster_dist=2.0,
        icm_iterations=15
    )
    
    print(f"Triangulation complete: {len(triangulated)} trees found")
