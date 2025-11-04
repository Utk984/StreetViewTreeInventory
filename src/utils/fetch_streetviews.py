#!/usr/bin/env python3
"""
Fetch Street View panoramas by sampling points along OSM road network.
Simple approach: Extract roads → Sample points every 1m → Check Street View → Save.
"""

import asyncio
import json
import csv
from pathlib import Path
from typing import List, Tuple, Set
from datetime import datetime

import aiohttp
from shapely.geometry import shape, LineString, Point
from shapely.ops import unary_union
import geopandas as gpd
import osmnx as ox
from tqdm.asyncio import tqdm as atqdm
from tqdm import tqdm
import streetlevel.streetview as streetview


class OSMStreetViewFetcher:
    """Fetch Street View panoramas by sampling OSM road network."""
    
    def __init__(self, geojson_path: str, output_csv: str, sample_distance_m: float = 1.0):
        """
        Initialize fetcher.
        
        Args:
            geojson_path: Path to GeoJSON boundary file
            output_csv: Output CSV path
            sample_distance_m: Distance between sample points in meters (default: 1.0)
        """
        self.geojson_path = Path(geojson_path)
        self.output_csv = Path(output_csv)
        self.sample_distance_m = sample_distance_m
        
        # Load boundary
        print(f"Loading boundary from {geojson_path}...")
        with open(geojson_path) as f:
            geojson_data = json.load(f)
        
        if geojson_data['type'] == 'FeatureCollection':
            geometries = [shape(f['geometry']) for f in geojson_data['features']]
            self.boundary = unary_union(geometries)
        else:
            self.boundary = shape(geojson_data['geometry'])
        
        self.bounds = self.boundary.bounds
        print(f"Boundary: ({self.bounds[0]:.4f}, {self.bounds[1]:.4f}) to ({self.bounds[2]:.4f}, {self.bounds[3]:.4f})")
        
        # Track discovered panoramas (dedupe) - using set ensures no duplicates
        self.discovered_panos: Set[str] = set()
        self.pano_lock = asyncio.Lock()
        self.results = []  # Will only contain unique panos due to discovered_panos check
    
    def extract_road_network(self) -> gpd.GeoDataFrame:
        """
        Extract OSM road network within boundary.
        
        Returns:
            GeoDataFrame with road geometries
        """
        print("\n=== Extracting OSM road network ===")
        
        # Get road network from OSM
        # Use boundary polygon directly
        try:
            print("Fetching road network from OSM...")
            G = ox.graph_from_polygon(
                self.boundary,
                network_type='drive',  # Driveable roads (where Street View cars go)
                simplify=True
            )
            
            print(f"✓ Fetched {len(G.nodes)} nodes, {len(G.edges)} edges")
            
            # Convert to GeoDataFrame
            edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
            
            # Project to UTM for accurate distance calculations
            edges_projected = edges.to_crs(epsg=32643)  # WGS 84 / UTM zone 43N (for Delhi)
            
            print(f"✓ Road network: {len(edges)} road segments")
            print(f"  Total length: {edges_projected.length.sum() / 1000:.1f} km")
            
            # Return original (lat/lon) for point sampling
            return edges
            
        except Exception as e:
            print(f"Error fetching OSM data: {e}")
            print("Falling back to bounding box query...")
            
            # Fallback: use bounding box
            G = ox.graph_from_bbox(
                north=self.bounds[3],
                south=self.bounds[1],
                east=self.bounds[2],
                west=self.bounds[0],
                network_type='drive',
                simplify=True
            )
            
            edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
            
            # Filter to boundary
            edges = edges[edges.intersects(self.boundary)]
            
            # Project for accurate length
            edges_projected = edges.to_crs(epsg=32643)
            
            print(f"✓ Road network (filtered): {len(edges)} road segments")
            print(f"  Total length: {edges_projected.length.sum() / 1000:.1f} km")
            
            return edges
    
    def sample_road_points(self, roads: gpd.GeoDataFrame) -> List[Tuple[float, float]]:
        """
        Sample points along roads at regular intervals.
        
        Args:
            roads: GeoDataFrame with road geometries (in lat/lon)
            
        Returns:
            List of (lat, lon) tuples
        """
        print(f"\n=== Sampling points every {self.sample_distance_m}m along roads ===")
        
        # Project to UTM for accurate distance calculations
        roads_projected = roads.to_crs(epsg=32643)  # UTM zone 43N for Delhi
        
        points = []
        total_length_m = 0
        
        for (idx, row), (idx_proj, row_proj) in tqdm(
            zip(roads.iterrows(), roads_projected.iterrows()), 
            total=len(roads), 
            desc="Sampling roads"
        ):
            geom = row.geometry  # Original lat/lon geometry
            geom_proj = row_proj.geometry  # Projected geometry
            
            # Handle both LineString and MultiLineString
            if geom.geom_type == 'LineString':
                linestrings = [geom]
                linestrings_proj = [geom_proj]
            elif geom.geom_type == 'MultiLineString':
                linestrings = list(geom.geoms)
                linestrings_proj = list(geom_proj.geoms)
            else:
                continue
            
            for line, line_proj in zip(linestrings, linestrings_proj):
                # Get accurate length in meters from projected geometry
                length_m = line_proj.length
                total_length_m += length_m
                
                # Calculate number of sample points
                num_points = max(2, int(length_m / self.sample_distance_m))
                
                # Sample points along the ORIGINAL lat/lon line
                for i in range(num_points):
                    fraction = i / (num_points - 1) if num_points > 1 else 0
                    point = line.interpolate(fraction, normalized=True)
                    
                    # Check if within boundary
                    if self.boundary.contains(point):
                        points.append((point.y, point.x))  # (lat, lon)
        
        print(f"✓ Generated {len(points):,} sample points")
        print(f"  Total road length: {total_length_m/1000:.1f} km")
        print(f"  Average spacing: {total_length_m/len(points):.1f}m" if points else "  No points generated")
        
        return points
    
    async def check_streetview(self, session: aiohttp.ClientSession, lat: float, lon: float) -> dict:
        """
        Check if Street View exists at a point and return minimal data.
        
        Args:
            session: Aiohttp session
            lat: Latitude
            lon: Longitude
            
        Returns:
            Dict with panoid, lat, lon if found, else None
        """
        try:
            # Search for panorama (official only, 50m radius)
            pano = await streetview.find_panorama_async(
                lat, lon,
                session=session,
                radius=50,
                search_third_party=False
            )
            
            if pano:
                # Deduplicate
                async with self.pano_lock:
                    if pano.id not in self.discovered_panos:
                        self.discovered_panos.add(pano.id)
                        return {
                            'panoid': pano.id,
                            'lat': pano.lat,
                            'lon': pano.lon
                        }
        except Exception:
            pass  # Silent fail
        
        return None
    
    async def worker(self, session: aiohttp.ClientSession, points: List[Tuple[float, float]], 
                     worker_id: int, progress_bar):
        """
        Worker to check Street View availability for a batch of points.
        
        Args:
            session: Aiohttp session
            points: List of (lat, lon) points
            worker_id: Worker ID
            progress_bar: Progress bar
        """
        for lat, lon in points:
            result = await self.check_streetview(session, lat, lon)
            if result:
                self.results.append(result)
            progress_bar.update(1)
    
    async def run(self, num_workers: int = 100):
        """
        Run the complete fetching process.
        
        Args:
            num_workers: Number of concurrent workers
        """
        start_time = datetime.now()
        
        # Step 1: Extract road network
        roads = self.extract_road_network()
        
        # Step 2: Sample points along roads
        sample_points = self.sample_road_points(roads)
        
        if not sample_points:
            print("No sample points generated!")
            return
        
        # Step 3: Check Street View availability in parallel
        print(f"\n=== Checking Street View availability ({num_workers} workers) ===")
        
        # Split points among workers
        chunk_size = len(sample_points) // num_workers + 1
        point_chunks = [
            sample_points[i:i+chunk_size] 
            for i in range(0, len(sample_points), chunk_size)
        ]
        
        async with aiohttp.ClientSession() as session:
            progress_bar = atqdm(total=len(sample_points), desc="Checking points")
            
            tasks = [
                self.worker(session, chunk, i, progress_bar)
                for i, chunk in enumerate(point_chunks)
            ]
            
            await asyncio.gather(*tasks)
            progress_bar.close()
        
        # Step 4: Save results
        self.save_results()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"\n✓ Completed in {duration:.1f} seconds")
        print(f"  Rate: {len(sample_points)/duration:.1f} points/sec")
        print(f"  Coverage: {len(self.results)/len(sample_points)*100:.1f}% of sampled points have Street View")
    
    def save_results(self):
        """Save results to CSV."""
        if not self.results:
            print("No panoramas found!")
            return
        
        # Create output directory
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        
        # Write CSV (just panoid, lat, lon)
        with open(self.output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['panoid', 'lat', 'lon'])
            writer.writeheader()
            writer.writerows(self.results)
        
        print(f"\n✓ Saved {len(self.results):,} unique panoramas to {self.output_csv}")


async def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Fetch Street View panoramas by sampling OSM road network'
    )
    parser.add_argument(
        '--geojson',
        type=str,
        default='streetviews/delhi.geojson',
        help='Path to GeoJSON boundary file (default: delhi-ac.geojson)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='streetviews/delhi_osm_streetviews.csv',
        help='Output CSV path (default: streetviews/delhi_osm_streetviews.csv)'
    )
    parser.add_argument(
        '--sample-distance',
        type=float,
        default=1.0,
        help='Distance between sample points in meters (default: 1.0)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=100,
        help='Number of concurrent workers (default: 100)'
    )
    
    args = parser.parse_args()
    
    fetcher = OSMStreetViewFetcher(
        geojson_path=args.geojson,
        output_csv=args.output,
        sample_distance_m=args.sample_distance
    )
    
    await fetcher.run(num_workers=args.workers)


if __name__ == '__main__':
    asyncio.run(main())

