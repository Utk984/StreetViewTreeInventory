import os
import torch
from dotenv import load_dotenv
import logging
from datetime import datetime

class Config:
    def __init__(self):
        load_dotenv()

        # Base directories
        self.ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

        # Data directories
        self.DATA_DIR = os.path.join(self.ROOT_DIR, "data")
        self.VIEW_DIR = os.path.join(self.DATA_DIR, "views")
        self.FULL_DIR = os.path.join(self.DATA_DIR, "full")
        self.LOG_DIR = os.path.join(self.DATA_DIR, "logs")
        self.DEPTH_DIR = os.path.join(self.DATA_DIR, "depth_maps")
        self.MASK_DIR = os.path.join(self.DATA_DIR, "masks")
        self.OUTPUT_DIR = os.path.join(self.ROOT_DIR, "outputs")
        self.STREETVIEW_DIR = os.path.join(self.ROOT_DIR, "streetviews")

        # Ensure output folders exist
        os.makedirs(self.VIEW_DIR, exist_ok=True)
        os.makedirs(self.FULL_DIR, exist_ok=True)
        os.makedirs(self.LOG_DIR, exist_ok=True)
        os.makedirs(self.DEPTH_DIR, exist_ok=True)
        os.makedirs(self.MASK_DIR, exist_ok=True)
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.STREETVIEW_DIR, exist_ok=True)
        
        # Configure logging - every level INFO and DEBUG are logged
        logging.basicConfig(
            level=logging.INFO,  # DEBUG level captures DEBUG, INFO, WARNING, ERROR, CRITICAL
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.LOG_DIR, f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_pipeline.log")),
            ]
        )
        self.LOG_FILE = os.path.join(self.LOG_DIR, "pipeline.log")

        # Model config
        self.TREE_MODEL_PATH = os.path.join(
            self.ROOT_DIR, "models", "TreeModel", "weights", "best.pt"
        )

        self.DEPTH_MODEL_PATH = os.path.join(
            self.ROOT_DIR, "models", "DepthAnything", "checkpoints",
            "depth_anything_v2_metric_vkitti_vitl.pth"
        )

        self.DEPTH_CALIBRATION_MODEL_PATH = os.path.join(
            self.ROOT_DIR, "models", "CalibrateDepth", "weights", "random_forest.pkl"
        )

        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.DEPTH_MODEL_CONFIGS = {
            "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
            "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
            "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        }

        ### EDITABLE SETTINGS ###
        
        # Panorama CSV input
        self.PANORAMA_CSV = os.path.join(self.STREETVIEW_DIR, "chandigarh_streets.csv")

        # Output CSV
        self.OUTPUT_CSV = os.path.join(self.OUTPUT_DIR, "chandigarh_trees.csv")

        # Max concurrent
        self.MAX_CONCURRENT = 3

        # Image settings
        self.FOV = 90
        self.WIDTH = 1024
        self.HEIGHT = 720
        self.BATCH_SIZE = 10

        # Save data
        self.SAVE_DEPTH_MAPS = False
        self.SAVE_MASK_JSON = True
        self.SAVE_VIEWS = True
        self.SAVE_FULL = True
        
        # MRF Triangulation settings
        self.ENABLE_TRIANGULATION = True
        self.TRIANGULATION_MAX_OBJECT_DIST = 20.0      # Max distance from camera to trees (meters)
        self.TRIANGULATION_MAX_CLUSTER_DIST = 3.0       # Max cluster size for grouping trees (meters)
        self.TRIANGULATION_ICM_ITERATIONS = 20          # Number of ICM optimization iterations
        self.TRIANGULATION_DEPTH_WEIGHT = 0.3           # Weight for depth consistency (alpha)
        self.TRIANGULATION_MULTIVIEW_WEIGHT = 0.2       # Weight for multi-view preference (beta)
        self.TRIANGULATED_OUTPUT_CSV = os.path.join(self.OUTPUT_DIR, "chandigarh_trees_triangulated.csv")
        
        ### END EDITABLE SETTINGS ###