"""
This script converts Record3D data into the BundleSDF format.
NOTE: You first need to replace the record3d `.r3d` extension with `.zip` and then unzip
the file to get the expected record3d data format.

Record3D Data Format:
---------------------
The input Record3D dataset is structured as follows:
    data_dir/
      - metadata
      - rgbd/
        - 0.jpg
        - 0.depth
        - 1.jpg
        - 1.depth
        - ...

- The `metadata` file is a JSON file containing the camera intrinsics:
  {"w": width, "h": height, "K": [...]} where K is a flattened 3x3 intrinsic matrix.
- The RGB images are stored as .jpg files.
- The depth maps are stored in a compressed .depth format, containing floating-point depth values in meters.

BundleSDF Data Format:
----------------------
The converted output follows the BundleSDF format:
    root/
      ├──rgb/    (PNG files)
      ├──depth/  (PNG files, stored in mm, uint16 format, filenames match rgb images)
      ├──cam_K.txt   (3x3 intrinsic matrix, space and newline delimited)

- RGB images are saved as PNGs in the `rgb/` directory.
- Depth maps are converted to 16-bit PNGs in millimeters and stored in `depth/`.
- Camera intrinsics are saved as a `cam_K.txt` file.
- Optionally, colorized depth maps can be saved if a path is specified.

The script performs the following steps:
1. Loads metadata and extracts camera intrinsics.
2. Processes each frame in parallel:
   - Reads and saves the RGB image.
   - Loads and resizes the depth image, converting it from meters to millimeters.
   - Saves the depth image as a 16-bit PNG.
   - Optionally generates and saves a colorized depth image.
3. Saves the intrinsic matrix to `cam_K.txt`.

Usage:
------
Run the script with:
    python convert_record3d_to_bundlesdf.py <record3d_dir> <bundlesdf_dir> [--color_depth_dir <path>]

"""

import numpy as np
import cv2
import json
import pathlib
import liblzfse
import argparse
from PIL import Image
from tqdm.contrib.concurrent import process_map

def load_depth(filepath):
    """Load depth from compressed .depth file."""
    with open(filepath, 'rb') as depth_fh:
        raw_bytes = depth_fh.read()
        decompressed_bytes = liblzfse.decompress(raw_bytes)
        depth_img = np.frombuffer(decompressed_bytes, dtype=np.float32)
        depth_img = depth_img.reshape((256, 192))  # Original resolution
    return depth_img

def colorize_depth(depth_img):
    """Convert depth image to a color map for visualization."""
    depth_normalized = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX)
    depth_colored = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_JET)
    return depth_colored

def process_frame(args):
    depth_file, rgbd_path, bundlesdf_path, color_depth_path = args
    basename = depth_file.stem  # Get filename without extension
    rgb_file = rgbd_path / f"{basename}.jpg"
    
    # Load and save RGB image
    rgb_img = cv2.imread(str(rgb_file))
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    img_h, img_w = rgb_img.shape[:2]  # Get image dimensions
    Image.fromarray(rgb_img).save(bundlesdf_path / "rgb" / f"{basename}.png")
    
    # Load and convert depth image
    depth_img = load_depth(depth_file)
    depth_resized = cv2.resize(depth_img, (img_w, img_h))  # Resize based on image size
    
    # Sanitize depth data by replacing NaN and infinite values with max depth
    max_depth = np.nanmax(depth_resized[np.isfinite(depth_resized)]) if np.isfinite(depth_resized).any() else 0.0
    depth_resized = np.nan_to_num(depth_resized, nan=max_depth, posinf=max_depth, neginf=0.0)
    depth_mm = (depth_resized * 1000).astype(np.uint16)  # Convert meters to mm
    
    Image.fromarray(depth_mm).save(bundlesdf_path / "depth" / f"{basename}.png")
    
    # Save colorized depth image if specified
    if color_depth_path:
        depth_colored = colorize_depth(depth_resized)
        Image.fromarray(depth_colored).save(color_depth_path / f"{basename}.png")

def convert_record3d_to_bundlesdf(record3d_dir, bundlesdf_dir, color_depth_dir=None):
    record3d_path = pathlib.Path(record3d_dir)
    bundlesdf_path = pathlib.Path(bundlesdf_dir)
    bundlesdf_path.mkdir(parents=True, exist_ok=True)
    (bundlesdf_path / "rgb").mkdir(exist_ok=True)
    (bundlesdf_path / "depth").mkdir(exist_ok=True)
    
    color_depth_path = None
    if color_depth_dir:
        color_depth_path = pathlib.Path(color_depth_dir)
        color_depth_path.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    with open(record3d_path / "metadata", "r") as f:
        metadata = json.load(f)

    K = np.array(metadata['K']).reshape(3, 3).T  # Transpose to get correct intrinsics

    # Save camera intrinsics
    np.savetxt(bundlesdf_path / "cam_K.txt", K, fmt="%.6f")

    # Process images and depth
    rgbd_path = record3d_path / "rgbd"
    depth_files = sorted(rgbd_path.glob("*.depth"))
    
    # Use tqdm's process_map for parallel processing
    process_map(process_frame, [(df, rgbd_path, bundlesdf_path, color_depth_path) for df in depth_files], chunksize=1)
    
    print("Conversion complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Record3D data to BundleSDF format")
    parser.add_argument("record3d_dir", type=str, help="Path to Record3D data directory")
    parser.add_argument("bundlesdf_dir", type=str, help="Path to output BundleSDF directory")
    parser.add_argument("--color_depth_dir", type=str, default=None, help="Optional path to save colorized depth images")
    args = parser.parse_args()
    
    convert_record3d_to_bundlesdf(args.record3d_dir, args.bundlesdf_dir, args.color_depth_dir)
