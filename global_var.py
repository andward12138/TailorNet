import os

# Root directory of the TailorNet project (where this file is located)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset root directory (correct absolute path)
DATA_DIR = '/home/kangdong/TailorNet/tailornet_data'

# Set the paths to SMPL 1.1.0 models (correct absolute paths, match case with directory)
SMPL_PATH_NEUTRAL = '/home/kangdong/TailorNet/tailornet_data/smpl/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl'
SMPL_PATH_MALE = '/home/kangdong/TailorNet/tailornet_data/smpl/basicmodel_m_lbs_10_207_0_v1.1.0.pkl'
SMPL_PATH_FEMALE = '/home/kangdong/TailorNet/tailornet_data/smpl/basicmodel_f_lbs_10_207_0_v1.1.0.pkl'

# Log directory (used for training logs, but not needed for inference; keep as is)
LOG_DIR = '/BS/cpatel/work/data/learn_anim'

# Path to downloaded TailorNet pre-trained models (absolute path, assuming same structure)
MODEL_WEIGHTS_PATH = "/home/kangdong/TailorNet/tailornet_weights"

# Output directory for inference results (absolute path)
OUTPUT_PATH = "/home/kangdong/TailorNet/tailornet_output"

# --------------------------------------------------------------------
# Variables below hardly need to change
# --------------------------------------------------------------------

# Available genders
GENDERS = ['neutral', 'male', 'female']

# File in DATA_DIR containing pose indices for train/test splits
POSE_SPLIT_FILE = 'split_static_pose_shape.npz'

# File in DATA_DIR containing garment template information
GAR_INFO_FILE = 'garment_class_info.pkl'

# Root dir for smooth data (keep as DATA_DIR since you are using pre-trained models)
SMOOTH_DATA_DIR = DATA_DIR

# Indicates that smooth groundtruth data is available (keep True for inference)
SMOOTH_STORED = True

# Using smoothing in posed space for skirt (keep True as per your requirement)
POSED_SMOOTH_SKIRT = True

# Lists the indices of joints which affect the deformations of particular garment
VALID_THETA = {
    't-shirt': [0, 1, 2, 3, 6, 9, 12, 13, 14, 16, 17, 18, 19],
    'old-t-shirt': [0, 1, 2, 3, 6, 9, 12, 13, 14, 16, 17, 18, 19],
    'shirt': [0, 1, 2, 3, 6, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21],
    'pant': [0, 1, 2, 4, 5, 7, 8],
    'short-pant': [0, 1, 2, 4, 5],
    'skirt': [0, 1, 2, 4, 5],
}