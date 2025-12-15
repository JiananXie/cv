import numpy as np
import os

def quat_to_rot_matrix(q):
    """
    Convert quaternion [w, x, y, z] to 3x3 rotation matrix.
    """
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])

def get_focal_length_from_nvm(nvm_path):
    """
    Reads the first focal length found in the NVM file.
    Assumes NVM format:
    <Header>
    <Number of Cameras>
    <Filename> <Focal Length> ...
    """
    if not os.path.exists(nvm_path):
        print(f"Warning: NVM file not found at {nvm_path}")
        return None

    with open(nvm_path, 'r') as f:
        # Skip header (usually 'NVM_V3')
        header = f.readline()
        # Skip number of cameras (or empty lines)
        line = f.readline()
        while not line.strip().isdigit():
            line = f.readline()
            if not line: return None
        
        # Read first camera line
        cam_line = f.readline()
        parts = cam_line.split()
        if len(parts) > 1:
            return float(parts[1])
    return None

def get_intrinsics(focal_length, original_size, new_size):
    """
    Constructs intrinsic matrix K for the resized/cropped image.
    original_size: (W, H)
    new_size: (W_new, H_new) - usually (224, 224)
    """
    # Assuming the image was resized to (256, 256) then cropped to (224, 224)
    # But PoseDataset resizes to loadSize (256) then crops to fineSize (224).
    # The focal length scales with the resize.
    
    # Note: PoseDataset resizes (W, H) -> (256, 256) ignoring aspect ratio.
    # So fx scales by 256/W, fy scales by 256/H.
    # Then crop doesn't change focal length, just principal point.
    
    orig_w, orig_h = original_size
    load_size = 256 # Hardcoded based on typical options, but should be passed if possible
    fine_size = new_size[0] # Assuming square
    
    fx = focal_length * (load_size / orig_w)
    fy = focal_length * (load_size / orig_h)
    
    # Principal point is usually center of the image
    cx = fine_size / 2.0
    cy = fine_size / 2.0
    
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    return K

def compute_reprojection_error(gt_pose, pred_pose, K):
    """
    gt_pose: [x, y, z, qw, qx, qy, qz]
    pred_pose: [x, y, z, qw, qx, qy, qz]
    K: Intrinsic matrix (3x3)
    """
    # 1. Parse poses
    t_gt = gt_pose[:3]
    q_gt = gt_pose[3:]
    R_gt = quat_to_rot_matrix(q_gt)
    
    t_pred = pred_pose[:3]
    q_pred = pred_pose[3:]
    R_pred = quat_to_rot_matrix(q_pred)
    
    # 2. Generate Virtual 3D Points in GT Camera Coordinate System
    # Create a grid of points 5 meters in front of the camera
    # Shape: (3, N)
    # Points at z=5, x=[-1, 0, 1], y=[-1, 0, 1]
    local_points = []
    for x in [-1, 0, 1]:
        for y in [-1, 0, 1]:
            local_points.append([x, y, 5.0])
    local_points = np.array(local_points).T 
    
    # 3. Transform to World Coordinate System using GT Pose
    # PoseNet output is usually Camera-to-World (position and orientation of camera)
    # X_world = R * X_cam + t
    world_points = np.dot(R_gt, local_points) + t_gt.reshape(3, 1)
    
    # 4. Project back to image using Predicted Pose
    # World-to-Camera: X_cam = R.T * (X_world - t)
    cam_points_pred = np.dot(R_pred.T, (world_points - t_pred.reshape(3, 1)))
    
    # 5. Project to Pixel Coordinates
    # pixels = K * X_cam
    # Handle points behind camera (z <= 0)
    valid_indices = cam_points_pred[2, :] > 0.1
    if np.sum(valid_indices) == 0:
        return 1000.0 # Large error if all points are behind camera
        
    cam_points_pred = cam_points_pred[:, valid_indices]
    pixel_coords_homo = np.dot(K, cam_points_pred)
    pixel_coords_pred = pixel_coords_homo[:2, :] / pixel_coords_homo[2, :]
    
    # 6. Project GT points (Ground Truth projection)
    # Since local_points are defined in GT camera frame, we just project them directly
    # But we need to filter the same indices
    local_points_valid = local_points[:, valid_indices]
    
    # Note: local_points are already in GT camera frame.
    # But wait, if we used non-uniform scaling (resize 1920x1080 -> 256x256),
    # the "GT Camera Frame" implies the physical camera.
    # But K is adapted to the distorted image.
    # So projecting local_points using K gives the pixel coordinates in the distorted image.
    gt_cam_points_homo = np.dot(K, local_points_valid)
    pixel_coords_gt = gt_cam_points_homo[:2, :] / gt_cam_points_homo[2, :]
    
    # 7. Calculate Error
    errors = np.linalg.norm(pixel_coords_pred - pixel_coords_gt, axis=0)
    return np.mean(errors)
