import torch
import torch.nn as nn

class ReprojectionLoss(nn.Module):
    def __init__(self, focal_length, image_size, device='cuda'):
        super(ReprojectionLoss, self).__init__()
        self.device = device
        
        # Construct Intrinsic Matrix K
        # Assuming principal point is at the center
        w, h = image_size
        cx, cy = w / 2.0, h / 2.0
        
        self.K = torch.tensor([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ], dtype=torch.float32, device=device)

        # Generate Virtual 3D Points (Grid 5m in front of camera)
        # Shape: (3, N)
        points = []
        for x in [-1, 0, 1]:
            for y in [-1, 0, 1]:
                points.append([x, y, 5.0])
        self.local_points = torch.tensor(points, dtype=torch.float32, device=device).t() # (3, 9)

    def forward(self, pred_t, pred_q, target_t, target_q):
        """
        pred_t: (B, 3)
        pred_q: (B, 4) [w, x, y, z]
        target_t: (B, 3)
        target_q: (B, 4) [w, x, y, z]
        """
        batch_size = pred_t.size(0)
        
        # 1. Project GT points (Target Pixels)
        # Since local_points are defined in GT camera frame, 
        # their projection is simply K * local_points
        proj_gt = torch.matmul(self.K, self.local_points) # (3, 9)
        pixels_gt = proj_gt[:2] / (proj_gt[2:3] + 1e-7) # (2, 9)
        pixels_gt = pixels_gt.unsqueeze(0).expand(batch_size, -1, -1) # (B, 2, 9)

        # 2. Project points using Predicted Pose
        # We need to transform local_points (in GT Camera Frame) -> World -> Predicted Camera -> Pixels
        
        # A. GT Camera -> World
        # P_world = R_gt * P_local + t_gt
        R_gt = self.quat_to_mat(target_q) # (B, 3, 3)
        # (B, 3, 3) * (B, 3, 9) -> (B, 3, 9)
        points_world = torch.bmm(R_gt, self.local_points.unsqueeze(0).expand(batch_size, -1, -1)) + target_t.unsqueeze(2)

        # B. World -> Predicted Camera
        # P_pred = R_pred.T * (P_world - t_pred)
        R_pred = self.quat_to_mat(pred_q) # (B, 3, 3)
        # R_pred.T is transpose of rotation matrix (inverse rotation)
        R_pred_inv = R_pred.transpose(1, 2)
        
        points_pred_cam = torch.bmm(R_pred_inv, points_world - pred_t.unsqueeze(2))

        # C. Predicted Camera -> Pixels
        # pixels = K * P_pred
        # We need to handle points that might fall behind the camera in the predicted pose
        # But for loss calculation, we usually assume they are visible or use robust loss
        proj_pred = torch.matmul(self.K, points_pred_cam) # (B, 3, 9)
        
        # Avoid division by zero or negative depth issues
        depth = proj_pred[:, 2:3]
        depth = torch.clamp(depth, min=0.1) 
        
        pixels_pred = proj_pred[:, :2] / depth # (B, 2, 9)

        # 3. Calculate Loss
        # Use L1 Loss (Huber/SmoothL1 is also good) for robustness against outliers
        loss = torch.nn.functional.l1_loss(pixels_pred, pixels_gt)
        
        return loss

    def quat_to_mat(self, q):
        # q: (B, 4) [w, x, y, z]
        # Normalize quaternion just in case
        q = torch.nn.functional.normalize(q, p=2, dim=1)
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        
        B = q.size(0)
        R = torch.zeros(B, 3, 3, device=self.device)
        
        R[:, 0, 0] = 1 - 2*y**2 - 2*z**2
        R[:, 0, 1] = 2*x*y - 2*z*w
        R[:, 0, 2] = 2*x*z + 2*y*w
        
        R[:, 1, 0] = 2*x*y + 2*z*w
        R[:, 1, 1] = 1 - 2*x**2 - 2*z**2
        R[:, 1, 2] = 2*y*z - 2*x*w
        
        R[:, 2, 0] = 2*x*z - 2*y*w
        R[:, 2, 1] = 2*y*z + 2*x*w
        R[:, 2, 2] = 1 - 2*x**2 - 2*y**2
        return R
