import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse
from data.data_loader import CreateDataLoader
from models.models import create_model

# Use Agg backend for headless servers
plt.switch_backend('agg')

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataroot', required=True)
    parser.add_argument('--name', type=str, default='experiment_name')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')
    parser.add_argument('--which_epoch', type=str, default='490')
    parser.add_argument('--model', type=str, default='posenet')
    parser.add_argument('--backbone', type=str, default='inception')
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--batchSize', type=int, default=1)
    parser.add_argument('--nThreads', default=4, type=int)
    parser.add_argument('--serial_batches', action='store_true', default=True)
    parser.add_argument('--no_flip', action='store_true', default=True)
    parser.add_argument('--beta', type=float, default=500)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--phase', type=str, default='test')
    parser.add_argument('--continue_train', action='store_true', default=False)

    # Dummy args
    parser.add_argument('--input_nc', type=int, default=3)
    parser.add_argument('--output_nc', type=int, default=7)
    parser.add_argument('--loadSize', type=int, default=256)
    parser.add_argument('--fineSize', type=int, default=224)
    parser.add_argument('--no_dropout', action='store_true', default=True)
    parser.add_argument('--max_dataset_size', type=int, default=float("inf"))
    parser.add_argument('--img_ret', action='store_true')
    parser.add_argument('--lstm_hidden_size', type=int, default=None)
    parser.add_argument('--transformer_hidden_size', type=int, default=None)
    
    # Visualization args
    parser.add_argument('--figsize_w', type=float, default=10.0, help='Figure width in inches')
    parser.add_argument('--figsize_h', type=float, default=10.0, help='Figure height in inches')
    parser.add_argument('--dpi', type=int, default=300, help='Output DPI')
    parser.add_argument('--padding', type=float, default=0.01, help='Padding factor around trajectory (default 0.01)')
    parser.add_argument('--auto_height', action='store_true', help='Automatically adjust figure height to match data aspect ratio')
    parser.add_argument('--line_width', type=float, default=0.5, help='Line width for trajectory')
    parser.add_argument('--marker_size', type=float, default=0.6, help='Size of the frustum markers')
    parser.add_argument('--swap_axes', action='store_true', help='Swap X and Z axes (rotate 90 deg and reflect)')

    opt = parser.parse_args()
    opt.isTrain = False

    str_ids = opt.gpu_ids.split(',')
    opt.gpu_ids = [int(s) for s in str_ids if int(s) >= 0]
    if len(opt.gpu_ids) > 0:
        torch.cuda.set_device(opt.gpu_ids[0])
    return opt

def get_yaw_from_q(q):
    w, x, y, z = q / (np.linalg.norm(q) + 1e-8)
    dx = 2 * (x * z + w * y)
    dz = 1 - 2 * (x**2 + y**2)
    return np.arctan2(dx, dz)

def draw_frustum(ax, x, z, yaw, color, size=0.6, alpha=0.5):
    pts = np.array([[0, 1.2], [-0.6, -0.6], [0.6, -0.6]]) * size
    rot = np.array([
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw),  np.cos(yaw)]
    ])
    pts = pts @ rot.T + [x, z]
    poly = patches.Polygon(
        pts,
        closed=True,
        color=color,
        alpha=alpha,
        linewidth=0,
        zorder=10
    )
    ax.add_patch(poly)

if __name__ == '__main__':
    opt = parse_args()

    dataset = CreateDataLoader(opt)
    model = create_model(opt)
    model.load_network(model.netG, 'G', opt.which_epoch)
    model.netG.eval()

    test_gt, test_pred = [], []

    for data in dataset:
        model.set_input(data)
        model.test()
        p = model.get_current_pose()
        g = model.input_B.cpu().numpy()

        if p.ndim == 1:
            p = np.expand_dims(p, 0)
        if g.ndim == 1:
            g = np.expand_dims(g, 0)

        test_pred.append(p[0])
        test_gt.append(g[0])

    test_gt = np.array(test_gt)
    test_pred = np.array(test_pred)

    if opt.swap_axes:
        # Swap X and Z columns (0 and 1)
        test_gt[:, [0, 1]] = test_gt[:, [1, 0]]
        test_pred[:, [0, 1]] = test_pred[:, [1, 0]]

    all_x = np.concatenate([test_gt[:, 0], test_pred[:, 0]])
    all_z = np.concatenate([test_gt[:, 1], test_pred[:, 1]])

    min_x, max_x = np.min(all_x), np.max(all_x)
    min_z, max_z = np.min(all_z), np.max(all_z)

    # Calculate data range and center
    data_w = max_x - min_x
    data_h = max_z - min_z
    data_cx = (min_x + max_x) / 2.0
    data_cz = (min_z + max_z) / 2.0

    if opt.auto_height:
        # Adjust figsize_h to match data aspect ratio
        if data_w == 0: data_w = 1.0
        ratio = data_h / data_w
        
        # Estimate margins for title, labels, etc. (in inches)
        # We assume figsize_w is mostly data width, but figsize_h needs extra for title/labels
        margin_h = 1.5 
        
        # We want the data area height to be (figsize_w * ratio)
        # So total height = data_area_h + margin_h
        opt.figsize_h = (opt.figsize_w * ratio) + margin_h
        
        if opt.figsize_h < 3.0: opt.figsize_h = 3.0 # Minimum height
        print(f"Auto-adjusted figsize_h to {opt.figsize_h:.2f} inches")

    # Add padding
    pad_factor = opt.padding
    w_span = max_x - min_x
    h_span = max_z - min_z
    
    min_x -= w_span * pad_factor
    max_x += w_span * pad_factor
    min_z -= h_span * pad_factor
    max_z += h_span * pad_factor

    plt.figure(figsize=(opt.figsize_w, opt.figsize_h))
    ax = plt.gca()
    ax.set_facecolor('#f4f1e6')

    for i in range(len(test_gt)):
        ax.plot(
            [test_gt[i, 0], test_pred[i, 0]],
            [test_gt[i, 1], test_pred[i, 1]],
            color='black',
            linewidth=opt.line_width * 0.5, # Connection lines slightly thinner
            alpha=0.25,
            zorder=5
        )

        draw_frustum(
            ax,
            test_gt[i, 0],
            test_gt[i, 1],
            -get_yaw_from_q(test_gt[i, 3:]) - np.pi/2 if opt.swap_axes else get_yaw_from_q(test_gt[i, 3:]),
            color='blue',
            size=opt.marker_size,
            alpha=0.35
        )

        draw_frustum(
            ax,
            test_pred[i, 0],
            test_pred[i, 1],
            -get_yaw_from_q(test_pred[i, 3:]) - np.pi/2 if opt.swap_axes else get_yaw_from_q(test_pred[i, 3:]),
            color='red',
            size=opt.marker_size,
            alpha=0.55
        )

    plt.xlim(min_x, max_x)
    plt.ylim(min_z, max_z)

    ax.set_aspect('equal', adjustable='box')
    plt.grid(True, linestyle='--', color='white', alpha=0.4)

    plt.title(f"Trajectory Comparison (GT vs Prediction)", fontsize=14)
    if opt.swap_axes:
        plt.xlabel('Z (m)')
        plt.ylabel('X (m)')
    else:
        plt.xlabel('X (m)')
        plt.ylabel('Z (m)')

    legend_elements = [
        patches.Patch(color='blue', alpha=0.4, label='Ground Truth'),
        patches.Patch(color='red', alpha=0.6, label='Prediction')
    ]
    plt.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()

    output_filename = f"trajectory_gt_vs_pred_{opt.which_epoch}.png"
    # Remove bbox_inches='tight' to keep fixed image dimensions
    plt.savefig(output_filename, dpi=opt.dpi)
    print(f"Image saved to {output_filename}")
