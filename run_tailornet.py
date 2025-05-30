import os
import numpy as np
import torch
from scipy.spatial import KDTree

from psbody.mesh import Mesh

from models.tailornet_model import get_best_runner as get_tn_runner
from models.smpl4garment import SMPL4Garment
from utils.rotation import normalize_y_rotation
from visualization.blender_renderer import visualize_garment_body

from dataset.canonical_pose_dataset import get_style, get_shape
from visualization.vis_utils import get_specific_pose, get_specific_style_old_tshirt
from visualization.vis_utils import get_specific_shape, get_amass_sequence_thetas
from utils.interpenetration import remove_interpenetration_fast

# Import global_var to access DATA_DIR and OUTPUT_PATH
import global_var

# Set output path where inference results will be stored (use path from global_var)
OUT_PATH = global_var.OUTPUT_PATH  # Will be "/home/kangdong/TailorNet/tailornet_output"

def get_single_frame_inputs(garment_class, gender):
    """Prepare some individual frame inputs."""
    betas = [
        get_specific_shape('tallthin'),
        get_specific_shape('shortfat'),
        get_specific_shape('mean'),
        get_specific_shape('somethin'),
        get_specific_shape('somefat'),
    ]
    # old t-shirt style parameters are centered around [1.5, 0.5, 1.5, 0.0]
    # whereas all other garments styles are centered around [0, 0, 0, 0]
    if garment_class == 'old-t-shirt':
        gammas = [
            get_specific_style_old_tshirt('mean'),
            get_specific_style_old_tshirt('big'),
            get_specific_style_old_tshirt('small'),
            get_specific_style_old_tshirt('shortsleeve'),
            get_specific_style_old_tshirt('big_shortsleeve'),
        ]
    else:
        # Check available style files
        style_ids = ['000', '001', '002', '003', '004']
        available_styles = []
        for style_id in style_ids:
            style_path = os.path.join(global_var.DATA_DIR, 
                f"{garment_class}_{gender}/style/gamma_{style_id}.npy")
            if os.path.exists(style_path):
                available_styles.append(style_id)
            else:
                print(f"Style file not found: {style_path}")

        # If no style files are found, generate random gamma (assume 4 dimensions)
        if not available_styles:
            print(f"No style files found for {garment_class}_{gender}, generating random gamma...")
            gammas = [np.random.randn(4) for _ in range(5)]
        else:
            # Use available styles, repeat the first available style if fewer than 5
            while len(available_styles) < 5:
                available_styles.append(available_styles[0])
            gammas = [
                get_style(available_styles[0], garment_class=garment_class, gender=gender),
                get_style(available_styles[1], garment_class=garment_class, gender=gender),
                get_style(available_styles[2], garment_class=garment_class, gender=gender),
                get_style(available_styles[3], garment_class=garment_class, gender=gender),
                get_style(available_styles[4], garment_class=garment_class, gender=gender),
            ]
    thetas = [
        get_specific_pose(0),
        get_specific_pose(1),
        get_specific_pose(2),
        get_specific_pose(3),
        get_specific_pose(4),
    ]
    return thetas, betas, gammas


def get_sequence_inputs(garment_class, gender):
    """Prepare sequence inputs."""
    beta = get_specific_shape('somethin')
    if garment_class == 'old-t-shirt':
        gamma = get_specific_style_old_tshirt('big_longsleeve')
    else:
        # Check if style '000' exists, otherwise try other styles or use random gamma
        style_ids = ['000', '001', '002', '003', '004']
        style_id = None
        for sid in style_ids:
            style_path = os.path.join(global_var.DATA_DIR, 
                f"{garment_class}_{gender}/style/gamma_{sid}.npy")
            if os.path.exists(style_path):
                style_id = sid
                break
        if style_id is None:
            print(f"No style files found for {garment_class}_{gender}, generating random gamma...")
            gamma = np.random.randn(4)  # Assume 4 dimensions
        else:
            gamma = get_style(style_id, gender=gender, garment_class=garment_class)

    # downsample sequence frames by 2
    thetas = get_amass_sequence_thetas('05_02')[::2]

    betas = np.tile(beta[None, :], [thetas.shape[0], 1])
    gammas = np.tile(gamma[None, :], [thetas.shape[0], 1])
    return thetas, betas, gammas


def apply_gaussian_splash(clothing_verts, body_verts, body_faces, sigma=0.01, decay_factor=10.0):
    """
    将高斯溅射应用于服装顶点，使其更好地贴合人体
    
    参数:
    - clothing_verts: 服装顶点坐标 [N, 3]
    - body_verts: 人体顶点坐标 [M, 3]
    - body_faces: 人体面片索引 [K, 3]
    - sigma: 高斯扰动的基础标准差
    - decay_factor: 距离衰减因子，控制远离人体的点扰动减小的速率
    
    返回:
    - adjusted_clothing_verts: 调整后的服装顶点
    """
    # 构建KD树用于快速查找最近点
    body_tree = KDTree(body_verts)
    
    # 计算每个服装顶点到人体的最短距离
    distances, indices = body_tree.query(clothing_verts)
    
    # 计算人体表面法线方向
    body_normals = compute_vertex_normals(body_verts, body_faces)
    
    # 根据距离计算自适应的sigma值（距离越远，扰动越小）
    adaptive_sigma = sigma * np.exp(-distances / decay_factor)
    
    # 生成高斯噪声
    noise = np.random.normal(0, 1, clothing_verts.shape) * adaptive_sigma[:, np.newaxis]
    
    # 获取对应的人体法线方向
    closest_normals = body_normals[indices]
    
    # 将噪声投影到法线方向上，使服装更好地贴合人体
    dot_products = np.sum(noise * closest_normals, axis=1)
    projected_noise = closest_normals * dot_products[:, np.newaxis]
    
    # 应用噪声
    adjusted_clothing_verts = clothing_verts + projected_noise
    
    return adjusted_clothing_verts

def compute_vertex_normals(verts, faces):
    """计算顶点法线"""
    # 初始化法线
    vertex_normals = np.zeros_like(verts)
    
    # 计算每个面的法线
    face_normals = np.zeros((len(faces), 3))
    for i, face in enumerate(faces):
        v0, v1, v2 = verts[face]
        face_normal = np.cross(v1 - v0, v2 - v0)
        # 归一化
        norm = np.linalg.norm(face_normal)
        if norm > 1e-10:
            face_normal = face_normal / norm
        face_normals[i] = face_normal
    
    # 将面法线分配给顶点
    for i, face in enumerate(faces):
        for vertex_idx in face:
            vertex_normals[vertex_idx] += face_normals[i]
    
    # 归一化顶点法线
    norms = np.linalg.norm(vertex_normals, axis=1)
    mask = norms > 1e-10
    vertex_normals[mask] = vertex_normals[mask] / norms[mask, np.newaxis]
    
    return vertex_normals

# 评估贴合度函数
def evaluate_clothing_fit(clothing_verts, body_verts):
    """
    评估服装与人体的贴合度
    """
    body_tree = KDTree(body_verts)
    distances, _ = body_tree.query(clothing_verts)
    
    # 计算统计信息
    mean_dist = np.mean(distances)
    max_dist = np.max(distances)
    close_ratio = np.sum(distances < 0.01) / len(distances)  # 0.01单位内的点的比例
    
    return {
        "mean_distance": mean_dist,
        "max_distance": max_dist,
        "close_points_ratio": close_ratio
    }

def run_tailornet(garment_class='skirt', gender='female', use_gaussian_splash=True):
    # Use arguments for garment_class and gender
    thetas, betas, gammas = get_single_frame_inputs(garment_class, gender)
    # # uncomment the line below to run inference on sequence data
    # thetas, betas, gammas = get_sequence_inputs(garment_class, gender)

    # load model
    tn_runner = get_tn_runner(gender=gender, garment_class=garment_class)
    smpl = SMPL4Garment(gender=gender)

    # make out directory if doesn't exist
    if not os.path.isdir(OUT_PATH):
        os.makedirs(OUT_PATH, exist_ok=True)

    # run inference
    for i, (theta, beta, gamma) in enumerate(zip(thetas, betas, gammas)):
        print(f"Inference frame {i}/{len(thetas)}")
        # normalize y-rotation to make it front facing
        theta_normalized = normalize_y_rotation(theta)
        with torch.no_grad():
            pred_verts_d = tn_runner.forward(
                thetas=torch.from_numpy(theta_normalized[None, :].astype(np.float32)).cuda(),
                betas=torch.from_numpy(beta[None, :].astype(np.float32)).cuda(),
                gammas=torch.from_numpy(gamma[None, :].astype(np.float32)).cuda(),
            )[0].cpu().numpy()

        # 获取原始的身体和服装网格
        body, pred_gar = smpl.run(beta=beta, theta=theta, garment_class=garment_class, garment_d=pred_verts_d)
        
        # 保存原始服装网格用于对比（如果使用高斯溅射）
        original_pred_gar = None
        if use_gaussian_splash:
            # 创建一个新的 Mesh 对象而不是使用 copy()
            original_pred_gar = Mesh(v=pred_gar.v.copy(), f=pred_gar.f.copy())
        
        # 应用高斯溅射（如果启用）
        if use_gaussian_splash:
            print("应用高斯溅射...")
            # 提取点云和面片数据
            clothing_verts = pred_gar.v
            body_verts = body.v
            body_faces = body.f
            
            # 应用高斯溅射
            adjusted_clothing_verts = apply_gaussian_splash(
                clothing_verts, body_verts, body_faces, sigma=0.005, decay_factor=15.0
            )
            
            # 更新服装网格的顶点
            pred_gar.v = adjusted_clothing_verts
            
            # 评估贴合度并打印结果
            fit_metrics = evaluate_clothing_fit(adjusted_clothing_verts, body_verts)
            print(f"贴合度评估: 平均距离={fit_metrics['mean_distance']:.4f}, "
                  f"最大距离={fit_metrics['max_distance']:.4f}, "
                  f"接触点比例={fit_metrics['close_points_ratio']:.2f}")
        
        # 移除穿模
        pred_gar = remove_interpenetration_fast(pred_gar, body)

        # 保存处理后的网格
        output_prefix = f"{garment_class}_{gender}_{i:04d}"
        body.write_ply(os.path.join(OUT_PATH, f"body_{output_prefix}.ply"))
        pred_gar.write_ply(os.path.join(OUT_PATH, f"pred_gar_{output_prefix}.ply"))
        
        # 保存原始网格（用于对比）
        if use_gaussian_splash and original_pred_gar is not None:
            original_pred_gar.write_ply(os.path.join(OUT_PATH, f"pred_gar_original_{output_prefix}.ply"))


def render_images(garment_class='skirt', gender='female'):
    """Render garment and body using blender."""
    i = 0
    while True:
        body_path = os.path.join(OUT_PATH, f"body_{garment_class}_{gender}_{i:04d}.ply")
        if not os.path.exists(body_path):
            break
        body = Mesh(filename=body_path)
        pred_gar = Mesh(filename=os.path.join(OUT_PATH, f"pred_gar_{garment_class}_{gender}_{i:04d}.ply"))

        visualize_garment_body(
            pred_gar, body, os.path.join(OUT_PATH, f"img_{garment_class}_{gender}_{i:04d}.png"),
            garment_class=garment_class, side='front')
        i += 1

    # Concatenate frames of sequence data using this command
    # ffmpeg -r 10 -i img_%04d.png -vcodec libx264 -crf 10  -pix_fmt yuv420p check.mp4
    # Make GIF
    # convert -delay 200 -loop 0 -dispose 2 *.png check.gif
    # convert check.gif -resize 512x512 check_small.gif


if __name__ == '__main__':
    import sys
    use_gaussian_splash = True  # 默认启用高斯溅射
    
    if len(sys.argv) > 1 and sys.argv[1] == '--no-gaussian':
        use_gaussian_splash = False
        sys.argv.pop(1)
    
    if len(sys.argv) == 1 or sys.argv[1] == 'inference':
        # 默认使用skirt_female
        run_tailornet(use_gaussian_splash=use_gaussian_splash)
    elif sys.argv[1] == 'render':
        # 支持可选的服装类别和性别参数
        garment_class = sys.argv[2] if len(sys.argv) > 2 else 'skirt'
        gender = sys.argv[3] if len(sys.argv) > 3 else 'female'
        render_images(garment_class=garment_class, gender=gender)
    elif sys.argv[1] == '--garment_class' and sys.argv[3] == '--gender':
        run_tailornet(garment_class=sys.argv[2], gender=sys.argv[4], use_gaussian_splash=use_gaussian_splash)
    else:
        raise AttributeError("用法: python run_tailornet.py [--no-gaussian] [--garment_class <class> --gender <gender>] [inference|render]")