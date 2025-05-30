#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from psbody.mesh import Mesh, MeshViewer
import argparse
import glob
import time
from tqdm import tqdm

# 导入全局变量以获取输出路径
import global_var

# 设置输出路径
OUT_PATH = global_var.OUTPUT_PATH


def calculate_surface_error(garment_mesh, body_mesh):
    """
    计算服装网格与身体网格之间的表面误差
    
    参数:
    - garment_mesh: 服装网格 (Mesh 对象)
    - body_mesh: 身体网格 (Mesh 对象)
    
    返回:
    - 误差统计信息字典
    """
    # 构建KD树用于快速查找最近点
    body_tree = KDTree(body_mesh.v)
    
    # 计算每个服装顶点到人体的最短距离
    distances, indices = body_tree.query(garment_mesh.v)
    
    # 计算统计信息
    mean_dist = np.mean(distances)
    median_dist = np.median(distances)
    max_dist = np.max(distances)
    min_dist = np.min(distances)
    std_dist = np.std(distances)
    
    # 计算不同距离阈值下的贴合点百分比
    close_ratio_001 = np.sum(distances < 0.01) / len(distances)  # 0.01单位内的点的比例
    close_ratio_005 = np.sum(distances < 0.05) / len(distances)  # 0.05单位内的点的比例
    close_ratio_01 = np.sum(distances < 0.1) / len(distances)   # 0.1单位内的点的比例
    
    return {
        "mean_distance": mean_dist,
        "median_distance": median_dist,
        "max_distance": max_dist,
        "min_distance": min_dist,
        "std_distance": std_dist,
        "close_points_ratio_001": close_ratio_001,
        "close_points_ratio_005": close_ratio_005,
        "close_points_ratio_01": close_ratio_01
    }


def visualize_fitting_error(garment_mesh, body_mesh, output_path, title):
    """
    可视化服装与身体的贴合误差，并保存图像
    
    参数:
    - garment_mesh: 服装网格
    - body_mesh: 身体网格
    - output_path: 输出图像路径
    - title: 图像标题
    """
    # 构建身体KD树
    body_tree = KDTree(body_mesh.v)
    
    # 计算距离
    distances, _ = body_tree.query(garment_mesh.v)
    
    # 创建颜色映射
    max_dist = min(0.1, np.max(distances))  # 限制最大距离为0.1，以便更好地可视化近距离差异
    colors = plt.cm.jet(distances / max_dist)
    
    # 将颜色映射到服装网格
    colored_garment = Mesh(v=garment_mesh.v, f=garment_mesh.f)
    colored_garment.vc = colors[:, :3]  # 顶点颜色
    
    # 使用MeshViewer可视化
    try:
        mv = MeshViewer(width=1024, height=768)
        mv.static_meshes = [body_mesh, colored_garment]
        mv.save_snapshot(output_path)
        
        # 生成颜色条图例
        plt.figure(figsize=(8, 1))
        plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.jet), 
                     orientation='horizontal', 
                     label='Distance (meters)')
        plt.title(title)
        legend_path = output_path.replace('.png', '_legend.png')
        plt.savefig(legend_path)
        plt.close()
        
        return True
    except Exception as e:
        print(f"可视化失败: {e}")
        return False


def evaluate_model_pairs(garment_class='skirt', gender='female', compare_gaussian=True):
    """
    评估所有生成的模型对，计算表面误差并可视化结果
    
    参数:
    - garment_class: 服装类别
    - gender: 性别
    - compare_gaussian: 是否比较高斯溅射前后的结果
    """
    # 创建结果目录
    result_dir = os.path.join(OUT_PATH, 'evaluation_results')
    os.makedirs(result_dir, exist_ok=True)
    
    # 创建可视化结果目录
    vis_dir = os.path.join(result_dir, 'visualization')
    os.makedirs(vis_dir, exist_ok=True)
    
    # 查找所有生成的模型对
    pattern = f"body_{garment_class}_{gender}_*.ply"
    body_files = glob.glob(os.path.join(OUT_PATH, pattern))
    
    if not body_files:
        print(f"没有找到匹配的模型: {pattern}")
        return
    
    # 初始化结果列表
    results = []
    
    # 遍历所有模型对
    for body_file in tqdm(body_files, desc="评估模型"):
        # 从文件名中提取帧号
        frame_id = os.path.basename(body_file).replace(f"body_{garment_class}_{gender}_", "").replace(".ply", "")
        
        # 加载身体模型
        body_mesh = Mesh(filename=body_file)
        
        # 加载处理后的服装模型
        gar_file = os.path.join(OUT_PATH, f"pred_gar_{garment_class}_{gender}_{frame_id}.ply")
        if not os.path.exists(gar_file):
            print(f"找不到服装模型: {gar_file}")
            continue
        gar_mesh = Mesh(filename=gar_file)
        
        # 计算表面误差
        error_stats = calculate_surface_error(gar_mesh, body_mesh)
        error_stats['frame_id'] = frame_id
        error_stats['model_type'] = 'with_gaussian' if compare_gaussian else 'standard'
        
        # 将结果添加到列表中
        results.append(error_stats)
        
        # 可视化并保存图像
        vis_path = os.path.join(vis_dir, f"fitting_{garment_class}_{gender}_{frame_id}.png")
        visualize_fitting_error(gar_mesh, body_mesh, vis_path, 
                               f"贴合误差: {garment_class}_{gender}_{frame_id}")
        
        # 如果需要比较高斯溅射前后的结果
        if compare_gaussian:
            # 加载原始服装模型（未应用高斯溅射）
            orig_gar_file = os.path.join(OUT_PATH, f"pred_gar_original_{garment_class}_{gender}_{frame_id}.ply")
            if os.path.exists(orig_gar_file):
                orig_gar_mesh = Mesh(filename=orig_gar_file)
                
                # 计算原始模型的表面误差
                orig_error_stats = calculate_surface_error(orig_gar_mesh, body_mesh)
                orig_error_stats['frame_id'] = frame_id
                orig_error_stats['model_type'] = 'without_gaussian'
                
                # 将结果添加到列表中
                results.append(orig_error_stats)
                
                # 可视化并保存图像
                orig_vis_path = os.path.join(vis_dir, f"fitting_original_{garment_class}_{gender}_{frame_id}.png")
                visualize_fitting_error(orig_gar_mesh, body_mesh, orig_vis_path, 
                                      f"原始贴合误差: {garment_class}_{gender}_{frame_id}")
    
    # 将结果保存为CSV文件
    results_file = os.path.join(result_dir, f"fitting_metrics_{garment_class}_{gender}.csv")
    with open(results_file, 'w') as f:
        # 写入表头
        headers = list(results[0].keys())
        f.write(','.join(headers) + '\n')
        
        # 写入数据
        for result in results:
            values = [str(result[h]) for h in headers]
            f.write(','.join(values) + '\n')
    
    print(f"评估完成，结果已保存到: {results_file}")
    
    # 如果比较高斯溅射前后的结果，生成对比图表
    if compare_gaussian and len(results) > 0:
        generate_comparison_plots(results, garment_class, gender, result_dir)


def generate_comparison_plots(results, garment_class, gender, output_dir):
    """
    生成比较高斯溅射前后效果的图表
    
    参数:
    - results: 评估结果列表
    - garment_class: 服装类别
    - gender: 性别
    - output_dir: 输出目录
    """
    # 将结果分组为有高斯溅射和无高斯溅射
    with_gaussian = [r for r in results if r['model_type'] == 'with_gaussian']
    without_gaussian = [r for r in results if r['model_type'] == 'without_gaussian']
    
    # 确保两组数据具有相同数量的样本
    if len(with_gaussian) != len(without_gaussian):
        print("警告: 有/无高斯溅射的样本数量不匹配")
        return
    
    # 计算平均指标
    metrics = ['mean_distance', 'median_distance', 'max_distance', 
               'close_points_ratio_001', 'close_points_ratio_005', 'close_points_ratio_01']
    
    avg_with = {m: np.mean([r[m] for r in with_gaussian]) for m in metrics}
    avg_without = {m: np.mean([r[m] for r in without_gaussian]) for m in metrics}
    
    # 创建比较图表
    plt.figure(figsize=(12, 8))
    
    # 距离指标（越小越好）
    plt.subplot(2, 1, 1)
    distance_metrics = ['mean_distance', 'median_distance', 'max_distance']
    x = np.arange(len(distance_metrics))
    width = 0.35
    
    with_values = [avg_with[m] for m in distance_metrics]
    without_values = [avg_without[m] for m in distance_metrics]
    
    plt.bar(x - width/2, with_values, width, label='有高斯溅射')
    plt.bar(x + width/2, without_values, width, label='无高斯溅射')
    
    plt.xlabel('指标')
    plt.ylabel('距离 (米)')
    plt.title(f'{garment_class}_{gender} 距离指标比较 (越小越好)')
    plt.xticks(x, distance_metrics)
    plt.legend()
    
    # 比例指标（越大越好）
    plt.subplot(2, 1, 2)
    ratio_metrics = ['close_points_ratio_001', 'close_points_ratio_005', 'close_points_ratio_01']
    x = np.arange(len(ratio_metrics))
    
    with_values = [avg_with[m] for m in ratio_metrics]
    without_values = [avg_without[m] for m in ratio_metrics]
    
    plt.bar(x - width/2, with_values, width, label='有高斯溅射')
    plt.bar(x + width/2, without_values, width, label='无高斯溅射')
    
    plt.xlabel('指标')
    plt.ylabel('比例')
    plt.title(f'{garment_class}_{gender} 贴合比例指标比较 (越大越好)')
    plt.xticks(x, [m.replace('close_points_ratio_', '距离<') for m in ratio_metrics])
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"comparison_{garment_class}_{gender}.png"))
    plt.close()
    
    # 创建改进百分比表格
    improvement_file = os.path.join(output_dir, f"improvement_{garment_class}_{gender}.txt")
    with open(improvement_file, 'w') as f:
        f.write(f"高斯溅射前后指标改进百分比 - {garment_class}_{gender}\n")
        f.write("=" * 50 + "\n\n")
        
        # 距离指标（越小越好，所以减少为正向改进）
        f.write("距离指标 (减少为改进):\n")
        for m in distance_metrics:
            improvement = (avg_without[m] - avg_with[m]) / avg_without[m] * 100
            f.write(f"{m}: {improvement:.2f}%\n")
        
        # 比例指标（越大越好，所以增加为正向改进）
        f.write("\n贴合比例指标 (增加为改进):\n")
        for m in ratio_metrics:
            improvement = (avg_with[m] - avg_without[m]) / avg_without[m] * 100
            f.write(f"{m}: {improvement:.2f}%\n")
    
    print(f"比较图表和改进百分比已保存到: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='评估服装贴合度')
    parser.add_argument('--garment_class', type=str, default='skirt', help='服装类别')
    parser.add_argument('--gender', type=str, default='female', help='性别')
    parser.add_argument('--no-compare', action='store_false', dest='compare_gaussian',
                      help='不比较高斯溅射前后的结果')
    
    args = parser.parse_args()
    
    print(f"开始评估 {args.garment_class}_{args.gender} 模型...")
    evaluate_model_pairs(args.garment_class, args.gender, args.compare_gaussian)
    print("评估完成！")


if __name__ == "__main__":
    main() 