#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
from scipy.spatial import KDTree
from psbody.mesh import Mesh
import argparse
import glob
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


def evaluate_model_pairs(garment_class='skirt', gender='female', compare_gaussian=True):
    """
    评估所有生成的模型对，计算表面误差
    
    参数:
    - garment_class: 服装类别
    - gender: 性别
    - compare_gaussian: 是否比较高斯溅射前后的结果
    """
    # 创建结果目录
    result_dir = os.path.join(OUT_PATH, 'evaluation_results')
    os.makedirs(result_dir, exist_ok=True)
    
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
    
    # 计算并打印平均指标
    if compare_gaussian and len(results) > 0:
        print_comparison_results(results, garment_class, gender, result_dir)


def print_comparison_results(results, garment_class, gender, output_dir):
    """
    计算并输出高斯溅射前后的比较结果
    
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
    
    # 创建比较结果文件
    comparison_file = os.path.join(output_dir, f"comparison_{garment_class}_{gender}.txt")
    with open(comparison_file, 'w') as f:
        f.write(f"高斯溅射前后指标比较 - {garment_class}_{gender}\n")
        f.write("=" * 50 + "\n\n")
        
        # 距离指标
        f.write("距离指标 (越小越好):\n")
        f.write("-" * 40 + "\n")
        f.write("{:<20} {:<15} {:<15} {:<15}\n".format("指标", "有高斯溅射", "无高斯溅射", "改进百分比"))
        f.write("-" * 40 + "\n")
        
        for m in ['mean_distance', 'median_distance', 'max_distance']:
            improvement = (avg_without[m] - avg_with[m]) / avg_without[m] * 100
            f.write("{:<20} {:<15.6f} {:<15.6f} {:<15.2f}%\n".format(
                m, avg_with[m], avg_without[m], improvement))
        
        # 比例指标
        f.write("\n贴合比例指标 (越大越好):\n")
        f.write("-" * 40 + "\n")
        f.write("{:<20} {:<15} {:<15} {:<15}\n".format("指标", "有高斯溅射", "无高斯溅射", "改进百分比"))
        f.write("-" * 40 + "\n")
        
        for m in ['close_points_ratio_001', 'close_points_ratio_005', 'close_points_ratio_01']:
            improvement = (avg_with[m] - avg_without[m]) / avg_without[m] * 100
            f.write("{:<20} {:<15.6f} {:<15.6f} {:<15.2f}%\n".format(
                m, avg_with[m], avg_without[m], improvement))
    
    # 打印结果到控制台
    print("\n高斯溅射前后指标比较:")
    print("=" * 50)
    
    print("\n距离指标 (越小越好):")
    print("-" * 40)
    print("{:<20} {:<15} {:<15} {:<15}".format("指标", "有高斯溅射", "无高斯溅射", "改进百分比"))
    print("-" * 40)
    
    for m in ['mean_distance', 'median_distance', 'max_distance']:
        improvement = (avg_without[m] - avg_with[m]) / avg_without[m] * 100
        print("{:<20} {:<15.6f} {:<15.6f} {:<15.2f}%".format(
            m, avg_with[m], avg_without[m], improvement))
    
    print("\n贴合比例指标 (越大越好):")
    print("-" * 40)
    print("{:<20} {:<15} {:<15} {:<15}".format("指标", "有高斯溅射", "无高斯溅射", "改进百分比"))
    print("-" * 40)
    
    for m in ['close_points_ratio_001', 'close_points_ratio_005', 'close_points_ratio_01']:
        improvement = (avg_with[m] - avg_without[m]) / avg_without[m] * 100
        print("{:<20} {:<15.6f} {:<15.6f} {:<15.2f}%".format(
            m, avg_with[m], avg_without[m], improvement))
    
    print(f"\n比较结果已保存到: {comparison_file}")


def main():
    parser = argparse.ArgumentParser(description='评估服装贴合度 (简化版)')
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