#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from psbody.mesh import Mesh, MeshViewer
import argparse
import glob
import pandas as pd
import time
from tqdm import tqdm
from datetime import datetime

# 导入全局变量以获取输出路径
import global_var

# 设置输出路径
OUT_PATH = global_var.OUTPUT_PATH

# 配置matplotlib支持中文显示
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']  # 用来正常显示中文
    plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
    print("已配置matplotlib支持中文显示")
except:
    print("警告: matplotlib可能无法正确显示中文，图表标题可能显示为乱码或方框")

class TailorNetEvaluator:
    """TailorNet模型服装贴合度评估器"""
    
    def __init__(self, output_path=None, use_english_labels=False):
        """
        初始化评估器
        
        参数:
        - output_path: 输出路径，默认使用全局变量中的路径
        - use_english_labels: 是否使用英文标签 (避免中文显示问题)
        """
        self.output_path = output_path or OUT_PATH
        self.use_english_labels = use_english_labels
        
        # 创建结果目录
        self.result_dir = os.path.join(self.output_path, 'evaluation_results')
        os.makedirs(self.result_dir, exist_ok=True)
        
        # 创建可视化结果目录
        self.vis_dir = os.path.join(self.result_dir, 'visualization')
        os.makedirs(self.vis_dir, exist_ok=True)
        
        # 创建图表目录
        self.plots_dir = os.path.join(self.result_dir, 'plots')
        os.makedirs(self.plots_dir, exist_ok=True)
        
        print(f"评估结果将保存到: {self.result_dir}")
    
    def calculate_surface_error(self, garment_mesh, body_mesh):
        """
        计算服装网格与身体网格之间的表面误差
        
        参数:
        - garment_mesh: 服装网格 (Mesh 对象)
        - body_mesh: 身体网格 (Mesh 对象)
        
        返回:
        - 误差统计信息字典
        - 每个点的距离数组
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
        }, distances
    
    def visualize_fitting_error(self, garment_mesh, body_mesh, distances, output_path, title):
        """
        可视化服装与身体的贴合误差，并保存图像
        
        参数:
        - garment_mesh: 服装网格
        - body_mesh: 身体网格
        - distances: 距离数组
        - output_path: 输出图像路径
        - title: 图像标题
        
        返回:
        - 是否可视化成功
        """
        # 创建颜色映射
        max_dist = min(0.1, np.max(distances))  # 限制最大距离为0.1，以便更好地可视化近距离差异
        colors = plt.cm.jet(distances / max_dist)
        
        # 将颜色映射到服装网格
        colored_garment = Mesh(v=garment_mesh.v, f=garment_mesh.f)
        colored_garment.vc = colors[:, :3]  # 顶点颜色
        
        # 使用MeshViewer可视化
        try:
            # 修改初始化参数，适配当前版本的MeshViewer
            mv = MeshViewer()
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
    
    def evaluate_model(self, garment_class, gender, frame_id):
        """
        评估单个模型的贴合度
        
        参数:
        - garment_class: 服装类别
        - gender: 性别
        - frame_id: 帧ID
        
        返回:
        - 评估结果字典
        """
        # 构建文件路径
        body_file = os.path.join(self.output_path, f"body_{garment_class}_{gender}_{frame_id}.ply")
        gar_file = os.path.join(self.output_path, f"pred_gar_{garment_class}_{gender}_{frame_id}.ply")
        
        # 检查文件是否存在
        if not os.path.exists(body_file) or not os.path.exists(gar_file):
            print(f"找不到模型文件: {body_file} 或 {gar_file}")
            return None
        
        # 加载模型
        body_mesh = Mesh(filename=body_file)
        gar_mesh = Mesh(filename=gar_file)
        
        # 计算表面误差
        error_stats, distances = self.calculate_surface_error(gar_mesh, body_mesh)
        error_stats['frame_id'] = frame_id
        error_stats['garment_class'] = garment_class
        error_stats['gender'] = gender
        
        # 可视化并保存图像
        vis_path = os.path.join(self.vis_dir, f"fitting_{garment_class}_{gender}_{frame_id}.png")
        self.visualize_fitting_error(gar_mesh, body_mesh, distances, vis_path, 
                                  f"贴合误差: {garment_class}_{gender}_{frame_id}")
        
        # 保存距离分布直方图
        self.plot_distance_histogram(distances, frame_id, garment_class, gender)
        
        return error_stats
    
    def plot_distance_histogram(self, distances, frame_id, garment_class, gender):
        """
        绘制距离分布直方图
        
        参数:
        - distances: 距离数组
        - frame_id: 帧ID
        - garment_class: 服装类别
        - gender: 性别
        """
        plt.figure(figsize=(10, 6))
        plt.hist(distances, bins=50, alpha=0.75, color='blue')
        
        if self.use_english_labels:
            plt.xlabel('Distance (meters)')
            plt.ylabel('Number of Points')
            plt.title(f'Distance Distribution: {garment_class}_{gender}_{frame_id}')
        else:
            plt.xlabel('距离 (米)')
            plt.ylabel('点数')
            plt.title(f'{garment_class}_{gender}_{frame_id} 距离分布')
        
        # 添加关键统计信息
        mean_dist = np.mean(distances)
        median_dist = np.median(distances)
        max_dist = np.max(distances)
        
        if self.use_english_labels:
            plt.axvline(mean_dist, color='r', linestyle='dashed', linewidth=1, label=f'Mean: {mean_dist:.4f}')
            plt.axvline(median_dist, color='g', linestyle='dashed', linewidth=1, label=f'Median: {median_dist:.4f}')
            
            # 标注阈值线
            plt.axvline(0.01, color='orange', linestyle='dotted', linewidth=1, label='Threshold: 0.01')
            plt.axvline(0.05, color='purple', linestyle='dotted', linewidth=1, label='Threshold: 0.05')
            plt.axvline(0.1, color='brown', linestyle='dotted', linewidth=1, label='Threshold: 0.1')
        else:
            plt.axvline(mean_dist, color='r', linestyle='dashed', linewidth=1, label=f'平均值: {mean_dist:.4f}')
            plt.axvline(median_dist, color='g', linestyle='dashed', linewidth=1, label=f'中位值: {median_dist:.4f}')
            
            # 标注阈值线
            plt.axvline(0.01, color='orange', linestyle='dotted', linewidth=1, label='阈值: 0.01')
            plt.axvline(0.05, color='purple', linestyle='dotted', linewidth=1, label='阈值: 0.05')
            plt.axvline(0.1, color='brown', linestyle='dotted', linewidth=1, label='阈值: 0.1')
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 保存图像
        hist_path = os.path.join(self.plots_dir, f"hist_{garment_class}_{gender}_{frame_id}.png")
        plt.savefig(hist_path)
        plt.close()
    
    def evaluate_all_models(self, garment_class='skirt', gender='female'):
        """
        评估所有匹配的模型
        
        参数:
        - garment_class: 服装类别
        - gender: 性别
        
        返回:
        - 评估结果列表
        """
        # 查找所有匹配的模型
        pattern = f"body_{garment_class}_{gender}_*.ply"
        body_files = glob.glob(os.path.join(self.output_path, pattern))
        
        if not body_files:
            print(f"没有找到匹配的模型: {pattern}")
            return []
        
        print(f"找到 {len(body_files)} 个匹配的模型")
        
        # 评估所有模型
        results = []
        for body_file in tqdm(body_files, desc=f"评估 {garment_class}_{gender} 模型"):
            # 从文件名中提取帧号
            frame_id = os.path.basename(body_file).replace(f"body_{garment_class}_{gender}_", "").replace(".ply", "")
            
            # 评估模型
            result = self.evaluate_model(garment_class, gender, frame_id)
            if result:
                results.append(result)
        
        # 保存结果
        if results:
            self.save_results(results, garment_class, gender)
            self.generate_trend_plots(results, garment_class, gender)
        
        return results
    
    def save_results(self, results, garment_class, gender):
        """
        保存评估结果
        
        参数:
        - results: 评估结果列表
        - garment_class: 服装类别
        - gender: 性别
        """
        # 转换为DataFrame
        df = pd.DataFrame(results)
        
        # 按帧ID排序
        df['frame_id'] = df['frame_id'].astype(str).str.zfill(4)
        df = df.sort_values('frame_id')
        
        # 保存为CSV文件
        csv_path = os.path.join(self.result_dir, f"evaluation_{garment_class}_{gender}.csv")
        df.to_csv(csv_path, index=False)
        print(f"评估结果已保存到: {csv_path}")
        
        # 尝试保存为Excel文件，如果有openpyxl库的话
        try:
            excel_path = os.path.join(self.result_dir, f"evaluation_{garment_class}_{gender}.xlsx")
            df.to_excel(excel_path, index=False)
            print(f"评估结果已保存到: {excel_path}")
        except ImportError:
            print("警告: 缺少openpyxl库，无法保存Excel格式。请使用pip install openpyxl安装。")
        except Exception as e:
            print(f"保存Excel文件时出错: {e}")
        
        # 生成摘要报告
        self.generate_summary_report(df, garment_class, gender)
    
    def generate_summary_report(self, df, garment_class, gender):
        """
        生成摘要报告
        
        参数:
        - df: 评估结果DataFrame
        - garment_class: 服装类别
        - gender: 性别
        """
        report_path = os.path.join(self.result_dir, f"summary_{garment_class}_{gender}.md")
        
        with open(report_path, 'w') as f:
            if self.use_english_labels:
                f.write(f"# TailorNet Model Evaluation Summary - {garment_class}_{gender}\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write("## Evaluation Metrics Overview\n\n")
                
                # 计算平均值
                mean_stats = df.mean(numeric_only=True)
                
                f.write("### Distance Metrics (Lower is Better)\n\n")
                f.write("| Metric | Average | Minimum | Maximum |\n")
                f.write("|--------|---------|---------|--------|\n")
                
                for metric in ['mean_distance', 'median_distance', 'max_distance']:
                    f.write(f"| {metric} | {mean_stats[metric]:.6f} | {df[metric].min():.6f} | {df[metric].max():.6f} |\n")
                
                f.write("\n### Fitting Ratio Metrics (Higher is Better)\n\n")
                f.write("| Metric | Average | Minimum | Maximum |\n")
                f.write("|--------|---------|---------|--------|\n")
                
                for metric in ['close_points_ratio_001', 'close_points_ratio_005', 'close_points_ratio_01']:
                    f.write(f"| {metric} | {mean_stats[metric]:.6f} | {df[metric].min():.6f} | {df[metric].max():.6f} |\n")
                
                f.write("\n## Visualization Results\n\n")
                f.write("Detailed visualization results can be found in the following directories:\n\n")
                f.write(f"- Fitting error heatmaps: `{self.vis_dir}`\n")
                f.write(f"- Distance distribution histograms: `{self.plots_dir}`\n")
                f.write(f"- Trend charts: `{self.plots_dir}`\n")
            else:
                f.write(f"# TailorNet 模型评估摘要 - {garment_class}_{gender}\n\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write("## 评估指标概览\n\n")
                
                # 计算平均值
                mean_stats = df.mean(numeric_only=True)
                
                f.write("### 距离指标 (越小越好)\n\n")
                f.write("| 指标 | 平均值 | 最小值 | 最大值 |\n")
                f.write("|------|--------|--------|--------|\n")
                
                for metric in ['mean_distance', 'median_distance', 'max_distance']:
                    f.write(f"| {metric} | {mean_stats[metric]:.6f} | {df[metric].min():.6f} | {df[metric].max():.6f} |\n")
                
                f.write("\n### 贴合比例指标 (越大越好)\n\n")
                f.write("| 指标 | 平均值 | 最小值 | 最大值 |\n")
                f.write("|------|--------|--------|--------|\n")
                
                for metric in ['close_points_ratio_001', 'close_points_ratio_005', 'close_points_ratio_01']:
                    f.write(f"| {metric} | {mean_stats[metric]:.6f} | {df[metric].min():.6f} | {df[metric].max():.6f} |\n")
                
                f.write("\n## 评估结果可视化\n\n")
                f.write("详细的可视化结果可在以下目录中找到：\n\n")
                f.write(f"- 贴合度热图: `{self.vis_dir}`\n")
                f.write(f"- 距离分布直方图: `{self.plots_dir}`\n")
                f.write(f"- 趋势图表: `{self.plots_dir}`\n")
        
        print(f"摘要报告已保存到: {report_path}")
    
    def generate_trend_plots(self, results, garment_class, gender):
        """
        生成评估指标随时间变化的趋势图
        
        参数:
        - results: 评估结果列表
        - garment_class: 服装类别
        - gender: 性别
        """
        # 转换为DataFrame
        df = pd.DataFrame(results)
        
        # 按帧ID排序
        df['frame_id'] = df['frame_id'].astype(str).str.zfill(4)
        df = df.sort_values('frame_id')
        
        # 距离指标趋势图
        plt.figure(figsize=(12, 6))
        plt.plot(df['frame_id'], df['mean_distance'], 'o-', label='平均距离' if not self.use_english_labels else 'Mean Distance')
        plt.plot(df['frame_id'], df['median_distance'], 's-', label='中位距离' if not self.use_english_labels else 'Median Distance')
        plt.plot(df['frame_id'], df['max_distance'], '^-', label='最大距离' if not self.use_english_labels else 'Max Distance')
        
        if self.use_english_labels:
            plt.xlabel('Frame ID')
            plt.ylabel('Distance (meters)')
            plt.title(f'{garment_class}_{gender} Distance Metrics Trend')
        else:
            plt.xlabel('帧ID')
            plt.ylabel('距离 (米)')
            plt.title(f'{garment_class}_{gender} 距离指标趋势')
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # 保存图像
        trend_path = os.path.join(self.plots_dir, f"trend_distance_{garment_class}_{gender}.png")
        plt.savefig(trend_path)
        plt.close()
        
        # 贴合比例指标趋势图
        plt.figure(figsize=(12, 6))
        
        if self.use_english_labels:
            plt.plot(df['frame_id'], df['close_points_ratio_001'], 'o-', label='Points within 0.01')
            plt.plot(df['frame_id'], df['close_points_ratio_005'], 's-', label='Points within 0.05')
            plt.plot(df['frame_id'], df['close_points_ratio_01'], '^-', label='Points within 0.1')
            
            plt.xlabel('Frame ID')
            plt.ylabel('Ratio')
            plt.title(f'{garment_class}_{gender} Fitting Ratio Trend')
        else:
            plt.plot(df['frame_id'], df['close_points_ratio_001'], 'o-', label='距离<0.01的点比例')
            plt.plot(df['frame_id'], df['close_points_ratio_005'], 's-', label='距离<0.05的点比例')
            plt.plot(df['frame_id'], df['close_points_ratio_01'], '^-', label='距离<0.1的点比例')
            
            plt.xlabel('帧ID')
            plt.ylabel('比例')
            plt.title(f'{garment_class}_{gender} 贴合比例指标趋势')
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # 保存图像
        ratio_path = os.path.join(self.plots_dir, f"trend_ratio_{garment_class}_{gender}.png")
        plt.savefig(ratio_path)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='TailorNet模型服装贴合度评估')
    parser.add_argument('--garment_class', type=str, default='skirt', help='服装类别')
    parser.add_argument('--gender', type=str, default='female', help='性别')
    parser.add_argument('--frame_id', type=str, help='特定帧ID，如果不指定则评估所有帧')
    parser.add_argument('--output_path', type=str, help='自定义输出路径')
    parser.add_argument('--use_english', action='store_true', help='使用英文标签（避免中文显示问题）')
    
    args = parser.parse_args()
    
    # 创建评估器
    evaluator = TailorNetEvaluator(output_path=args.output_path, use_english_labels=args.use_english)
    
    if args.frame_id:
        # 评估单个模型
        print(f"评估单个模型: {args.garment_class}_{args.gender}_{args.frame_id}")
        result = evaluator.evaluate_model(args.garment_class, args.gender, args.frame_id)
        if result:
            print("评估结果:")
            for k, v in result.items():
                print(f"  {k}: {v}")
    else:
        # 评估所有模型
        print(f"评估所有 {args.garment_class}_{args.gender} 模型")
        evaluator.evaluate_all_models(args.garment_class, args.gender)
    
    print("评估完成！")


if __name__ == "__main__":
    main() 