#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
提取HybrikAPI高级参数并应用于渲染过程的脚本
实现两步法方案中的参数提取和精确渲染

作者: Claude
日期: 2023-05-15
"""

import os
import sys
import json
import numpy as np
import cv2
import glob
import re
import argparse
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import trimesh
from pyrender import Scene, Mesh, Viewer, OffscreenRenderer, RenderFlags, DirectionalLight, PerspectiveCamera
from scipy.spatial import KDTree

# 导入全局变量
try:
    import global_var
    OUT_PATH = global_var.OUTPUT_PATH
except ImportError:
    print("警告: 未找到global_var模块，将使用当前目录作为输出路径")
    OUT_PATH = os.path.join(os.getcwd(), "output")
    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH, exist_ok=True)

class HybrikParamsExtractor:
    """从HybrikAPI结果中提取高级参数并应用于渲染过程的类"""
    
    def __init__(self, output_dir=None):
        """
        初始化参数提取器
        
        参数:
        - output_dir: 输出目录，默认使用OUT_PATH
        """
        self.output_dir = output_dir or OUT_PATH
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        
        # 默认渲染参数
        self.default_render_size = (1024, 1024)
        self.default_bg_color = (1.0, 1.0, 1.0, 0.0)  # 带有透明度的白色背景
        
        # 姿势矩阵的缓存
        self.pose_matrices = {}  # {person_id: {frame_id: transformation_matrix}}
        
    def _ensure_valid_path(self, path):
        """尝试转换为有效路径，如果失败则返回原始路径"""
        if not path:
            return path
            
        # 如果路径已经存在，直接返回绝对路径
        if os.path.exists(path):
            return os.path.abspath(path)
            
        # 尝试各种路径转换
        alternative_paths = []
        
        # 添加绝对路径
        if not os.path.isabs(path):
            abs_path = os.path.abspath(path)
            alternative_paths.append(abs_path)
            
        # 检查是否是Windows路径(在WSL中)
        if '\\' in path or ':' in path:
            # 尝试转换为WSL路径
            if ':' in path:  # 包含驱动器号 (如 C:)
                drive, rest = path.split(':', 1)
                # 修复语法错误：先处理字符串替换，然后再插入f-string
                replaced_path = rest.replace('\\', '/')
                wsl_path = f"/mnt/{drive.lower()}{replaced_path}"
                alternative_paths.append(wsl_path)
        else:  # 可能是Linux路径
            # 检查当前目录拼接
            rel_path = os.path.join(os.getcwd(), path)
            alternative_paths.append(rel_path)
            
        # 尝试所有可能的路径
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                return alt_path
                
        # 如果找不到有效路径，返回原始路径
        return path

    def load_hybrik_results(self, hybrik_result_file):
        """
        加载HybrikAPI结果文件
        
        参数:
        - hybrik_result_file: HybrikAPI结果文件路径
        
        返回:
        - hybrik_data: 解析后的HybrikAPI数据
        """
        print(f"加载HybrikAPI结果: {hybrik_result_file}")
        hybrik_result_file = self._ensure_valid_path(hybrik_result_file)
        
        if not os.path.exists(hybrik_result_file):
            print(f"错误: HybrikAPI结果文件不存在: {hybrik_result_file}")
            return None
        
        try:
            with open(hybrik_result_file, 'r') as f:
                hybrik_data = json.load(f)
            
            print(f"成功加载HybrikAPI结果，包含{len(hybrik_data)}帧数据")
            return hybrik_data
        except Exception as e:
            print(f"加载HybrikAPI结果时出错: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def extract_parameters(self, hybrik_data, start_frame=0, end_frame=None, person_ids=None):
        """
        从HybrikAPI结果中提取关键参数
        
        参数:
        - hybrik_data: HybrikAPI数据
        - start_frame: 起始帧
        - end_frame: 结束帧
        - person_ids: 人物ID列表，None表示处理所有人物
        
        返回:
        - 参数字典
        """
        if not hybrik_data:
            print("错误: 无有效的HybrikAPI数据")
            return None
            
        # 设置帧范围
        if end_frame is None:
            end_frame = len(hybrik_data)
        else:
            end_frame = min(end_frame, len(hybrik_data))
            
        # 初始化参数字典
        params = {}
        
        # 分析第一帧以确定数据结构
        first_frame = hybrik_data[0]
        data_format = self._detect_data_format(first_frame)
        print(f"检测到的数据格式: {data_format}")
        
        # 提取每一帧的参数
        print("提取HybrikAPI参数...")
        for frame_idx in tqdm(range(start_frame, end_frame)):
            if frame_idx >= len(hybrik_data):
                break
                
            frame_data = hybrik_data[frame_idx]
            
            # 根据不同的数据格式提取人物数据
            people_list = self._extract_people_from_frame(frame_data, data_format)
            
            # 对每个人提取参数
            for person_idx, person_data in enumerate(people_list):
                # 检查是否处理此人物
                if person_ids is not None and person_idx not in person_ids:
                    continue
                    
                # 提取此人物的参数
                if person_idx not in params:
                    params[person_idx] = {}
                    
                # 提取边界框
                bbox = self._extract_bbox(person_data)
                
                # 提取姿势矩阵
                pose_matrix = self._extract_pose_matrix(person_data)
                
                # 提取相机参数
                camera_params = self._extract_camera_params(person_data)
                
                # 存储参数
                params[person_idx][frame_idx] = {
                    'bbox': bbox,
                    'pose_matrix': pose_matrix,
                    'camera_params': camera_params
                }
                
        # 将提取的参数保存到类属性中
        self.pose_matrices = params
        
        print(f"成功提取{len(params)}个人物的参数")
        return params
    
    def _detect_data_format(self, frame_data):
        """检测HybrikAPI数据的格式"""
        if isinstance(frame_data, dict):
            if 'results' in frame_data:
                return 'results_dict'
            elif 'track_id' in frame_data or 'bbox' in frame_data:
                return 'person_dict'
        elif isinstance(frame_data, list):
            return 'person_list'
        return 'unknown'
    
    def _extract_people_from_frame(self, frame_data, data_format):
        """从帧数据中提取人物列表"""
        if data_format == 'results_dict':
            return frame_data.get('results', [])
        elif data_format == 'person_list':
            return frame_data
        elif data_format == 'person_dict':
            return [frame_data]
        return []
    
    def _extract_bbox(self, person_data):
        """从人物数据中提取边界框"""
        if not isinstance(person_data, dict):
            return None
            
        # 尝试不同的可能键名
        for key in ['bbox', 'bboxes', 'box']:
            if key in person_data:
                bbox = person_data[key]
                if isinstance(bbox, list) and len(bbox) == 4:
                    return bbox
        
        # 如果找不到边界框，尝试从关键点估计
        if 'keypoints' in person_data or 'joints' in person_data:
            keypoints = person_data.get('keypoints', person_data.get('joints', []))
            if keypoints and len(keypoints) > 0:
                # 从关键点计算边界框
                if isinstance(keypoints, list):
                    # 过滤掉无效关键点
                    valid_points = [p for p in keypoints if len(p) >= 2 and p[0] is not None and p[1] is not None]
                    if valid_points:
                        x_coords = [p[0] for p in valid_points]
                        y_coords = [p[1] for p in valid_points]
                        x1, y1 = min(x_coords), min(y_coords)
                        x2, y2 = max(x_coords), max(y_coords)
                        # 添加边距
                        margin = 0.1  # 10%的边距
                        width, height = x2 - x1, y2 - y1
                        x1 -= width * margin
                        y1 -= height * margin
                        x2 += width * margin
                        y2 += height * margin
                        return [x1, y1, x2, y2]
        
        return None
    
    def _extract_pose_matrix(self, person_data):
        """从人物数据中提取姿势矩阵"""
        if not isinstance(person_data, dict):
            return np.eye(4)  # 默认为单位矩阵
            
        # 尝试不同的可能键名
        # 首先尝试直接获取姿势矩阵
        for key in ['pred_thetas_mat', 'pose_mat', 'poses_mat', 'global_poses']:
            if key in person_data:
                mat = person_data[key]
                if isinstance(mat, list) and len(mat) == 4 and len(mat[0]) == 4:
                    return np.array(mat)
        
        # 然后尝试获取旋转和平移向量
        rot = None
        trans = None
        
        for rot_key in ['rot', 'rotation', 'global_orient', 'global_rotation']:
            if rot_key in person_data:
                rot_data = person_data[rot_key]
                if isinstance(rot_data, list):
                    rot = np.array(rot_data)
                    break
        
        for trans_key in ['trans', 'translation', 'global_translation']:
            if trans_key in person_data:
                trans_data = person_data[trans_key]
                if isinstance(trans_data, list):
                    trans = np.array(trans_data)
                    break
        
        # 如果有旋转和平移，创建变换矩阵
        if rot is not None and trans is not None:
            matrix = np.eye(4)
            if rot.shape == (3, 3):
                matrix[:3, :3] = rot
            elif rot.shape == (3,):
                from scipy.spatial.transform import Rotation
                matrix[:3, :3] = Rotation.from_rotvec(rot).as_matrix()
            
            if trans.shape == (3,):
                matrix[:3, 3] = trans
            
            return matrix
        
        # 如果没有找到姿势信息，尝试从关键点估计简单的变换
        if 'keypoints' in person_data or 'joints' in person_data:
            # 这里可以实现从关键点估计姿势的算法
            # 但简单起见，我们现在就返回一个单位矩阵
            pass
        
        return np.eye(4)  # 默认为单位矩阵
    
    def _extract_camera_params(self, person_data):
        """从人物数据中提取相机参数"""
        if not isinstance(person_data, dict):
            return None
            
        # 尝试不同的可能键名
        for key in ['camera', 'cam', 'cam_param', 'cam_params', 'camera_params']:
            if key in person_data:
                cam_data = person_data[key]
                if isinstance(cam_data, list) or isinstance(cam_data, dict):
                    return cam_data
        
        # 如果有内部参数（focal length等）
        for key in ['focal_length', 'focal', 'f']:
            if key in person_data:
                return [person_data[key]]  # 只返回焦距
        
        return None
    
    def save_parameters(self, params, output_file=None):
        """
        保存提取的参数到文件
        
        参数:
        - params: 参数字典
        - output_file: 输出文件路径
        
        返回:
        - 保存的文件路径
        """
        if output_file is None:
            output_file = os.path.join(self.output_dir, f"hybrik_params_{int(time.time())}.json")
            
        # 将NumPy数组转换为列表以便JSON序列化
        json_params = {}
        for person_id, person_params in params.items():
            json_params[str(person_id)] = {}
            for frame_id, frame_params in person_params.items():
                json_params[str(person_id)][str(frame_id)] = {}
                for param_name, param_value in frame_params.items():
                    if param_name == 'pose_matrix' and isinstance(param_value, np.ndarray):
                        json_params[str(person_id)][str(frame_id)][param_name] = param_value.tolist()
                    else:
                        json_params[str(person_id)][str(frame_id)][param_name] = param_value
        
        try:
            with open(output_file, 'w') as f:
                json.dump(json_params, f, indent=2)
            print(f"参数已保存到: {output_file}")
            return output_file
        except Exception as e:
            print(f"保存参数时出错: {e}")
            return None
    
    def render_with_params(self, models_dir, output_dir=None, start_frame=0, end_frame=None, 
                         person_ids=None, render_size=None, flip_vertical=True):
        """
        使用提取的参数渲染模型
        
        参数:
        - models_dir: 包含模型的目录
        - output_dir: 输出目录
        - start_frame: 起始帧
        - end_frame: 结束帧
        - person_ids: 人物ID列表
        - render_size: 渲染尺寸
        - flip_vertical: 是否垂直翻转模型
        
        返回:
        - 渲染的帧信息
        """
        if not self.pose_matrices:
            print("错误: 未提取参数，请先调用extract_parameters")
            return None
            
        # 设置输出目录
        if output_dir is None:
            output_dir = os.path.join(self.output_dir, f"param_renders_{int(time.time())}")
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        # 设置渲染尺寸
        if render_size is None:
            render_size = self.default_render_size
            
        # 如果未指定人物ID，使用所有提取的人物
        if person_ids is None:
            person_ids = list(self.pose_matrices.keys())
            
        # 处理每个人物
        rendered_frames = {}
        for person_id in person_ids:
            if person_id not in self.pose_matrices:
                print(f"警告: 未找到人物{person_id}的参数")
                continue
                
            print(f"渲染人物{person_id}...")
            person_frames = self.pose_matrices[person_id]
            
            # 为该人物创建输出目录
            person_dir = os.path.join(output_dir, f"person{person_id}")
            os.makedirs(person_dir, exist_ok=True)
            
            # 渲染每一帧
            person_rendered_frames = []
            for frame_id in sorted(person_frames.keys()):
                if frame_id < start_frame:
                    continue
                if end_frame is not None and frame_id > end_frame:
                    continue
                    
                # 获取参数
                frame_params = person_frames[frame_id]
                bbox = frame_params['bbox']
                pose_matrix = frame_params['pose_matrix']
                camera_params = frame_params['camera_params']
                
                # 查找对应的模型文件
                body_file, garment_file = self._find_model_files(models_dir, frame_id, person_id)
                
                if not body_file or not garment_file:
                    print(f"警告: 未找到帧{frame_id}人物{person_id}的模型文件")
                    continue
                
                # 输出图像路径
                output_image = os.path.join(person_dir, f"frame_{frame_id:04d}.png")
                
                # 渲染模型
                success = self._render_model_with_params(
                    body_file, 
                    garment_file,
                    pose_matrix,
                    camera_params,
                    bbox,
                    output_image,
                    render_size,
                    flip_vertical
                )
                
                if success:
                    person_rendered_frames.append((frame_id, output_image))
                    
            rendered_frames[person_id] = person_rendered_frames
            print(f"人物{person_id}渲染了{len(person_rendered_frames)}帧")
            
        return rendered_frames
    
    def _find_model_files(self, models_dir, frame_id, person_id):
        """查找特定帧和人物的模型文件"""
        models_dir = self._ensure_valid_path(models_dir)
        
        # 查找body文件
        body_pattern = os.path.join(models_dir, f"body_*frame{frame_id:04d}_person{person_id:02d}.ply")
        body_candidates = glob.glob(body_pattern)
        if not body_candidates:
            # 尝试不同的命名格式
            body_pattern = os.path.join(models_dir, f"body_*{frame_id:04d}*{person_id:02d}*.ply")
            body_candidates = glob.glob(body_pattern)
            
        if not body_candidates:
            return None, None
            
        body_file = body_candidates[0]
        
        # 查找对应的garment文件
        garment_file = body_file.replace("body_", "pred_gar_")
        if not os.path.exists(garment_file):
            # 尝试查找任何匹配的服装文件
            garment_pattern = os.path.join(models_dir, f"pred_gar_*frame{frame_id:04d}_person{person_id:02d}.ply")
            garment_candidates = glob.glob(garment_pattern)
            if not garment_candidates:
                return body_file, None
            garment_file = garment_candidates[0]
            
        return body_file, garment_file
    
    def _render_model_with_params(self, body_file, garment_file, pose_matrix, camera_params, 
                               bbox, output_image, render_size, flip_vertical=True):
        """使用指定参数渲染单帧模型"""
        try:
            # 加载模型
            body_mesh = trimesh.load(body_file)
            garment_mesh = trimesh.load(garment_file)
            
            # 如果需要垂直翻转
            if flip_vertical:
                flip_matrix = np.array([
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0]
                ])
                body_mesh.apply_transform(flip_matrix)
                garment_mesh.apply_transform(flip_matrix)
            
            # 设置场景
            width, height = render_size
            scene = Scene(bg_color=[0.0, 0.0, 0.0, 0.0])  # 透明背景
            
            # 计算适当的相机参数
            focal_length = None
            if camera_params:
                if isinstance(camera_params, list) and len(camera_params) > 0:
                    focal_length = float(camera_params[0])
                elif isinstance(camera_params, dict) and 'focal_length' in camera_params:
                    focal_length = float(camera_params['focal_length'])
            
            if focal_length is None:
                # 如果没有焦距信息，使用边界框估计
                if bbox:
                    x1, y1, x2, y2 = bbox
                    box_size = max(x2 - x1, y2 - y1)
                    focal_length = box_size * 2.0
                else:
                    focal_length = min(width, height)
            
            # 创建相机
            camera = PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=float(width) / height)
            camera_pose = np.array([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, focal_length/100.0],  # 缩放焦距
                [0.0, 0.0, 0.0, 1.0]
            ])
            scene.add(camera, pose=camera_pose)
            
            # 添加光源
            light = DirectionalLight(color=np.ones(3), intensity=3.0)
            scene.add(light, pose=np.eye(4))
            
            # 如果有姿势矩阵，使用它
            if pose_matrix is not None and isinstance(pose_matrix, np.ndarray):
                model_pose = pose_matrix
            else:
                model_pose = np.eye(4)
                
            # 添加模型到场景
            scene.add(Mesh.from_trimesh(body_mesh, smooth=True), pose=model_pose)
            scene.add(Mesh.from_trimesh(garment_mesh, smooth=True), pose=model_pose)
            
            # 创建渲染器
            renderer = OffscreenRenderer(width, height)
            color, depth = renderer.render(scene, flags=RenderFlags.RGBA)
            
            # 保存渲染结果 - 使用matplotlib确保正确保存透明度
            plt.figure(figsize=(width/100, height/100), dpi=100)
            plt.imshow(color)
            plt.axis('off')
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            plt.savefig(output_image, transparent=True, bbox_inches='tight', pad_inches=0)
            plt.close()
            
            # 释放资源
            renderer.delete()
            
            return True
        except Exception as e:
            print(f"渲染模型时出错: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def create_video_from_frames(self, rendered_frames, output_video=None, fps=30):
        """
        从渲染的帧创建视频
        
        参数:
        - rendered_frames: 渲染的帧信息
        - output_video: 输出视频路径
        - fps: 帧率
        
        返回:
        - 输出视频路径列表
        """
        if not rendered_frames:
            print("错误: 没有渲染的帧")
            return None
            
        videos = []
        for person_id, frames in rendered_frames.items():
            if not frames:
                continue
                
            # 设置输出视频路径
            if output_video:
                person_video = output_video.replace('.mp4', f'_person{person_id}.mp4')
            else:
                person_video = os.path.join(self.output_dir, f"render_person{person_id}.mp4")
                
            # 提取帧到一个目录
            frames_dir = os.path.dirname(frames[0][1])
            
            # 使用ffmpeg创建视频
            try:
                # 构建ffmpeg命令
                frames_pattern = os.path.join(frames_dir, "frame_%04d.png")
                cmd = f"ffmpeg -y -framerate {fps} -i \"{frames_pattern}\" -vf \"scale=trunc(iw/2)*2:trunc(ih/2)*2\" -c:v libx264 -pix_fmt yuva420p \"{person_video}\""
                
                print(f"为人物{person_id}创建视频...")
                print(f"执行命令: {cmd}")
                
                # 执行命令
                os.system(cmd)
                
                if os.path.exists(person_video):
                    print(f"成功创建视频: {person_video}")
                    videos.append(person_video)
                else:
                    print(f"视频创建失败: {person_video}")
            except Exception as e:
                print(f"创建视频时出错: {e}")
                
        return videos


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="HybrikAPI参数提取与渲染工具")
    
    parser.add_argument('--hybrik_result', type=str, required=True,
                       help="HybrikAPI结果文件路径")
    parser.add_argument('--models_dir', type=str, required=True,
                       help="包含模型的目录路径")
    parser.add_argument('--output_dir', type=str, default=None,
                       help="输出目录")
    parser.add_argument('--params_file', type=str, default=None,
                       help="参数输出文件路径")
    parser.add_argument('--start_frame', type=int, default=0,
                       help="起始帧")
    parser.add_argument('--end_frame', type=int, default=None,
                       help="结束帧")
    parser.add_argument('--person_ids', type=str, default=None,
                       help="要处理的人物ID，用逗号分隔(如\"0,1\")")
    parser.add_argument('--render_width', type=int, default=1024,
                       help="渲染宽度")
    parser.add_argument('--render_height', type=int, default=1024,
                       help="渲染高度")
    parser.add_argument('--no_flip', action='store_true',
                       help="不要垂直翻转模型(默认会翻转以修正方向)")
    parser.add_argument('--fps', type=int, default=30,
                       help="输出视频的帧率")
    parser.add_argument('--no_video', action='store_true',
                       help="不要创建视频，只渲染帧")
    
    args = parser.parse_args()
    
    # 解析人物ID
    person_ids = None
    if args.person_ids:
        person_ids = [int(pid.strip()) for pid in args.person_ids.split(',')]
    
    # 创建提取器
    extractor = HybrikParamsExtractor(output_dir=args.output_dir)
    
    # 加载HybrikAPI结果
    hybrik_data = extractor.load_hybrik_results(args.hybrik_result)
    if not hybrik_data:
        print("错误: 无法加载HybrikAPI结果")
        return 1
    
    # 提取参数
    params = extractor.extract_parameters(
        hybrik_data,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        person_ids=person_ids
    )
    
    if not params:
        print("错误: 无法提取参数")
        return 1
    
    # 保存参数
    params_file = extractor.save_parameters(params, args.params_file)
    if not params_file:
        print("警告: 无法保存参数文件")
    
    # 使用参数渲染模型
    print("\n使用提取的参数渲染模型...")
    rendered_frames = extractor.render_with_params(
        args.models_dir,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        person_ids=person_ids,
        render_size=(args.render_width, args.render_height),
        flip_vertical=not args.no_flip
    )
    
    if not rendered_frames:
        print("错误: 无法渲染模型")
        return 1
    
    # 创建视频
    if not args.no_video:
        videos = extractor.create_video_from_frames(rendered_frames, fps=args.fps)
        if videos:
            print("\n成功创建以下视频:")
            for video in videos:
                print(f"- {video}")
    
    print("\n处理完成!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
