import os
import sys
import numpy as np
import torch
import json
import requests
import time
from scipy.spatial.transform import Rotation as R
from psbody.mesh import Mesh
import cv2
from pyrender import Scene, Mesh, Viewer, OffscreenRenderer, RenderFlags
import trimesh
import matplotlib.pyplot as plt

# 导入TailorNet相关模块
from models.tailornet_model import get_best_runner as get_tn_runner
from models.smpl4garment import SMPL4Garment
from utils.rotation import normalize_y_rotation
from utils.interpenetration import remove_interpenetration_fast
from visualization.blender_renderer import visualize_garment_body

# 导入全局变量
import global_var

# 设置输出路径
OUT_PATH = global_var.OUTPUT_PATH

class HybrikTailorNetIntegration:
    """集成HybrIK API和TailorNet的类，用于生成穿衣服的3D人体模型"""
    
    def __init__(self, garment_class='t-shirt', gender='female', api_base_url="http://localhost:8000"):
        """
        初始化集成类
        
        参数:
        - garment_class: 服装类型，如't-shirt', 'shirt', 'pant', 'skirt'等
        - gender: 性别，'male'或'female'
        - api_base_url: HybrIK API的基础URL
        """
        self.garment_class = garment_class
        self.gender = gender
        self.api_base_url = api_base_url
        
        # 加载TailorNet模型
        print(f"加载TailorNet模型: {garment_class}_{gender}")
        self.tn_runner = get_tn_runner(gender=gender, garment_class=garment_class)
        self.smpl = SMPL4Garment(gender=gender)
        
        # 创建输出目录
        if not os.path.isdir(OUT_PATH):
            os.makedirs(OUT_PATH, exist_ok=True)
            
        # 设置默认服装风格参数
        self.default_gamma = self._get_default_style()
    
    def _get_default_style(self):
        """获取默认的服装风格参数"""
        if self.garment_class == 'old-t-shirt':
            # 老式T恤风格参数通常围绕[1.5, 0.5, 1.5, 0.0]
            return np.array([1.5, 0.5, 1.5, 0.0], dtype=np.float32)
        else:
            # 检查可用风格文件
            style_ids = ['000', '001', '002', '003', '004']
            for style_id in style_ids:
                style_path = os.path.join(global_var.DATA_DIR, 
                    f"{self.garment_class}_{self.gender}/style/gamma_{style_id}.npy")
                if os.path.exists(style_path):
                    try:
                        return np.load(style_path).astype(np.float32)
                    except:
                        continue
            
            # 如果没有找到风格文件，生成随机gamma（假设4维）
            print(f"未找到{self.garment_class}_{self.gender}的风格文件，生成随机参数...")
            return np.random.randn(4).astype(np.float32)
    
    def _matrix_to_axis_angle(self, rotation_data):
        """
        将旋转数据转换为轴角表示
        
        参数:
        - rotation_data: 旋转数据，可能是以下格式之一：
          - (N, 3, 3) 的旋转矩阵
          - (24*3*3,) = (216,) 的展平旋转矩阵（24个关节，每个3x3）
          - (55*3*3,) = (495,) 的SMPLX格式展平旋转矩阵
          - (K,) 的其他长度数组
        
        返回:
        - axis_angle: 形状为(N*3,) 或 (72,)的轴角表示（24个关节，每个3维）
        """
        try:
            # 打印输入数据的形状，帮助调试
            rotation_shape = np.array(rotation_data).shape
            print(f"旋转数据形状: {rotation_shape}")
            
            # 检查是否是展平的旋转矩阵
            if len(rotation_shape) == 1:
                total_elements = rotation_shape[0]
                
                # 对于标准SMPL的24个关节，展平的旋转矩阵应该有216或218个元素
                if total_elements >= 216 and total_elements <= 220:
                    # 确保我们有24*3*3=216个元素
                    if total_elements > 216:
                        print(f"警告：旋转数据包含额外的{total_elements - 216}个元素，将取最后216个元素")
                        rotation_data = rotation_data[-216:]
                    
                    # 重新整形为(24, 3, 3)
                    rotation_matrices = np.array(rotation_data).reshape(24, 3, 3)
                    
                    # 为每个关节计算轴角
                    axis_angles = []
                    for joint_rot in rotation_matrices:
                        rot = R.from_matrix(joint_rot)
                        axis_angles.append(rot.as_rotvec())
                    
                    # 将结果展平为72维向量（24个关节，每个3维）
                    return np.array(axis_angles).flatten()
                
                # 对于SMPLX格式，可能有55个关节，展平的旋转矩阵有495个元素
                elif total_elements >= 495 and total_elements <= 500:
                    print(f"检测到SMPLX格式的旋转数据，包含{total_elements}个元素")
                    try:
                        # 尝试将其解释为55个关节的旋转矩阵
                        # 取前24个关节的部分（SMPL格式）
                        rotation_data_smpl = rotation_data[:216]  # 24*3*3 = 216
                        rotation_matrices = np.array(rotation_data_smpl).reshape(24, 3, 3)
                        
                        # 为每个关节计算轴角
                        axis_angles = []
                        for joint_rot in rotation_matrices:
                            rot = R.from_matrix(joint_rot)
                            axis_angles.append(rot.as_rotvec())
                        
                        # 将结果展平为72维向量（24个关节，每个3维）
                        return np.array(axis_angles).flatten()
                    except Exception as e:
                        print(f"处理SMPLX格式旋转数据时出错: {e}")
                
                # 如果不是216或495个元素，可能是其他类型的旋转表示
                else:
                    print(f"警告：未知的旋转数据格式，包含{total_elements}个元素")
                    
                    # 作为备选方案，返回标准姿势（零旋转）
                    print("返回标准姿势（零旋转）")
                    return np.zeros(72)  # 24个关节，每个3维
            
            else:
                # 标准的旋转矩阵形式(N, 3, 3)
                rot = R.from_matrix(rotation_data)
                return rot.as_rotvec()
                
        except Exception as e:
            print(f"转换旋转矩阵时出错: {e}")
            print("返回标准姿势（零旋转）")
            return np.zeros(72)  # 返回标准姿势（零旋转）
    
    def process_hybrik_output(self, hybrik_result):
        """
        处理HybrIK API的输出结果，提取SMPL参数
        
        参数:
        - hybrik_result: HybrIK API返回的JSON结果，可能是列表或字典
        
        返回:
        - processed_frames: 处理后的帧列表，每帧包含所有检测人物的SMPL参数
        """
        processed_frames = []
        
        # 检查结果格式
        if isinstance(hybrik_result, list):
            # 如果结果是列表，直接处理每一帧
            frame_data_list = hybrik_result
        elif isinstance(hybrik_result, dict) and 'results' in hybrik_result:
            # 如果结果是字典且包含'results'键，取出结果列表
            frame_data_list = hybrik_result.get('results', [])
        else:
            print(f"警告：未知的结果格式: {type(hybrik_result)}")
            print(f"结果内容: {str(hybrik_result)[:500]}...")
            return []
        
        # 处理每一帧数据
        for frame_idx, frame_data in enumerate(frame_data_list):
            # 如果frame_data是字典并且包含'results'，说明是单帧数据
            if isinstance(frame_data, dict) and 'results' in frame_data:
                frame_idx = frame_data.get('frame_idx', frame_idx)
                person_list = frame_data.get('results', [])
            else:
                # 否则，假设frame_data本身就是人物列表或单个人物的数据
                if isinstance(frame_data, list):
                    person_list = frame_data
                else:
                    person_list = [frame_data]
            
            # 打印当前帧的数据结构
            if frame_idx < 3:  # 仅打印前几帧以避免过多输出
                print(f"帧 {frame_idx} 数据类型: {type(frame_data)}")
                if isinstance(frame_data, dict):
                    print(f"帧 {frame_idx} 包含的键: {list(frame_data.keys())}")
                
                # 如果有track_id，pred_betas或pred_thetas_mat，那么frame_data可能是人物数据
                if isinstance(frame_data, dict) and ('track_id' in frame_data or 'pred_betas' in frame_data or 'pred_thetas_mat' in frame_data):
                    print(f"帧 {frame_idx} 可能是人物数据而不是帧数据")
                    person_list = [frame_data]  # 将整个帧作为一个人物处理
            
            frame_results = []
            
            for person_idx, person in enumerate(person_list):
                if not isinstance(person, dict):
                    print(f"警告：跳过非字典类型的人物数据: {person}")
                    continue
                
                # 只打印前几个人的数据结构
                if frame_idx < 2 and person_idx < 2:
                    print(f"帧 {frame_idx} 人物 {person_idx} 包含的键: {list(person.keys())}")
                
                track_id = person.get('track_id', 0)
                
                # 提取SMPL参数
                pred_betas = person.get('pred_betas', [])
                pred_thetas_mat = person.get('pred_thetas_mat', [])
                
                # 确保数据不为空
                if not pred_betas or not pred_thetas_mat:
                    print(f"警告：帧 {frame_idx} 人物 {person_idx} 缺少必要的SMPL参数，跳过此人物")
                    continue
                
                try:
                    pred_betas = np.array(pred_betas, dtype=np.float32)
                    pred_thetas_mat = np.array(pred_thetas_mat, dtype=np.float32)
                    
                    # 将旋转矩阵转换为轴角表示
                    pred_thetas = self._matrix_to_axis_angle(pred_thetas_mat)
                    
                    # 检查轴角的形状
                    if len(pred_thetas) != 72:
                        print(f"警告：轴角的长度为 {len(pred_thetas)}，而不是预期的72，将调整大小")
                        # 确保有72个元素
                        if len(pred_thetas) < 72:
                            # 如果少于72个元素，填充零
                            pred_thetas = np.pad(pred_thetas, (0, 72 - len(pred_thetas)), 'constant')
                        else:
                            # 如果多于72个元素，截断
                            pred_thetas = pred_thetas[:72]
                    
                    # 创建处理后的人物数据
                    person_data = {
                        'track_id': track_id,
                        'betas': pred_betas,
                        'thetas': pred_thetas,
                        'transl': np.array(person.get('transl', [0, 0, 0]), dtype=np.float32)
                    }
                    
                    frame_results.append(person_data)
                    
                except Exception as e:
                    print(f"处理帧 {frame_idx} 人物 {person_idx} 时出错: {e}")
                    continue
            
            processed_frames.append({
                'frame_idx': frame_idx,
                'people': frame_results
            })
        
        total_people = sum(len(frame['people']) for frame in processed_frames)
        print(f"处理了 {len(processed_frames)} 帧数据，包含 {total_people} 个人物实例")
        
        # 如果没有有效的人物数据，警告用户
        if total_people == 0:
            print("警告：处理后没有找到有效的人物数据！")
        
        return processed_frames
    
    def generate_clothed_model(self, thetas, betas, gamma=None, output_prefix=None):
        """
        为单个人物生成带衣服的3D模型
        
        参数:
        - thetas: SMPL姿势参数，形状为(72,)
        - betas: SMPL形状参数，形状为(10,)
        - gamma: 服装风格参数，如果为None则使用默认参数
        - output_prefix: 输出文件前缀，如果为None则自动生成
        
        返回:
        - body_path: 生成的身体模型路径
        - garment_path: 生成的服装模型路径
        - image_path: 如果渲染了图像，返回图像路径，否则为None
        """
        if gamma is None:
            gamma = self.default_gamma
        
        if output_prefix is None:
            output_prefix = f"{self.garment_class}_{self.gender}_{int(time.time())}"
        
        # 确保betas是正确的形状（10维）
        if len(betas) > 10:
            print(f"警告：betas维度({len(betas)})超过10，将截断到前10维")
            betas = betas[:10]
        elif len(betas) < 10:
            print(f"警告：betas维度({len(betas)})小于10，将填充零")
            betas = np.pad(betas, (0, 10 - len(betas)), 'constant')
        
        # 确保gamma是正确的形状（通常是4维）
        expected_gamma_dim = 4  # TailorNet中通常使用4维的gamma
        if len(gamma) > expected_gamma_dim:
            print(f"警告：gamma维度({len(gamma)})超过{expected_gamma_dim}，将截断")
            gamma = gamma[:expected_gamma_dim]
        elif len(gamma) < expected_gamma_dim:
            print(f"警告：gamma维度({len(gamma)})小于{expected_gamma_dim}，将填充零")
            gamma = np.pad(gamma, (0, expected_gamma_dim - len(gamma)), 'constant')
        
        # 规范化Y旋转，使其面向前方
        theta_normalized = normalize_y_rotation(thetas)
        
        # 打印调试信息
        print(f"betas 形状: {betas.shape}, thetas 形状: {theta_normalized.shape}, gamma 形状: {gamma.shape}")
        
        # 使用TailorNet预测服装顶点位移
        try:
            with torch.no_grad():
                pred_verts_d = self.tn_runner.forward(
                    thetas=torch.from_numpy(theta_normalized[None, :].astype(np.float32)).cuda(),
                    betas=torch.from_numpy(betas[None, :].astype(np.float32)).cuda(),
                    gammas=torch.from_numpy(gamma[None, :].astype(np.float32)).cuda(),
                )[0].cpu().numpy()
            
            # 从预测的顶点位移获取服装
            body, pred_gar = self.smpl.run(beta=betas, theta=thetas, garment_class=self.garment_class, garment_d=pred_verts_d)
            pred_gar = remove_interpenetration_fast(pred_gar, body)
            
            # 保存身体和预测的服装
            body_path = os.path.join(OUT_PATH, f"body_{output_prefix}.ply")
            garment_path = os.path.join(OUT_PATH, f"pred_gar_{output_prefix}.ply")
            
            body.write_ply(body_path)
            pred_gar.write_ply(garment_path)
            
            print(f"保存了身体模型到: {body_path}")
            print(f"保存了服装模型到: {garment_path}")
            
            return body_path, garment_path, None
            
        except Exception as e:
            print(f"生成服装模型时出错: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None
    
    def render_model(self, body_path, garment_path, output_path=None):
        """
        渲染衣服和身体模型
        
        参数:
        - body_path: 身体模型路径
        - garment_path: 服装模型路径
        - output_path: 输出图像路径，如果为None则自动生成
        
        返回:
        - output_path: 渲染的图像路径
        """
        if output_path is None:
            base_name = os.path.basename(body_path).replace("body_", "")
            output_path = os.path.join(OUT_PATH, f"img_{base_name.replace('.ply', '.png')}")
        
        body = Mesh(filename=body_path)
        pred_gar = Mesh(filename=garment_path)
        
        visualize_garment_body(
            pred_gar, body, output_path,
            garment_class=self.garment_class, side='front')
        
        print(f"渲染了图像到: {output_path}")
        
        return output_path
    
    def process_video(self, video_path, threshold=0.7, video_type="general", yolo_path=None, skip_frames=1):
        """
        处理视频并生成一系列带衣服的3D模型
        
        参数:
        - video_path: 视频文件路径
        - threshold: 人体检测阈值
        - video_type: 视频类型，可选"general"、"badminton"或"dance"
        - yolo_path: YOLO模型路径，用于提高检测精度
        - skip_frames: 跳帧设置，默认为1（处理所有帧）
        
        返回:
        - output_dir: 包含所有生成模型的目录
        """
        print(f"处理视频: {video_path}")
        
        # 调用HybrIK API处理视频
        api_endpoint = f"{self.api_base_url}/predict/video_path"
        
        files = {
            'video_path': (None, video_path),
            'threshold': (None, str(threshold)),
            'video_type': (None, video_type),
            'skip_frames': (None, str(skip_frames)),
            'save_frames': (None, 'true'),
            'save_video': (None, 'true')
        }
        
        # 如果提供了YOLO模型路径，添加到请求中
        if yolo_path:
            files['yolo_path'] = (None, yolo_path)
            print(f"使用自定义YOLO模型: {yolo_path}")
        
        try:
            response = requests.post(api_endpoint, files=files)
            
            if response.status_code != 200:
                print(f"API请求失败: {response.status_code}")
                print(response.text)
                return None
            
            # 获取任务ID
            response_json = response.json()
            if not isinstance(response_json, dict) or 'task_id' not in response_json:
                print(f"API响应格式错误，未找到task_id:")
                print(f"响应内容: {response.text[:500]}...")
                return None
            
            task_id = response_json['task_id']
            print(f"任务ID: {task_id}")
            
            # 等待处理完成
            status_endpoint = f"{self.api_base_url}/predict/video/status/{task_id}"
            
            while True:
                try:
                    status_response = requests.get(status_endpoint)
                    status_data = status_response.json()
                    
                    if status_data.get('status') == 'completed':
                        print("视频处理完成")
                        break
                    elif status_data.get('status') == 'failed':
                        print(f"视频处理失败: {status_data.get('message')}")
                        return None
                    
                    print(f"处理进度: {status_data.get('progress', 0):.2f}%")
                    time.sleep(2)
                except Exception as e:
                    print(f"检查状态时出错: {e}")
                    time.sleep(5)  # 遇到错误时等待更长时间
            
            # 获取结果
            result_endpoint = f"{self.api_base_url}/predict/video/result/{task_id}"
            try:
                result_response = requests.get(result_endpoint)
                hybrik_result = result_response.json()
                
                # 打印结果格式信息，帮助调试
                if isinstance(hybrik_result, list):
                    print(f"API返回了列表格式的结果，包含 {len(hybrik_result)} 个元素")
                    if hybrik_result and len(hybrik_result) > 0:
                        print(f"第一个元素类型: {type(hybrik_result[0])}")
                elif isinstance(hybrik_result, dict):
                    print(f"API返回了字典格式的结果，包含以下键: {list(hybrik_result.keys())}")
                else:
                    print(f"API返回了意外的结果类型: {type(hybrik_result)}")
                
                # 使用辅助函数打印API结果的详细结构
                print_api_result_structure(hybrik_result)
                
                # 如果需要，可以将结果保存到文件，以便后续分析
                try:
                    result_file = os.path.join(OUT_PATH, f"hybrik_result_{task_id}.json")
                    with open(result_file, 'w') as f:
                        json.dump(hybrik_result, f)
                    print(f"保存了API结果到文件: {result_file}")
                except Exception as e:
                    print(f"保存API结果到文件时出错: {e}")
                
            except Exception as e:
                print(f"获取结果时出错: {e}")
                return None
            
            # 处理HybrIK输出
            processed_frames = self.process_hybrik_output(hybrik_result)
            
            if not processed_frames:
                print("处理后没有有效的帧数据，无法生成模型")
                return None
            
            # 为每一帧的每个人生成带衣服的模型
            output_dir = os.path.join(OUT_PATH, f"video_{task_id}")
            os.makedirs(output_dir, exist_ok=True)
            
            body_paths = []
            garment_paths = []
            
            for frame in processed_frames:
                frame_idx = frame['frame_idx']
                
                if not frame['people']:
                    print(f"帧 {frame_idx} 中没有检测到人物，跳过")
                    continue
                
                print(f"处理帧 {frame_idx}，检测到 {len(frame['people'])} 个人物")
                
                for person_idx, person in enumerate(frame['people']):
                    output_prefix = f"{self.garment_class}_{self.gender}_frame{frame_idx:04d}_person{person_idx:02d}"
                    
                    body_path, garment_path, _ = self.generate_clothed_model(
                        thetas=person['thetas'],
                        betas=person['betas'],
                        output_prefix=output_prefix
                    )
                    
                    body_paths.append(body_path)
                    garment_paths.append(garment_path)
            
            return output_dir
            
        except requests.exceptions.ConnectionError:
            print(f"无法连接到API服务器: {self.api_base_url}")
            print("请确保HybrIK API服务正在运行，并且URL正确")
            return None
        except Exception as e:
            print(f"处理视频时出错: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def process_image(self, image_path, threshold=0.7, yolo_path=None):
        """
        处理单张图像并生成带衣服的3D模型
        
        参数:
        - image_path: 图像文件路径
        - threshold: 人体检测阈值
        - yolo_path: YOLO模型路径，用于提高检测精度
        
        返回:
        - body_paths: 生成的身体模型路径列表
        - garment_paths: 生成的服装模型路径列表
        - image_paths: 渲染的图像路径列表
        """
        print(f"处理图像: {image_path}")
        
        # 调用HybrIK API处理图像
        api_endpoint = f"{self.api_base_url}/predict/path"
        
        files = {
            'image_path': (None, image_path),
            'threshold': (None, str(threshold))
        }
        
        # 如果提供了YOLO模型路径，添加到请求中
        if yolo_path:
            files['yolo_path'] = (None, yolo_path)
            print(f"使用自定义YOLO模型: {yolo_path}")
        
        try:
            response = requests.post(api_endpoint, files=files)
            
            if response.status_code != 200:
                print(f"API请求失败: {response.status_code}")
                print(response.text)
                return None, None, None
            
            # 获取结果
            hybrik_result = response.json()
            
            # 打印结果格式信息，帮助调试
            if isinstance(hybrik_result, list):
                print(f"API返回了列表格式的结果，包含 {len(hybrik_result)} 个元素")
                if hybrik_result and len(hybrik_result) > 0:
                    print(f"第一个元素类型: {type(hybrik_result[0])}")
            elif isinstance(hybrik_result, dict):
                print(f"API返回了字典格式的结果，包含以下键: {list(hybrik_result.keys())}")
            else:
                print(f"API返回了意外的结果类型: {type(hybrik_result)}")
            
            # 使用辅助函数打印API结果的详细结构
            print_api_result_structure(hybrik_result)
            
            # 如果需要，可以将结果保存到文件，以便后续分析
            try:
                result_file = os.path.join(OUT_PATH, f"hybrik_result_image_{os.path.basename(image_path)}.json")
                with open(result_file, 'w') as f:
                    json.dump(hybrik_result, f)
                print(f"保存了API结果到文件: {result_file}")
            except Exception as e:
                print(f"保存API结果到文件时出错: {e}")
            
            # 根据API返回结果的格式，调整处理方式
            if isinstance(hybrik_result, dict) and 'results' in hybrik_result:
                # 创建假帧结构以兼容处理函数
                mock_frame = {
                    'frame_idx': 0,
                    'results': hybrik_result.get('results', [])
                }
                processed_frames = self.process_hybrik_output({'results': [mock_frame]})
            else:
                # 直接传递结果给处理函数
                processed_frames = self.process_hybrik_output(hybrik_result)
            
            if not processed_frames:
                print("处理后没有有效的帧数据，无法生成模型")
                return None, None, None
            
            # 为图像中的每个人生成带衣服的模型
            body_paths = []
            garment_paths = []
            image_paths = []
            
            for frame in processed_frames:
                if not frame['people']:
                    print(f"帧 {frame['frame_idx']} 中没有检测到人物，跳过")
                    continue
                
                print(f"处理帧 {frame['frame_idx']}，检测到 {len(frame['people'])} 个人物")
                
                for person_idx, person in enumerate(frame['people']):
                    output_prefix = f"{self.garment_class}_{self.gender}_image_person{person_idx:02d}"
                    
                    body_path, garment_path, _ = self.generate_clothed_model(
                        thetas=person['thetas'],
                        betas=person['betas'],
                        output_prefix=output_prefix
                    )
                    
                    # 渲染图像
                    image_path = self.render_model(body_path, garment_path)
                    
                    body_paths.append(body_path)
                    garment_paths.append(garment_path)
                    image_paths.append(image_path)
            
            return body_paths, garment_paths, image_paths
            
        except requests.exceptions.ConnectionError:
            print(f"无法连接到API服务器: {self.api_base_url}")
            print("请确保HybrIK API服务正在运行，并且URL正确")
            return None, None, None
        except Exception as e:
            print(f"处理图像时出错: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None


def test_with_dummy_data(garment_class='t-shirt', gender='male', pose_type='stand'):
    """
    使用预定义数据测试TailorNet功能，不需要HybrIK API
    
    参数:
    - garment_class: 服装类型，如't-shirt', 'shirt', 'pant', 'skirt'等
    - gender: 性别，'male'或'female'
    - pose_type: 姿势类型，'stand'(站立), 'a_pose'(A姿势), 't_pose'(T姿势), 'sit'(坐), 'walk'(走)
    """
    print(f"使用预定义数据测试TailorNet: {garment_class}_{gender}, 姿势: {pose_type}")
    
    # 创建集成对象
    integration = HybrikTailorNetIntegration(
        garment_class=garment_class,
        gender=gender,
    )
    
    # 创建不同的姿势
    if pose_type == 'stand':
        # 标准站立姿势
        dummy_thetas = np.zeros(72)
    elif pose_type == 'a_pose':
        # A姿势 (手臂稍微张开)
        dummy_thetas = np.zeros(72)
        # 左肩、右肩外展（13,14,16,17是肩/肘关节）
        dummy_thetas[13*3+1] = 0.4  # 左肩外展
        dummy_thetas[14*3+1] = 0.2  # 左肘弯曲
        dummy_thetas[16*3+1] = -0.4  # 右肩外展
        dummy_thetas[17*3+1] = -0.2  # 右肘弯曲
    elif pose_type == 't_pose':
        # T姿势 (手臂水平张开)
        dummy_thetas = np.zeros(72)
        dummy_thetas[13*3+1] = 1.5  # 左肩外展
        dummy_thetas[16*3+1] = -1.5  # 右肩外展
    elif pose_type == 'sit':
        # 坐姿 (膝盖弯曲)
        dummy_thetas = np.zeros(72)
        dummy_thetas[1*3+0] = 0.4  # 身体略微前倾
        dummy_thetas[4*3+0] = 1.0  # 左膝弯曲
        dummy_thetas[5*3+0] = -0.5  # 左踝调整
        dummy_thetas[7*3+0] = 1.0  # 右膝弯曲
        dummy_thetas[8*3+0] = -0.5  # 右踝调整
    elif pose_type == 'walk':
        # 走路姿势
        dummy_thetas = np.zeros(72)
        dummy_thetas[1*3+0] = 0.2  # 身体略微前倾
        dummy_thetas[4*3+0] = 0.3  # 左膝微弯
        dummy_thetas[7*3+0] = 0.6  # 右膝弯曲
        dummy_thetas[13*3+2] = 0.4  # 左臂摆动
        dummy_thetas[16*3+2] = -0.4  # 右臂摆动
    else:
        # 默认标准姿势
        dummy_thetas = np.zeros(72)
    
    # 创建不同的体型
    # 平均体型
    dummy_betas = np.zeros(10)
    
    # 生成服装模型
    body_path, garment_path, _ = integration.generate_clothed_model(
        thetas=dummy_thetas,
        betas=dummy_betas,
        output_prefix=f"{garment_class}_{gender}_test_{pose_type}"
    )
    
    if body_path and garment_path:
        # 渲染结果
        image_path = integration.render_model(body_path, garment_path)
        
        print(f"生成了测试模型:")
        print(f"  身体模型: {body_path}")
        print(f"  服装模型: {garment_path}")
        print(f"  渲染图像: {image_path}")
        
        return body_path, garment_path, image_path
    else:
        print("测试模型生成失败")
        return None, None, None


def print_api_result_structure(api_result, max_depth=2):
    """
    打印API返回结果的数据结构，帮助调试
    
    参数:
    - api_result: API返回的结果
    - max_depth: 最大递归深度
    """
    def _print_structure(obj, prefix="", depth=0):
        if depth > max_depth:
            return
        
        if isinstance(obj, dict):
            print(f"{prefix}字典 包含 {len(obj)} 个键: {list(obj.keys())}")
            if depth < max_depth:
                for k, v in list(obj.items())[:3]:  # 只显示前3个键
                    print(f"{prefix}  - 键 '{k}':")
                    _print_structure(v, prefix + "    ", depth + 1)
                if len(obj) > 3:
                    print(f"{prefix}  ... 还有 {len(obj) - 3} 个键")
        
        elif isinstance(obj, list):
            print(f"{prefix}列表 长度为 {len(obj)}")
            if depth < max_depth and obj:
                # 只显示第一个元素
                print(f"{prefix}  - 第一个元素:")
                _print_structure(obj[0], prefix + "    ", depth + 1)
                if len(obj) > 1:
                    print(f"{prefix}  ... 还有 {len(obj) - 1} 个元素")
        
        elif isinstance(obj, np.ndarray):
            print(f"{prefix}NumPy数组 形状: {obj.shape}, 类型: {obj.dtype}")
        
        else:
            print(f"{prefix}值 类型: {type(obj)}")
            # 尝试打印值的一部分
            try:
                value_str = str(obj)
                if len(value_str) > 100:
                    value_str = value_str[:100] + "..."
                print(f"{prefix}  值: {value_str}")
            except:
                print(f"{prefix}  值: <无法打印>")
    
    print("API结果结构:")
    _print_structure(api_result)


def load_hybrik_result_from_file(file_path):
    """从文件中加载HybrIK API的结果
    
    参数:
    - file_path: 保存的JSON文件路径
    
    返回:
    - hybrik_result: 加载的结果
    """
    try:
        with open(file_path, 'r') as f:
            hybrik_result = json.load(f)
        print(f"从文件加载了HybrIK结果: {file_path}")
        return hybrik_result
    except Exception as e:
        print(f"加载文件时出错: {e}")
        return None


if __name__ == "__main__":
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description="HybrIK到TailorNet的集成工具")
    
    parser.add_argument('--api_url', type=str, default="http://localhost:8000",
                        help="HybrIK API的基础URL")
    parser.add_argument('--garment_class', type=str, default="t-shirt",
                        help="服装类型: t-shirt, shirt, pant, short-pant, skirt等")
    parser.add_argument('--gender', type=str, default="female",
                        help="性别: male或female")
    parser.add_argument('--yolo_path', type=str, 
                        help="YOLO模型路径，用于提高检测精度")
    
    subparsers = parser.add_subparsers(dest='command', help='子命令')
    
    # 图像处理
    image_parser = subparsers.add_parser('image', help='处理单张图像')
    image_parser.add_argument('image_path', type=str, help='图像文件路径')
    image_parser.add_argument('--threshold', type=float, default=0.7, help='人体检测阈值')
    
    # 视频处理
    video_parser = subparsers.add_parser('video', help='处理视频')
    video_parser.add_argument('video_path', type=str, help='视频文件路径')
    video_parser.add_argument('--threshold', type=float, default=0.7, help='人体检测阈值')
    video_parser.add_argument('--video_type', type=str, default="general",
                             help='视频类型: general, badminton, dance')
    video_parser.add_argument('--skip_frames', type=int, default=1,
                             help='跳帧设置，默认为1（处理所有帧）')
    
    # 测试命令
    test_parser = subparsers.add_parser('test', help='使用预定义数据测试TailorNet功能')
    test_parser.add_argument('--pose_type', type=str, default='stand',
                          help='姿势类型: stand(站立), a_pose(A姿势), t_pose(T姿势), sit(坐), walk(走)')
    
    # 新增: 从文件加载HybrIK结果
    load_parser = subparsers.add_parser('load', help='从本地文件加载HybrIK结果并生成模型')
    load_parser.add_argument('result_file', type=str, help='HybrIK结果JSON文件路径')
    load_parser.add_argument('--process_frames', type=int, default=0, 
                            help='处理的帧数，0表示处理所有帧')
    load_parser.add_argument('--start_frame', type=int, default=0,
                            help='开始处理的帧索引')
    
    # 新增: 直接使用之前生成的任务ID
    task_parser = subparsers.add_parser('task', help='使用已有的任务ID获取结果')
    task_parser.add_argument('task_id', type=str, help='已存在的HybrIK任务ID')
    
    args = parser.parse_args()
    
    # 根据命令执行相应功能
    if args.command == 'test':
        test_with_dummy_data(garment_class=args.garment_class, gender=args.gender, pose_type=args.pose_type)
        
    elif args.command == 'load':
        # 从文件加载HybrIK结果
        hybrik_result = load_hybrik_result_from_file(args.result_file)
        if hybrik_result:
            # 创建集成对象
            integration = HybrikTailorNetIntegration(
                garment_class=args.garment_class,
                gender=args.gender
            )
            
            # 使用辅助函数打印API结果的详细结构
            print_api_result_structure(hybrik_result)
            
            # 处理HybrIK输出
            processed_frames = integration.process_hybrik_output(hybrik_result)
            
            if not processed_frames:
                print("处理后没有有效的帧数据，无法生成模型")
                sys.exit(1)
            
            # 限制处理的帧数
            if args.process_frames > 0:
                end_frame = args.start_frame + args.process_frames
                processed_frames = processed_frames[args.start_frame:end_frame]
                print(f"将只处理从 {args.start_frame} 开始的 {len(processed_frames)} 帧")
            
            # 为每一帧的每个人生成带衣服的模型
            output_dir = os.path.join(OUT_PATH, f"loaded_result_{int(time.time())}")
            os.makedirs(output_dir, exist_ok=True)
            
            success_count = 0
            fail_count = 0
            
            for frame in processed_frames:
                frame_idx = frame['frame_idx']
                
                if not frame['people']:
                    print(f"帧 {frame_idx} 中没有检测到人物，跳过")
                    continue
                
                print(f"处理帧 {frame_idx}，检测到 {len(frame['people'])} 个人物")
                
                for person_idx, person in enumerate(frame['people']):
                    try:
                        output_prefix = f"{args.garment_class}_{args.gender}_frame{frame_idx:04d}_person{person_idx:02d}"
                        
                        body_path, garment_path, _ = integration.generate_clothed_model(
                            thetas=person['thetas'],
                            betas=person['betas'],
                            output_prefix=output_prefix
                        )
                        
                        if body_path and garment_path:
                            success_count += 1
                        else:
                            fail_count += 1
                            
                    except Exception as e:
                        print(f"处理帧 {frame_idx} 人物 {person_idx} 时出错: {e}")
                        fail_count += 1
                        continue
            
            print(f"处理完成: 成功 {success_count} 个模型, 失败 {fail_count} 个模型")
            print(f"输出保存在: {output_dir}")
        else:
            print("无法加载文件，请检查文件路径是否正确")
    
    elif args.command == 'task':
        # 直接使用已有的任务ID
        # 创建集成对象
        integration = HybrikTailorNetIntegration(
            garment_class=args.garment_class,
            gender=args.gender,
            api_base_url=args.api_url
        )
        
        # 获取结果
        task_id = args.task_id
        result_endpoint = f"{integration.api_base_url}/predict/video/result/{task_id}"
        
        try:
            print(f"使用任务ID获取结果: {task_id}")
            result_response = requests.get(result_endpoint)
            hybrik_result = result_response.json()
            
            # 保存API结果到文件
            result_file = os.path.join(OUT_PATH, f"hybrik_result_{task_id}.json")
            with open(result_file, 'w') as f:
                json.dump(hybrik_result, f)
            print(f"保存了API结果到文件: {result_file}")
            
            # 处理HybrIK输出
            processed_frames = integration.process_hybrik_output(hybrik_result)
            
            if not processed_frames:
                print("处理后没有有效的帧数据，无法生成模型")
                sys.exit(1)
                
            # 为每一帧的每个人生成带衣服的模型
            output_dir = os.path.join(OUT_PATH, f"task_{task_id}")
            os.makedirs(output_dir, exist_ok=True)
            
            success_count = 0
            fail_count = 0
            
            # 只处理前5帧作为快速测试
            test_frames = processed_frames[:5]
            print(f"为了快速测试，只处理前 {len(test_frames)} 帧")
            
            for frame in test_frames:
                frame_idx = frame['frame_idx']
                
                if not frame['people']:
                    print(f"帧 {frame_idx} 中没有检测到人物，跳过")
                    continue
                
                print(f"处理帧 {frame_idx}，检测到 {len(frame['people'])} 个人物")
                
                for person_idx, person in enumerate(frame['people']):
                    try:
                        output_prefix = f"{args.garment_class}_{args.gender}_frame{frame_idx:04d}_person{person_idx:02d}"
                        
                        body_path, garment_path, _ = integration.generate_clothed_model(
                            thetas=person['thetas'],
                            betas=person['betas'],
                            output_prefix=output_prefix
                        )
                        
                        if body_path and garment_path:
                            success_count += 1
                            # 渲染模型以便查看
                            image_path = integration.render_model(body_path, garment_path)
                            print(f"渲染了图像: {image_path}")
                        else:
                            fail_count += 1
                            
                    except Exception as e:
                        print(f"处理帧 {frame_idx} 人物 {person_idx} 时出错: {e}")
                        fail_count += 1
                        continue
            
            print(f"\n====== 处理摘要 ======")
            print(f"任务ID: {task_id}")
            print(f"服装类型: {args.garment_class}_{args.gender}")
            print(f"处理了 {len(test_frames)} 帧, 共 {success_count + fail_count} 个人物实例")
            print(f"成功: {success_count} 个模型")
            print(f"失败: {fail_count} 个模型")
            print(f"输出保存在: {output_dir}")
            print(f"如果测试成功，你可以使用load命令处理更多帧:")
            print(f"python hybrik_to_tailornet.py --garment_class {args.garment_class} --gender {args.gender} load {result_file}")
            
        except Exception as e:
            print(f"获取或处理任务结果时出错: {e}")
            import traceback
            traceback.print_exc()
    
    elif args.command == 'image':
        # 创建集成对象
        integration = HybrikTailorNetIntegration(
            garment_class=args.garment_class,
            gender=args.gender,
            api_base_url=args.api_url
        )
        
        body_paths, garment_paths, image_paths = integration.process_image(
            args.image_path, threshold=args.threshold, yolo_path=args.yolo_path
        )
        
        if body_paths:
            print(f"成功处理图像并生成了{len(body_paths)}个服装模型")
            for i, (body, garment, image) in enumerate(zip(body_paths, garment_paths, image_paths)):
                print(f"人物 {i+1}:")
                print(f"  身体模型: {body}")
                print(f"  服装模型: {garment}")
                print(f"  渲染图像: {image}")
        else:
            print("处理图像失败或未检测到人体")
    
    elif args.command == 'video':
        # 创建集成对象
        integration = HybrikTailorNetIntegration(
            garment_class=args.garment_class,
            gender=args.gender,
            api_base_url=args.api_url
        )
        
        output_dir = integration.process_video(
            args.video_path, 
            threshold=args.threshold, 
            video_type=args.video_type,
            yolo_path=args.yolo_path,
            skip_frames=args.skip_frames
        )
        
        if output_dir:
            print(f"成功处理视频，输出保存在: {output_dir}")
        else:
            print("处理视频失败")
    
    else:
        parser.print_help() 