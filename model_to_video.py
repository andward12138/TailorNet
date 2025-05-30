import os
import sys
import numpy as np
import cv2
import json
import glob
import re
import time
import trimesh
from pyrender import Scene, Mesh, Viewer, OffscreenRenderer, RenderFlags, DirectionalLight, SpotLight, PointLight, PerspectiveCamera
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image, ImageOps

# 导入全局变量
try:
    import global_var
    OUT_PATH = global_var.OUTPUT_PATH
except ImportError:
    print("警告: 未找到global_var模块，将使用当前目录作为输出路径")
    OUT_PATH = os.path.join(os.getcwd(), "output")
    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH, exist_ok=True)

class ModelRenderer:
    """从3D模型生成视频的渲染器类"""
    
    def __init__(self, output_dir=None):
        """
        初始化渲染器
        
        参数:
        - output_dir: 输出目录，默认使用OUT_PATH
        """
        self.output_dir = output_dir or OUT_PATH
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
            
        # 设置默认渲染参数
        self.default_render_size = (1024, 1024)
        self.default_bg_color = (1.0, 1.0, 1.0, 1.0)  # 白色背景
    
    def _setup_basic_scene(self, width=1024, height=1024, bg_color=None):
        """
        设置基本渲染场景
        
        参数:
        - width: 渲染宽度
        - height: 渲染高度
        - bg_color: 背景颜色，默认白色
        
        返回:
        - scene: 渲染场景
        - camera: 相机对象
        - renderer: 渲染器
        """
        if bg_color is None:
            bg_color = self.default_bg_color
            
        # 创建场景
        scene = Scene(bg_color=bg_color)
        
        # 添加相机
        camera = PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=float(width) / height)
        scene.add(camera, pose=np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 2.5],
            [0.0, 0.0, 0.0, 1.0]
        ]))
        
        # 添加光源
        light = DirectionalLight(color=np.ones(3), intensity=3.0)
        scene.add(light, pose=np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0]
        ]))
        
        # 添加侧面光源
        side_light = DirectionalLight(color=np.ones(3), intensity=2.0)
        scene.add(side_light, pose=np.array([
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ]))
        
        # 创建渲染器
        renderer = OffscreenRenderer(width, height)
        
        return scene, camera, renderer
    
    def render_model_frame(self, body_file, garment_file, output_image=None, render_size=None, flip_vertical=False):
        """
        渲染单帧模型图像
        
        参数:
        - body_file: 身体模型文件路径
        - garment_file: 服装模型文件路径
        - output_image: 输出图像路径，如果为None则自动生成
        - render_size: 渲染尺寸，默认(1024, 1024)
        - flip_vertical: 是否垂直翻转模型
        
        返回:
        - output_image: 渲染的图像路径
        """
        if render_size is None:
            render_size = self.default_render_size
            
        if output_image is None:
            base_name = os.path.basename(body_file).replace("body_", "")
            output_image = os.path.join(self.output_dir, f"render_{base_name.replace('.ply', '.png')}")
        
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
            scene, camera, renderer = self._setup_basic_scene(render_size[0], render_size[1])
            
            # 添加身体和服装到场景
            body_pose = np.eye(4)
            scene.add(Mesh.from_trimesh(body_mesh, smooth=True), pose=body_pose)
            scene.add(Mesh.from_trimesh(garment_mesh, smooth=True), pose=body_pose)
            
            # 渲染场景
            color, depth = renderer.render(scene, flags=RenderFlags.RGBA)
            
            # 保存图像 - 修改部分开始
            plt.figure(figsize=(10, 10))
            plt.imshow(color)
            plt.axis('off')
            # 使用dpi和figsize来控制最终尺寸，确保为偶数
            dpi = 100
            fig_width = render_size[0] / dpi
            fig_height = render_size[1] / dpi
            plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
            plt.imshow(color)
            plt.axis('off')
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            plt.savefig(output_image, dpi=dpi, bbox_inches=None, pad_inches=0)
            # 修改部分结束
            plt.close()
            
            # 清理资源
            renderer.delete()
            
            print(f"渲染图像保存到: {output_image}")
            return output_image
            
        except Exception as e:
            print(f"渲染模型时出错: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def render_models_to_video(self, models_dir, output_video=None, fps=30, quality='medium', 
                              start_frame=0, end_frame=None, person_id=None, rotation=False,
                              rotation_speed=1.0, flip_vertical=False):
        """
        将模型序列渲染为视频
        
        参数:
        - models_dir: 包含模型文件的目录
        - output_video: 输出视频路径，如果为None则自动生成
        - fps: 视频帧率
        - quality: 视频质量，'low', 'medium', 'high'
        - start_frame: 起始帧
        - end_frame: 结束帧，None表示处理所有帧
        - person_id: 人物ID，None表示处理所有人物
        - rotation: 是否旋转模型
        - rotation_speed: 旋转速度，默认1.0
        - flip_vertical: 是否垂直翻转模型
        
        返回:
        - frames: 处理的帧信息
        """
        print(f"从目录渲染视频: {models_dir}")
        
        # 查找目录中的所有body模型
        body_files = sorted(glob.glob(os.path.join(models_dir, "body_*.ply")))
        if not body_files:
            print(f"未在{models_dir}中找到任何body模型")
            return None
        
        # 提取帧信息
        frame_pattern = re.compile(r'frame(\d+)_person(\d+)')
        frames = {}
        
        for body_file in body_files:
            filename = os.path.basename(body_file)
            match = frame_pattern.search(filename)
            if match:
                frame_num = int(match.group(1))
                person_num = int(match.group(2))
                
                # 检查是否在指定范围内
                if frame_num < start_frame:
                    continue
                if end_frame is not None and frame_num > end_frame:
                    continue
                if person_id is not None and person_num != person_id:
                    continue
                
                # 查找对应的garment文件
                garment_file = os.path.join(models_dir, filename.replace("body_", "pred_gar_"))
                if os.path.exists(garment_file):
                    if person_num not in frames:
                        frames[person_num] = []
                    frames[person_num].append((frame_num, body_file, garment_file))
        
        if not frames:
            print(f"在指定范围内未找到任何匹配的模型")
            return None
        
        # 设置视频编码器参数
        if quality == 'high':
            codec = 'h264'
            bitrate = '5000k'
        elif quality == 'medium':
            codec = 'h264'
            bitrate = '2500k'
        else:
            codec = 'h264'
            bitrate = '1000k'
        
        # 为每个人物创建视频
        for person_num, person_frames in frames.items():
            print(f"处理人物 {person_num} 的 {len(person_frames)} 帧...")
            person_frames.sort()  # 确保按帧排序
            
            # 设置默认输出视频路径
            if output_video is None:
                output_video = os.path.join(self.output_dir, f"render_person{person_num:02d}.mp4")
            
            # 创建渲染目录
            render_dir = os.path.join(self.output_dir, f"render_frames_person{person_num:02d}")
            os.makedirs(render_dir, exist_ok=True)
            
            # 记录生成的图像路径
            frame_images = []
            
            # 渲染每一帧
            print("渲染帧...")
            for frame_idx, (frame_num, body_file, garment_file) in enumerate(tqdm(person_frames)):
                try:
                    # 输出图像路径
                    output_image = os.path.join(render_dir, f"frame_{frame_num:04d}.png")
                    
                    # 如果开启旋转，计算当前旋转角度
                    if rotation:
                        angle = (frame_idx * rotation_speed) % 360
                        image_path = self.render_rotated_model(body_file, garment_file, angle, output_image, flip_vertical=flip_vertical)
                    else:
                        image_path = self.render_model_frame(body_file, garment_file, output_image, flip_vertical=flip_vertical)
                    
                    if image_path:
                        frame_images.append(image_path)
                except Exception as e:
                    print(f"渲染帧 {frame_num} 时出错: {e}")
            
            # 使用ffmpeg创建视频
            if frame_images:
                try:
                    # 构建ffmpeg命令
                    frames_pattern = os.path.join(render_dir, "frame_%04d.png")
                    # 添加-vf scale选项，强制宽高为偶数
                    cmd = f"ffmpeg -y -framerate {fps} -i \"{frames_pattern}\" -vf \"scale=trunc(iw/2)*2:trunc(ih/2)*2\" -c:v {codec} -b:v {bitrate} -pix_fmt yuv420p \"{output_video}\""
                    
                    print(f"正在创建视频: {output_video}")
                    print(f"执行命令: {cmd}")
                    
                    # 执行命令
                    os.system(cmd)
                    
                    if os.path.exists(output_video):
                        print(f"成功创建视频: {output_video}")
                    else:
                        print(f"视频创建失败: {output_video}")
                        
                except Exception as e:
                    print(f"创建视频时出错: {e}")
        
        return frames
    
    def render_rotated_model(self, body_file, garment_file, angle_degrees, output_image=None, render_size=None, flip_vertical=False):
        """
        渲染带有旋转角度的模型
        
        参数:
        - body_file: 身体模型文件路径
        - garment_file: 服装模型文件路径
        - angle_degrees: 旋转角度（度）
        - output_image: 输出图像路径
        - render_size: 渲染尺寸
        - flip_vertical: 是否垂直翻转模型
        
        返回:
        - output_image: 渲染的图像路径
        """
        if render_size is None:
            render_size = self.default_render_size
            
        if output_image is None:
            base_name = os.path.basename(body_file).replace("body_", "")
            output_image = os.path.join(self.output_dir, f"render_rot{angle_degrees:03.0f}_{base_name.replace('.ply', '.png')}")
        
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
            scene, camera, renderer = self._setup_basic_scene(render_size[0], render_size[1])
            
            # 创建旋转矩阵
            angle_rad = np.radians(angle_degrees)
            rotation = np.array([
                [np.cos(angle_rad), 0, np.sin(angle_rad), 0],
                [0, 1, 0, 0],
                [-np.sin(angle_rad), 0, np.cos(angle_rad), 0],
                [0, 0, 0, 1]
            ])
            
            # 添加身体和服装到场景
            scene.add(Mesh.from_trimesh(body_mesh, smooth=True), pose=rotation)
            scene.add(Mesh.from_trimesh(garment_mesh, smooth=True), pose=rotation)
            
            # 渲染场景
            color, depth = renderer.render(scene, flags=RenderFlags.RGBA)
            
            # 保存图像 - 修改部分开始
            dpi = 100
            fig_width = render_size[0] / dpi
            fig_height = render_size[1] / dpi
            plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
            plt.imshow(color)
            plt.axis('off')
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            plt.savefig(output_image, dpi=dpi, bbox_inches=None, pad_inches=0)
            # 修改部分结束
            plt.close()
            
            # 清理资源
            renderer.delete()
            
            return output_image
            
        except Exception as e:
            print(f"渲染旋转模型时出错: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def render_models_with_original_video(self, original_video, models_dir, hybrik_result_file, 
                                         output_video=None, start_frame=0, end_frame=None, 
                                         person_id=None, alpha=0.7):
        """
        将3D模型叠加到原始视频上
        
        参数:
        - original_video: 原始视频路径
        - models_dir: 包含模型文件的目录
        - hybrik_result_file: HybrIK结果文件，包含边界框信息
        - output_video: 输出视频路径
        - start_frame: 起始帧
        - end_frame: 结束帧
        - person_id: 人物ID
        - alpha: 透明度混合系数
        
        返回:
        - output_video: 输出视频路径
        """
        if output_video is None:
            output_video = os.path.join(self.output_dir, f"combined_video_{int(time.time())}.mp4")
        
        print(f"将3D模型叠加到原始视频上...")
        print(f"原始视频: {original_video}")
        print(f"模型目录: {models_dir}")
        print(f"HybrIK结果: {hybrik_result_file}")
        
        try:
            # 加载HybrIK结果以获取边界框信息
            with open(hybrik_result_file, 'r') as f:
                hybrik_results = json.load(f)
            
            # 打开原始视频
            cap = cv2.VideoCapture(original_video)
            if not cap.isOpened():
                print(f"无法打开视频: {original_video}")
                # 添加诊断信息
                print(f"文件是否存在: {os.path.exists(original_video)}")
                if os.path.exists(original_video):
                    print(f"文件权限: {oct(os.stat(original_video).st_mode)[-3:]}")
                else:
                    print("文件不存在，无法获取权限")
                print(f"尝试使用ffmpeg检查视频信息...")
                os.system(f"ffmpeg -i {original_video} 2>&1")
                
                # 尝试使用绝对路径
                abs_video_path = os.path.abspath(original_video)
                print(f"尝试使用绝对路径: {abs_video_path}")
                cap_abs = cv2.VideoCapture(abs_video_path)
                if not cap_abs.isOpened():
                    print(f"使用绝对路径仍然无法打开视频")
                    
                    # 检查OpenCV是否支持视频编解码器
                    print("\n检查OpenCV编解码器支持:")
                    import platform
                    print(f"平台: {platform.platform()}")
                    print(f"OpenCV版本: {cv2.__version__}")
                    
                    # 建议使用ffmpeg直接处理
                    print("\n建议: 使用video命令生成单独的模型视频，然后用ffmpeg手动叠加到原始视频上")
                    
                    return None
                else:
                    print(f"使用绝对路径成功打开视频！")
                    cap = cap_abs
            
            # 获取视频属性
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"视频信息: {width}x{height}@{fps}fps")
            
            # 创建输出视频编码器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
            
            # 检查输出视频是否创建成功
            if not out.isOpened():
                print(f"无法创建输出视频: {output_video}")
                return None
            
            # 创建渲染器
            renderer = OffscreenRenderer(width, height)
            
            # 处理视频帧
            frame_idx = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            with tqdm(total=total_frames) as pbar:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    if frame_idx < start_frame:
                        frame_idx += 1
                        pbar.update(1)
                        continue
                    
                    if end_frame is not None and frame_idx > end_frame:
                        break
                    
                    # 处理当前帧的所有人物
                    frame_processed = False
                    
                    if frame_idx < len(hybrik_results):
                        # 获取当前帧的结果
                        frame_result = hybrik_results[frame_idx]
                        
                        # 可能的结果格式处理
                        if isinstance(frame_result, dict) and 'results' in frame_result:
                            people_list = frame_result['results']
                        elif isinstance(frame_result, list):
                            people_list = frame_result
                        else:
                            people_list = [frame_result]
                        
                        # 处理每个人物
                        for person_idx, person_data in enumerate(people_list):
                            if person_id is not None and person_idx != person_id:
                                continue
                            
                            # 获取边界框
                            bbox = None
                            if isinstance(person_data, dict):
                                bbox = person_data.get('bbox', None)
                            
                            if bbox is None:
                                continue
                            
                            # 构造模型文件路径
                            body_file = os.path.join(models_dir, f"body_t-shirt_male_frame{frame_idx:04d}_person{person_idx:02d}.ply")
                            garment_file = os.path.join(models_dir, f"pred_gar_t-shirt_male_frame{frame_idx:04d}_person{person_idx:02d}.ply")
                            
                            if not (os.path.exists(body_file) and os.path.exists(garment_file)):
                                continue
                            
                            # 设置场景
                            scene = Scene(bg_color=[0.0, 0.0, 0.0, 0.0])  # 透明背景
                            
                            # 添加相机
                            camera = PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=float(width) / height)
                            scene.add(camera, pose=np.array([
                                [1.0, 0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0, 2.5],
                                [0.0, 0.0, 0.0, 1.0]
                            ]))
                            
                            # 添加光源
                            light = DirectionalLight(color=np.ones(3), intensity=3.0)
                            scene.add(light, pose=np.array([
                                [1.0, 0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0, 1.0],
                                [0.0, 0.0, 0.0, 1.0]
                            ]))
                            
                            # 加载3D模型
                            body_mesh = trimesh.load(body_file)
                            garment_mesh = trimesh.load(garment_file)
                            
                            # 添加身体和服装到场景
                            body_pose = np.eye(4)
                            scene.add(Mesh.from_trimesh(body_mesh, smooth=True), pose=body_pose)
                            scene.add(Mesh.from_trimesh(garment_mesh, smooth=True), pose=body_pose)
                            
                            # 渲染场景
                            color, depth = renderer.render(scene)
                            
                            # 将渲染的模型叠加到视频帧上
                            try:
                                x1, y1, x2, y2 = [int(float(coord)) for coord in bbox]
                                x1 = max(0, x1)
                                y1 = max(0, y1)
                                x2 = min(width, x2)
                                y2 = min(height, y2)
                                
                                # 调整渲染图像大小以适应边界框
                                render_height = y2 - y1
                                render_width = x2 - x1
                                resized_render = cv2.resize(color, (render_width, render_height))
                                
                                # 创建掩码
                                mask = resized_render[:, :, 3] > 0
                                
                                # 叠加渲染图像到原始帧
                                for c in range(3):  # 仅处理RGB通道
                                    frame[:, :, c] = np.where(
                                        mask,
                                        frame[:, :, c] * (1 - alpha) + resized_render[:, :, c] * alpha,
                                        frame[:, :, c]
                                    )
                                
                                frame_processed = True
                                
                            except Exception as e:
                                print(f"处理帧 {frame_idx} 人物 {person_idx} 时出错: {e}")
                                continue
                    
                    # 写入输出视频
                    out.write(frame)
                    frame_idx += 1
                    pbar.update(1)
            
            # 释放资源
            cap.release()
            out.release()
            renderer.delete()
            
            if frame_processed:
                print(f"成功创建叠加视频: {output_video}")
                return output_video
            else:
                print("警告: 没有处理任何帧，请检查边界框信息和模型文件路径")
                return None
                
        except Exception as e:
            print(f"叠加视频处理出错: {e}")
            import traceback
            traceback.print_exc()
            return None

    def auto_blend_videos(self, original_video, model_video, output_video=None, similarity_threshold=0.3, colorkey_value=0xFFFFFF, colorkey_similarity=0.3, colorkey_blend=0.2):
        """
        使用ffmpeg自动将模型视频叠加到原始视频上
        
        参数:
        - original_video: 原始视频路径
        - model_video: 模型视频路径
        - output_video: 输出视频路径
        - similarity_threshold: 相似度阈值，用于检测匹配度
        - colorkey_value: 颜色键值（通常是背景色）
        - colorkey_similarity: 颜色键相似度
        - colorkey_blend: 颜色键混合系数
        
        返回:
        - output_video: 输出视频路径
        """
        if output_video is None:
            output_video = os.path.join(self.output_dir, f"auto_combined_{int(time.time())}.mp4")
        
        try:
            # 检查视频是否存在
            if not os.path.exists(original_video):
                print(f"原始视频不存在: {original_video}")
                return None
                
            if not os.path.exists(model_video):
                print(f"模型视频不存在: {model_video}")
                return None
            
            # 构建ffmpeg命令
            colorkey_hex = f"0x{colorkey_value:06X}"
            cmd = f'ffmpeg -i "{original_video}" -i "{model_video}" -filter_complex "[1:v]colorkey={colorkey_hex}:{colorkey_similarity}:{colorkey_blend}[ckout];[0:v][ckout]overlay[out]" -map "[out]" -c:v libx264 -pix_fmt yuv420p "{output_video}"'
            
            print(f"正在叠加视频...")
            print(f"执行命令: {cmd}")
            
            # 执行命令
            os.system(cmd)
            
            if os.path.exists(output_video):
                print(f"成功创建叠加视频: {output_video}")
                return output_video
            else:
                print(f"视频叠加失败: {output_video}")
                return None
                
        except Exception as e:
            print(f"自动叠加视频时出错: {e}")
            import traceback
            traceback.print_exc()
            return None

    def auto_align_videos(self, original_video, models_dir, output_video=None, hybrik_result=None, start_frame=0, end_frame=None, person_ids=None, flip_vertical=True):
        """
        自动将3D模型与原始视频中的人物对齐并叠加
        
        参数:
        - original_video: 原始视频路径
        - models_dir: 包含模型的目录路径
        - output_video: 输出视频路径
        - hybrik_result: HybrIK结果文件路径（可选，用于更精确对齐）
        - start_frame: 起始帧
        - end_frame: 结束帧
        - person_ids: 人物ID列表，None表示处理所有人物
        - flip_vertical: 是否垂直翻转模型(默认开启以修正模型方向)
        
        返回:
        - output_video: 输出视频路径
        """
        if output_video is None:
            output_video = os.path.join(self.output_dir, f"aligned_video_{int(time.time())}.mp4")
            
        # 检查视频文件路径
        self._check_path_existence("原始视频", original_video)
        self._check_path_existence("模型目录", models_dir)
        if hybrik_result:
            self._check_path_existence("HybrIK结果文件", hybrik_result)
            
        # 尝试转换为绝对路径
        original_video = self._ensure_valid_path(original_video)
        models_dir = self._ensure_valid_path(models_dir)
        if hybrik_result:
            hybrik_result = self._ensure_valid_path(hybrik_result)
        
        # 步骤1: 加载HybrIK结果 - 先检查是否有更详细的信息可用
        advanced_alignment = False
        hybrik_data = None
        
        if hybrik_result and os.path.exists(hybrik_result):
            try:
                print(f"加载HybrIK结果以获取精确对齐信息: {hybrik_result}")
                with open(hybrik_result, 'r') as f:
                    hybrik_data = json.load(f)
                
                # 检查是否包含关键的高级对齐信息
                if hybrik_data and len(hybrik_data) > 0:
                    # 检查第一帧第一个人的数据格式
                    first_frame = hybrik_data[0]
                    person_data = None
                    
                    # 找到可能的人物数据
                    if isinstance(first_frame, dict) and 'results' in first_frame:
                        if first_frame['results'] and len(first_frame['results']) > 0:
                            person_data = first_frame['results'][0]
                    elif isinstance(first_frame, list) and len(first_frame) > 0:
                        person_data = first_frame[0]
                    elif isinstance(first_frame, dict) and 'track_id' in first_frame:
                        person_data = first_frame
                    
                    # 检查是否存在高级对齐所需的信息
                    if person_data and isinstance(person_data, dict):
                        has_keypoints = 'keypoints' in person_data or 'joints' in person_data
                        has_pose = 'poses' in person_data or 'pred_thetas_mat' in person_data or 'smpl_pose' in person_data
                        has_camera = 'camera' in person_data or 'cam' in person_data
                        
                        advanced_alignment = has_keypoints or has_pose or has_camera
                        print(f"检测到高级对齐信息: 关键点={has_keypoints}, 姿势={has_pose}, 相机参数={has_camera}")
            except Exception as e:
                print(f"解析HybrIK结果时出错: {e}")
                import traceback
                traceback.print_exc()
        
        # 步骤2: 准备输入和输出目录
        print(f"处理视频: {original_video}")
        print(f"模型目录: {models_dir}")
        
        # 获取视频属性
        cap = cv2.VideoCapture(original_video)
        if not cap.isOpened():
            print(f"无法打开视频: {original_video}")
            return None
        
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        print(f"视频信息: {video_width}x{video_height}@{fps}fps, 总帧数: {total_frames}")
        
        # 步骤3: 查找和处理所有人物
        body_files = sorted(glob.glob(os.path.join(models_dir, "body_*.ply")))
        person_pattern = re.compile(r'person(\d+)')
        unique_persons = set()
        
        for body_file in body_files:
            filename = os.path.basename(body_file)
            match = person_pattern.search(filename)
            if match:
                person_num = int(match.group(1))
                unique_persons.add(person_num)
        
        if person_ids is not None:
            unique_persons = [p for p in unique_persons if p in person_ids]
        
        print(f"检测到{len(unique_persons)}个人物: {unique_persons}")
        
        # 如果没有人物或没有HybrIK数据，则使用基本方法
        if not unique_persons or not hybrik_data:
            print("未检测到人物或无HybrIK数据，使用基本渲染方法")
            return self._basic_align_videos(original_video, models_dir, output_video, 
                                         start_frame, end_frame, person_ids, flip_vertical)
        
        # 步骤4: 如果有高级对齐信息，使用精确渲染方法
        if advanced_alignment:
            print("使用高级对齐方法")
            return self._advanced_align_videos(original_video, models_dir, hybrik_data, output_video,
                                           start_frame, end_frame, unique_persons, flip_vertical)
        else:
            print("使用基本渲染方法")
            return self._basic_align_videos(original_video, models_dir, output_video,
                                         start_frame, end_frame, person_ids, flip_vertical)

    def _basic_align_videos(self, original_video, models_dir, output_video=None, 
                         start_frame=0, end_frame=None, person_ids=None, flip_vertical=True):
        """基本对齐方法，不使用姿势或相机参数"""
        if output_video is None:
            output_video = os.path.join(self.output_dir, f"basic_aligned_{int(time.time())}.mp4")
            
        # 检查路径有效性
        original_video = self._ensure_valid_path(original_video)
        models_dir = self._ensure_valid_path(models_dir)
        
        # 检查视频文件是否存在
        if not os.path.exists(original_video):
            print(f"错误: 视频文件不存在: {original_video}")
            return None
            
        # 检查视频文件是否可以打开
        try:
            # 尝试使用ffmpeg获取视频信息，而不是OpenCV
            import subprocess
            print(f"使用ffmpeg检查视频: {original_video}")
            result = subprocess.run(['ffmpeg', '-i', original_video], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE)
            stderr = result.stderr.decode()
            
            # 从ffmpeg输出中提取视频信息
            import re
            duration_match = re.search(r'Duration: (\d+):(\d+):(\d+)', stderr)
            if duration_match:
                print(f"视频持续时间: {duration_match.group(0)}")
            else:
                print("警告: 无法从ffmpeg输出中获取视频持续时间")
                print(f"ffmpeg输出: {stderr}")
        except Exception as e:
            print(f"使用ffmpeg检查视频时出错: {e}")
            # 尝试使用OpenCV打开视频
            cap = cv2.VideoCapture(original_video)
            if not cap.isOpened():
                print(f"OpenCV也无法打开视频文件")
                print("尝试直接使用ffmpeg处理...")
            else:
                cap.release()
                
        # 查找目录中的所有人物
        body_files = sorted(glob.glob(os.path.join(models_dir, "body_*.ply")))
        person_pattern = re.compile(r'person(\d+)')
        unique_persons = set()
        
        for body_file in body_files:
            filename = os.path.basename(body_file)
            match = person_pattern.search(filename)
            if match:
                person_num = int(match.group(1))
                unique_persons.add(person_num)
        
        if person_ids is not None:
            unique_persons = [p for p in unique_persons if p in person_ids]
            
        # 为每个人物生成视频
        person_videos = {}
        for person_num in unique_persons:
            temp_video = os.path.join(self.output_dir, f"temp_person{person_num:02d}.mp4")
            self.render_models_to_video(
                models_dir,
                temp_video,
                start_frame=start_frame,
                end_frame=end_frame,
                person_id=person_num,
                flip_vertical=flip_vertical
            )
            person_videos[person_num] = temp_video
            
        # 使用ffmpeg直接叠加视频
        print("使用ffmpeg直接叠加视频...")
        # 确保至少有一个人物视频存在
        valid_person_videos = []
        for person_num, person_video in person_videos.items():
            if os.path.exists(person_video):
                valid_person_videos.append((person_num, person_video))
            else:
                print(f"警告: 人物视频不存在: {person_video}")
                
        if not valid_person_videos:
            print("错误: 没有有效的人物视频")
            return None
            
        # 使用ffmpeg直接叠加
        try:
            # 构建ffmpeg命令来叠加视频
            filter_complex = []
            # 首先添加原始视频
            inputs = ['-i', original_video]
            
            # 对每个人物视频，添加输入和overlay滤镜
            for i, (person_num, person_video) in enumerate(valid_person_videos):
                input_index = i + 1  # 0是原始视频
                inputs.extend(['-i', person_video])
                
                # 对每个叠加的视频应用colorkey滤镜
                filter_complex.append(f"[{input_index}:v]colorkey=0xFFFFFF:0.3:0.2[ckout{i}]")
                
                # 叠加到前一个结果上
                if i == 0:
                    # 第一个人物叠加到原始视频
                    filter_complex.append(f"[0:v][ckout{i}]overlay[out{i}]")
                else:
                    # 后续人物叠加到前一个结果
                    filter_complex.append(f"[out{i-1}][ckout{i}]overlay[out{i}]")
            
            # 构建完整的ffmpeg命令
            last_output = f"out{len(valid_person_videos)-1}"
            filter_expr = ";".join(filter_complex)
            cmd = ['ffmpeg', '-y'] + inputs + ['-filter_complex', filter_expr, 
                                             '-map', f'[{last_output}]', 
                                             '-c:v', 'libx264', 
                                             '-pix_fmt', 'yuv420p', 
                                             output_video]
            
            print(f"执行ffmpeg命令: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            
            if os.path.exists(output_video):
                print(f"成功创建叠加视频: {output_video}")
                return output_video
            else:
                print(f"ffmpeg未能创建输出视频: {output_video}")
                return None
                
        except Exception as e:
            print(f"使用ffmpeg叠加视频时出错: {e}")
            import traceback
            traceback.print_exc()
            
            # 尝试使用简单的方法
            print("尝试使用简单叠加方法...")
            temp_output = original_video
            for person_num, person_video in valid_person_videos:
                overlay_output = os.path.join(self.output_dir, f"temp_overlay_{person_num}.mp4")
                temp_output = self._simple_overlay(temp_output, person_video, overlay_output)
                
            # 复制最终结果到输出路径
            import shutil
            if temp_output != original_video:  # 确保至少处理了一个视频
                shutil.copy(temp_output, output_video)
                print(f"完成视频叠加: {output_video}")
                return output_video
            
            return None
        
    def _advanced_align_videos(self, original_video, models_dir, hybrik_data, output_video=None,
                            start_frame=0, end_frame=None, unique_persons=None, flip_vertical=True):
        """高级对齐方法，使用HybrIK的姿势和相机参数进行精确对齐"""
        if output_video is None:
            output_video = os.path.join(self.output_dir, f"advanced_aligned_{int(time.time())}.mp4")
            
        # 检查路径有效性
        original_video = self._ensure_valid_path(original_video)
        models_dir = self._ensure_valid_path(models_dir)
                    
        print("\n========== 使用HybrIK高级参数进行精确对齐 ==========")
        
        # 检查视频文件是否存在
        if not os.path.exists(original_video):
            print(f"错误: 视频文件不存在: {original_video}")
            return self._basic_align_videos(original_video, models_dir, output_video, 
                                         start_frame, end_frame, unique_persons, flip_vertical)
            
        # 使用ffmpeg获取视频信息，避免OpenCV的问题
        try:
            import subprocess
            import re
            
            print(f"使用ffmpeg获取视频信息: {original_video}")
            result = subprocess.run(['ffmpeg', '-i', original_video], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE)
            stderr = result.stderr.decode()
            
            # 从ffmpeg输出中提取视频信息
            duration_match = re.search(r'Duration: (\d+):(\d+):(\d+)', stderr)
            fps_match = re.search(r'(\d+(?:\.\d+)?) fps', stderr)
            resolution_match = re.search(r'(\d+x\d+)', stderr)
            
            if duration_match:
                hours, minutes, seconds = map(int, duration_match.groups())
                total_seconds = hours * 3600 + minutes * 60 + seconds
                total_frames = int(total_seconds * (float(fps_match.group(1)) if fps_match else 30))
                print(f"视频持续时间: {duration_match.group(0)}, 估计总帧数: {total_frames}")
            else:
                print("警告: 无法从ffmpeg输出中获取视频持续时间")
                print(f"ffmpeg输出: {stderr}")
                total_frames = 1000  # 默认值
                
            if resolution_match:
                resolution = resolution_match.group(1)
                width, height = map(int, resolution.split('x'))
                print(f"视频分辨率: {width}x{height}")
            else:
                print("警告: 无法从ffmpeg输出中获取视频分辨率")
                # 使用默认分辨率
                width, height = 1920, 1080
                
            fps = float(fps_match.group(1)) if fps_match else 30
            print(f"视频帧率: {fps}")
            
            # 如果没有明确指定结束帧，使用总帧数
            if end_frame is None:
                end_frame = total_frames
                
        except Exception as e:
            print(f"使用ffmpeg获取视频信息时出错: {e}")
            import traceback
            traceback.print_exc()
            
            # 尝试使用OpenCV打开视频
            try:
                cap = cv2.VideoCapture(original_video)
                if not cap.isOpened():
                    print(f"OpenCV也无法打开视频文件: {original_video}")
                    print("回退到基本对齐方法...")
                    return self._basic_align_videos(original_video, models_dir, output_video, 
                                                start_frame, end_frame, unique_persons, flip_vertical)
                else:
                    # 获取视频属性
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
            except Exception as e:
                print(f"使用OpenCV获取视频信息时出错: {e}")
                print("无法处理视频，回退到基本对齐方法...")
                return self._basic_align_videos(original_video, models_dir, output_video, 
                                            start_frame, end_frame, unique_persons, flip_vertical)
        
        # 使用提取到的视频信息继续处理
        effective_frames = end_frame - start_frame
        
        # 步骤1: 为每个人的每一帧创建方向和位置矩阵，用于精确对齐模型
        # 创建临时目录存储渲染的每个人的每一帧
        temp_dir = os.path.join(self.output_dir, f"temp_advanced_frames_{int(time.time())}")
        os.makedirs(temp_dir, exist_ok=True)
        
        # 计算转换矩阵
        print("计算HybrIK姿势和相机参数...")
        pose_matrices = {}  # {person_id: {frame_id: transformation_matrix}}
        for person_id in unique_persons:
            pose_matrices[person_id] = {}
        
        # ... 保持其余代码不变 ...
        
        # 由于视频读取问题，最终可能需要使用ffmpeg处理视频帧
        print("\n由于WSL环境下OpenCV可能无法正确处理视频，我们将采用两步法:")
        print("1. 渲染模型到图像序列")
        print("2. 使用ffmpeg将模型图像叠加到原始视频上")
        
        # 创建输出目录
        frames_dir = os.path.join(temp_dir, "original_frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        # 使用ffmpeg提取原始视频帧
        try:
            print(f"提取原始视频帧: {original_video}")
            extract_cmd = [
                'ffmpeg', '-i', original_video,
                '-vf', f'select=between(n\\,{start_frame}\\,{end_frame})',
                '-vsync', '0',
                os.path.join(frames_dir, 'frame_%04d.png')
            ]
            subprocess.run(extract_cmd, check=True)
        except Exception as e:
            print(f"提取视频帧时出错: {e}")
            print("尝试回退到基本方法...")
            return self._basic_align_videos(original_video, models_dir, output_video, 
                                        start_frame, end_frame, unique_persons, flip_vertical)
        
        # 准备合成命令的滤镜表达式
        filter_parts = []
        
        # 检查是否有原始帧
        original_frames = sorted(glob.glob(os.path.join(frames_dir, 'frame_*.png')))
        if not original_frames:
            print("错误: 未能提取原始视频帧")
            return None
            
        # 构建临时的overlay脚本
        overlay_script = os.path.join(temp_dir, "overlay_script.txt")
        with open(overlay_script, 'w') as f:
            f.write(f"# FFmpeg滤镜脚本\n")
            # 初始化滤镜链
            f.write(f"[0:v]format=rgba,setpts=PTS-STARTPTS[bg];\n")
            
            # 添加每个人物的模型叠加
            for i, person_id in enumerate(unique_persons):
                # 为该人物的帧创建目录
                person_frames_dir = os.path.join(temp_dir, f"person{person_id}_frames")
                os.makedirs(person_frames_dir, exist_ok=True)
                
                # 为每一帧渲染模型
                print(f"渲染人物{person_id}的模型帧...")
                for frame_idx in range(start_frame, end_frame):
                    if frame_idx not in pose_matrices[person_id]:
                        continue
                        
                    # 获取当前帧的模型文件名
                    garment_prefix = "pred_gar_"
                    body_prefix = "body_"
                    
                    # 查找匹配的模型文件
                    frame_pattern = f"frame{frame_idx:04d}_person{person_id:02d}.ply"
                    
                    body_candidates = glob.glob(os.path.join(models_dir, f"{body_prefix}*{frame_pattern}"))
                    if not body_candidates:
                        continue
                        
                    body_file = body_candidates[0]
                    garment_file = body_file.replace(body_prefix, garment_prefix)
                    
                    if not os.path.exists(garment_file):
                        continue
                    
                    # 获取转换信息
                    transform_info = pose_matrices[person_id][frame_idx]
                    bbox = transform_info['bbox']
                    
                    # 渲染模型
                    output_image = os.path.join(person_frames_dir, f"frame_{frame_idx - start_frame:04d}.png")
                    self._render_with_transform(
                        body_file, 
                        garment_file, 
                        transform_info['matrix'],
                        transform_info['camera'], 
                        output_image, 
                        (width, height),
                        bbox,
                        flip_vertical
                    )
                
                # 添加该人物的模型序列到滤镜链
                f.write(f"[{i+1}:v]format=rgba,setpts=PTS-STARTPTS[model{i}];\n")
                
                # 叠加到背景或前一个结果
                if i == 0:
                    f.write(f"[bg][model{i}]overlay=shortest=1[v{i}];\n")
                else:
                    f.write(f"[v{i-1}][model{i}]overlay=shortest=1[v{i}];\n")
                    
            # 最终输出
            last_v = f"v{len(unique_persons)-1}"
            f.write(f"[{last_v}]")
            
        # 使用ffmpeg合成最终视频
        try:
            print("使用ffmpeg合成最终视频...")
            # 为每个人物准备输入序列
            inputs = ['-i', original_video]
            for person_id in unique_persons:
                person_frames_dir = os.path.join(temp_dir, f"person{person_id}_frames")
                person_frames = sorted(glob.glob(os.path.join(person_frames_dir, 'frame_*.png')))
                if person_frames:
                    # 创建该人物的临时视频
                    person_video = os.path.join(temp_dir, f"person{person_id}.mp4")
                    sequence_cmd = [
                        'ffmpeg', '-y', 
                        '-framerate', str(fps),
                        '-i', os.path.join(person_frames_dir, 'frame_%04d.png'),
                        '-c:v', 'libx264', 
                        '-pix_fmt', 'yuva420p',  # 使用alpha通道
                        person_video
                    ]
                    subprocess.run(sequence_cmd, check=True)
                    inputs.extend(['-i', person_video])
            
            # 构建合成命令
            final_cmd = [
                'ffmpeg', '-y'
            ] + inputs + [
                '-filter_complex_script', overlay_script,
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                output_video
            ]
            
            print(f"执行合成命令: {' '.join(final_cmd)}")
            subprocess.run(final_cmd, check=True)
            
            if os.path.exists(output_video):
                print(f"成功创建高级对齐视频: {output_video}")
                return output_video
            else:
                print("合成失败，尝试基本对齐方法...")
                return self._basic_align_videos(original_video, models_dir, output_video, 
                                           start_frame, end_frame, unique_persons, flip_vertical)
        except Exception as e:
            print(f"合成视频时出错: {e}")
            import traceback
            traceback.print_exc()
            print("尝试基本对齐方法...")
            return self._basic_align_videos(original_video, models_dir, output_video, 
                                       start_frame, end_frame, unique_persons, flip_vertical)

    def _render_with_transform(self, body_file, garment_file, transform_matrix, camera_params, 
                            output_image, render_size, bbox, flip_vertical=True):
        """使用精确的转换矩阵和相机参数渲染模型"""
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
                # 应用翻转矩阵
                body_mesh.apply_transform(flip_matrix)
                garment_mesh.apply_transform(flip_matrix)
            
            # 设置场景和相机
            width, height = render_size
            scene = Scene(bg_color=[0.0, 0.0, 0.0, 0.0])  # 透明背景
            
            # 计算合适的相机焦距 - 使用边界框信息
            if bbox:
                x1, y1, x2, y2 = [float(b) for b in bbox]
                box_width = x2 - x1
                box_height = y2 - y1
                # 使用边界框大小计算合适的焦距
                focal_length = max(box_width, box_height) * 2.0
            else:
                focal_length = min(width, height)
            
            # 设置相机 - 如果有相机参数则使用
            if camera_params and isinstance(camera_params, list) and len(camera_params) >= 1:
                try:
                    # 使用API提供的相机参数
                    camera_scale = camera_params[0]  # 通常第一个参数是焦距
                    focal_length = focal_length * camera_scale
                except:
                    pass
            
            # 创建相机并添加到场景
            camera = PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=float(width) / height)
            camera_pose = np.array([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, focal_length/100.0],  # 缩放焦距以适应场景
                [0.0, 0.0, 0.0, 1.0]
            ])
            scene.add(camera, pose=camera_pose)
            
            # 添加光源
            light = DirectionalLight(color=np.ones(3), intensity=3.0)
            scene.add(light, pose=np.eye(4))
            
            # 应用转换矩阵
            if transform_matrix is not None:
                # 使用HybrIK提供的精确转换矩阵
                model_pose = transform_matrix
            else:
                # 使用默认姿势
                model_pose = np.eye(4)
            
            # 添加模型到场景
            scene.add(Mesh.from_trimesh(body_mesh, smooth=True), pose=model_pose)
            scene.add(Mesh.from_trimesh(garment_mesh, smooth=True), pose=model_pose)
            
            # 创建渲染器并渲染
            renderer = OffscreenRenderer(width, height)
            color, depth = renderer.render(scene, flags=RenderFlags.RGBA)
            
            # 保存渲染结果
            cv2.imwrite(output_image, color)
            
            # 释放资源
            renderer.delete()
            return True
            
        except Exception as e:
            print(f"使用转换矩阵渲染时出错: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _simple_overlay(self, base_video, overlay_video, output_video, colorkey_value=0xFFFFFF, colorkey_similarity=0.3, colorkey_blend=0.2):
        """使用colorkey简单叠加两个视频"""
        colorkey_hex = f"0x{colorkey_value:06X}"
        cmd = f'ffmpeg -i "{base_video}" -i "{overlay_video}" -filter_complex "[1:v]colorkey={colorkey_hex}:{colorkey_similarity}:{colorkey_blend}[ckout];[0:v][ckout]overlay[out]" -map "[out]" -c:v libx264 -pix_fmt yuv420p "{output_video}"'
        print(f"执行命令: {cmd}")
        os.system(cmd)
        return output_video if os.path.exists(output_video) else base_video

    def _check_path_existence(self, name, path):
        """检查路径是否存在，并输出详细诊断信息"""
        if not path:
            print(f"警告: {name}路径为空")
            return
            
        # 检查原始路径
        if os.path.exists(path):
            print(f"{name}路径存在: {path}")
            return
            
        # 尝试转换路径格式
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
                print(f"{name}路径存在(使用替代路径): {alt_path}")
                return
                
        # 如果所有尝试都失败，输出详细错误
        print(f"错误: 无法找到{name}: {path}")
        print(f"当前工作目录: {os.getcwd()}")
        print(f"尝试过的替代路径:")
        for alt_path in alternative_paths:
            print(f"  - {alt_path} (存在: {os.path.exists(alt_path)})")
            
        # 检查目录内容
        if os.path.dirname(path) and os.path.exists(os.path.dirname(path)):
            print(f"目录 {os.path.dirname(path)} 的内容:")
            for item in os.listdir(os.path.dirname(path)):
                print(f"  - {item}")
                
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


if __name__ == "__main__":
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description="3D模型转视频工具")
    
    parser.add_argument('--output_dir', type=str, default=OUT_PATH,
                        help="输出目录")
    
    subparsers = parser.add_subparsers(dest='command', help='子命令')
    
    # 渲染单一模型
    render_parser = subparsers.add_parser('render', help='渲染单一模型')
    render_parser.add_argument('body_file', type=str, help='身体模型文件路径')
    render_parser.add_argument('garment_file', type=str, help='服装模型文件路径')
    render_parser.add_argument('--output', type=str, help='输出图像路径')
    render_parser.add_argument('--flip_vertical', action='store_true', help='垂直翻转模型')
    
    # 渲染360度旋转视频
    rotate_parser = subparsers.add_parser('rotate', help='渲染360度旋转视频')
    rotate_parser.add_argument('body_file', type=str, help='身体模型文件路径')
    rotate_parser.add_argument('garment_file', type=str, help='服装模型文件路径')
    rotate_parser.add_argument('--output', type=str, help='输出视频路径')
    rotate_parser.add_argument('--frames', type=int, default=60, help='总帧数')
    rotate_parser.add_argument('--fps', type=int, default=30, help='视频帧率')
    rotate_parser.add_argument('--flip_vertical', action='store_true', help='垂直翻转模型')
    
    # 从模型目录渲染视频
    video_parser = subparsers.add_parser('video', help='从模型目录渲染视频')
    video_parser.add_argument('models_dir', type=str, help='包含模型的目录路径')
    video_parser.add_argument('--output', type=str, help='输出视频路径')
    video_parser.add_argument('--fps', type=int, default=30, help='视频帧率')
    video_parser.add_argument('--quality', type=str, default='medium', 
                            choices=['low', 'medium', 'high'], help='视频质量')
    video_parser.add_argument('--start_frame', type=int, default=0, help='起始帧')
    video_parser.add_argument('--end_frame', type=int, help='结束帧')
    video_parser.add_argument('--person_id', type=int, help='只处理特定人物ID')
    video_parser.add_argument('--rotation', action='store_true', help='旋转模型')
    video_parser.add_argument('--rotation_speed', type=float, default=1.0, help='旋转速度')
    video_parser.add_argument('--flip_vertical', action='store_true', help='垂直翻转模型')
    
    # 叠加到原始视频
    combine_parser = subparsers.add_parser('combine', help='将3D模型叠加到原始视频')
    combine_parser.add_argument('original_video', type=str, help='原始视频路径')
    combine_parser.add_argument('models_dir', type=str, help='包含模型的目录路径')
    combine_parser.add_argument('hybrik_result', type=str, help='HybrIK结果文件路径')
    combine_parser.add_argument('--output', type=str, help='输出视频路径')
    combine_parser.add_argument('--start_frame', type=int, default=0, help='起始帧')
    combine_parser.add_argument('--end_frame', type=int, help='结束帧')
    combine_parser.add_argument('--person_id', type=int, help='只处理特定人物ID')
    combine_parser.add_argument('--alpha', type=float, default=0.7, help='透明度混合系数')
    
    # 自动对齐和叠加视频
    auto_align_parser = subparsers.add_parser('autoalign', help='自动将3D模型与原始视频中的人物对齐并叠加')
    auto_align_parser.add_argument('original_video', type=str, help='原始视频路径')
    auto_align_parser.add_argument('models_dir', type=str, help='包含模型的目录路径')
    auto_align_parser.add_argument('--hybrik_result', type=str, help='HybrIK结果文件路径(可选)')
    auto_align_parser.add_argument('--output', type=str, help='输出视频路径')
    auto_align_parser.add_argument('--start_frame', type=int, default=0, help='起始帧')
    auto_align_parser.add_argument('--end_frame', type=int, help='结束帧')
    auto_align_parser.add_argument('--person_ids', type=str, help='要处理的人物ID，用逗号分隔(如"0,1")')
    auto_align_parser.add_argument('--no_flip', action='store_true', help='不要垂直翻转模型(默认会翻转以修正方向)')
    
    args = parser.parse_args()
    
    # 创建渲染器
    renderer = ModelRenderer(output_dir=args.output_dir)
    
    # 执行命令
    if args.command == 'render':
        renderer.render_model_frame(args.body_file, args.garment_file, args.output, flip_vertical=args.flip_vertical)
        
    elif args.command == 'rotate':
        # 创建临时目录
        temp_dir = os.path.join(args.output_dir, f"temp_rotation_{int(time.time())}")
        os.makedirs(temp_dir, exist_ok=True)
        
        # 渲染每个角度
        frame_images = []
        for i in range(args.frames):
            angle = (i / args.frames) * 360.0
            output_image = os.path.join(temp_dir, f"frame_{i:04d}.png")
            image_path = renderer.render_rotated_model(args.body_file, args.garment_file, angle, output_image, flip_vertical=args.flip_vertical)
            frame_images.append(image_path)
        
        # 设置默认输出视频路径
        if args.output is None:
            base_name = os.path.basename(args.body_file).replace("body_", "")
            args.output = os.path.join(args.output_dir, f"rotation_{base_name.replace('.ply', '.mp4')}")
        
        # 使用ffmpeg创建视频
        try:
            # 构建ffmpeg命令
            frames_pattern = os.path.join(temp_dir, "frame_%04d.png")
            # 添加-vf scale选项，强制宽高为偶数
            cmd = f"ffmpeg -y -framerate {args.fps} -i \"{frames_pattern}\" -vf \"scale=trunc(iw/2)*2:trunc(ih/2)*2\" -c:v h264 -pix_fmt yuv420p \"{args.output}\""
            
            print(f"正在创建旋转视频: {args.output}")
            print(f"执行命令: {cmd}")
            
            # 执行命令
            os.system(cmd)
            
            if os.path.exists(args.output):
                print(f"成功创建旋转视频: {args.output}")
            else:
                print(f"视频创建失败: {args.output}")
                
        except Exception as e:
            print(f"创建旋转视频时出错: {e}")
        
    elif args.command == 'video':
        renderer.render_models_to_video(
            args.models_dir,
            args.output,
            fps=args.fps,
            quality=args.quality,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            person_id=args.person_id,
            rotation=args.rotation,
            rotation_speed=args.rotation_speed,
            flip_vertical=args.flip_vertical
        )
        
    elif args.command == 'combine':
        renderer.render_models_with_original_video(
            args.original_video,
            args.models_dir,
            args.hybrik_result,
            args.output,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            person_id=args.person_id,
            alpha=args.alpha
        )
    
    elif args.command == 'autoalign':
        # 解析人物ID列表
        person_ids = None
        if args.person_ids:
            person_ids = [int(pid.strip()) for pid in args.person_ids.split(',')]
            
        renderer.auto_align_videos(
            args.original_video,
            args.models_dir,
            args.output,
            args.hybrik_result,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            person_ids=person_ids,
            flip_vertical=not args.no_flip
        )
        
    else:
        parser.print_help() 