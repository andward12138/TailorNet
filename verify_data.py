"""
验证TailorNet数据文件的完整性
"""
import os
import glob

def check_directory_structure():
    """检查必要的目录结构是否存在"""
    required_dirs = [
        'tailornet_data',
        'tailornet_data/t-shirt_female',
        'tailornet_data/t-shirt_male',
        'tailornet_weights',
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)
    
    return missing_dirs

def check_data_files():
    """检查数据文件数量"""
    file_counts = {}
    
    # 检查.npy文件
    patterns = [
        'tailornet_data/t-shirt_female/**/*.npy',
        'tailornet_data/t-shirt_male/**/*.npy',
    ]
    
    for pattern in patterns:
        files = glob.glob(pattern, recursive=True)
        key = pattern.split('/')[1]
        file_counts[f'{key}_npy_files'] = len(files)
    
    # 检查权重文件
    weight_patterns = [
        'tailornet_weights/**/*.pth.tar',
        'tailornet_weights/**/*.pth',
    ]
    
    total_weights = 0
    for pattern in weight_patterns:
        files = glob.glob(pattern, recursive=True)
        total_weights += len(files)
    
    file_counts['weight_files'] = total_weights
    
    # 检查results.json
    if os.path.exists('results.json'):
        size_mb = os.path.getsize('results.json') / (1024 * 1024)
        file_counts['results.json'] = f'存在 ({size_mb:.2f} MB)'
    else:
        file_counts['results.json'] = '缺失'
    
    return file_counts

def main():
    print("=" * 50)
    print("TailorNet 数据完整性验证")
    print("=" * 50)
    
    # 检查目录结构
    print("\n1. 检查目录结构...")
    missing_dirs = check_directory_structure()
    if missing_dirs:
        print("   ❌ 缺失以下目录:")
        for dir_path in missing_dirs:
            print(f"      - {dir_path}")
    else:
        print("   ✅ 所有必要目录都存在")
    
    # 检查文件
    print("\n2. 检查数据文件...")
    file_counts = check_data_files()
    for key, value in file_counts.items():
        print(f"   - {key}: {value}")
    
    # 总结
    print("\n" + "=" * 50)
    if missing_dirs:
        print("⚠️  警告: 某些目录缺失，请确保已正确下载并解压所有数据")
        print("请参考 DATA_DOWNLOAD.md 获取下载链接")
    else:
        print("✅ 基本目录结构完整")
        print("请检查上述文件数量是否符合预期")
    
    print("=" * 50)

if __name__ == "__main__":
    main() 