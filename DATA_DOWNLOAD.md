# TailorNet 数据和模型下载指南

由于数据文件较大（约32GB），我们使用网盘链接的方式提供下载。

## 下载链接

### 1. 完整数据包（推荐）
- **百度网盘**: [链接](YOUR_BAIDU_LINK) 提取码: xxxx
- **Google Drive**: [链接](YOUR_GOOGLE_DRIVE_LINK)
- **OneDrive**: [链接](YOUR_ONEDRIVE_LINK)

### 2. 分部分下载
如果你只需要部分数据，可以选择性下载：

#### tailornet_data（约30GB）
包含所有的训练数据：
- **百度网盘**: [链接](YOUR_BAIDU_LINK_DATA) 提取码: xxxx
- **Google Drive**: [链接](YOUR_GOOGLE_DRIVE_LINK_DATA)

#### tailornet_weights（约100MB）
包含预训练的模型权重：
- **百度网盘**: [链接](YOUR_BAIDU_LINK_WEIGHTS) 提取码: xxxx
- **Google Drive**: [链接](YOUR_GOOGLE_DRIVE_LINK_WEIGHTS)

#### results.json（约6MB）
- **百度网盘**: [链接](YOUR_BAIDU_LINK_RESULTS) 提取码: xxxx
- **Google Drive**: [链接](YOUR_GOOGLE_DRIVE_LINK_RESULTS)

## 下载后的使用方法

1. 下载完成后，请将文件解压到项目根目录
2. 确保目录结构如下：
```
TailorNet/
├── tailornet_data/
│   ├── t-shirt_female/
│   ├── t-shirt_male/
│   └── skirt_female/
├── tailornet_weights/
│   ├── t-shirt_female_weights/
│   ├── t-shirt_male_weights/
│   └── skirt_female_weights/
├── results.json
└── ... (其他代码文件)
```

## 验证下载

下载完成后，可以运行以下命令验证文件完整性：
```bash
python verify_data.py
```

## 注意事项

1. 请确保有足够的硬盘空间（至少40GB）
2. 下载可能需要较长时间，请保持网络稳定
3. 如果下载中断，大部分网盘都支持断点续传

## 问题反馈

如果下载链接失效或遇到其他问题，请通过以下方式联系：
- GitHub Issues: [创建Issue](https://github.com/YOUR_USERNAME/TailorNet/issues)
- Email: your_email@example.com 