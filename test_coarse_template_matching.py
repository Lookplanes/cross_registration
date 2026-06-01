import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import utils

def load_image_as_array(path):
    """加载图像并转换为灰度归一化数组"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"无法找到图像文件: {path}")
    img = Image.open(path).convert('L')
    img_array = np.array(img).astype(np.float32)
    # 归一化到 [0, 1]
    img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-8)
    return img_array

def save_result_plot(image1, template, pred_coords, save_dir, filename="result.png"):
    """
    生成拼图结果并保存
    :param image1: 搜索的大图
    :param template: 模板图 (image2)
    :param pred_coords: 预测的坐标 (y, x)
    :param save_dir: 保存目录
    :param filename: 文件名
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    pred_y, pred_x = pred_coords
    th, tw = template.shape
    
    # 从 image1 中截取匹配到的区域进行对比
    matched_patch = image1[pred_y : pred_y + th, pred_x : pred_x + tw]

    # 创建画布 (1行3列)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. 绘制大图并在匹配位置画框
    axes[0].imshow(image1, cmap='gray')
    rect = plt.Rectangle((pred_x, pred_y), tw, th, edgecolor='red', facecolor='none', linewidth=2)
    axes[0].add_patch(rect)
    axes[0].set_title(f"Search Image (Matched Rect)")
    
    # 2. 绘制模板图 (image2)
    axes[1].imshow(template, cmap='gray')
    axes[1].set_title(f"Template (image2) {tw}x{th}")
    
    # 3. 绘制从 image1 截取的对应区域
    axes[2].imshow(matched_patch, cmap='gray')
    axes[2].set_title(f"Actual Cropped Area from Image1")

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.05, right=0.95)
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"结果已保存至: {save_path}")

def main():
    # --- 配置区域 ---
    image1_path = "/data2/xujr/idr_data/dataset_processed/idr0003-breker-plasticity-512/screenA_test/channel_1/DTT p1 [Well 65_ Field 1 (Spot 193)]_z0_t0_p0.png"    # 大图路径
    image2_path = None               # 模板路径，None 则从 image1 随机截取
    save_dir = "results/matching_results"      # 结果保存目录
    template_size = (100, 100)       # 随机截取时的尺寸 (h, w)
    
    seed = 41
    rng = np.random.default_rng(seed)

    # 1. 加载或模拟 Image1
    try:
        image1 = load_image_as_array(image1_path)
    except FileNotFoundError:
        print("未找到指定 image1，生成测试数据...")
        image1 = np.random.rand(512, 512).astype(np.float32)

    # 2. 加载或截取 Image2 (Template)
    gt_y, gt_x = None, None
    if image2_path and os.path.exists(image2_path):
        image2 = load_image_as_array(image2_path)
    else:
        print("image2 为空，从 image1 随机截取...")
        h, w = image1.shape
        th, tw = template_size
        gt_y = int(rng.integers(0, h - th))
        gt_x = int(rng.integers(0, w - tw))
        image2 = image1[gt_y:gt_y+th, gt_x:gt_x+tw].copy()
        # 稍微加点干扰模拟真实情况
        image2 = image2 * 0.9 + 0.05 * rng.standard_normal(image2.shape)

    # 3. 模板匹配
    # 这里假设 utils.TemplateMatchTool.find_best_match 返回 ((y, x), score)
    (pred_y, pred_x), score = utils.TemplateMatchTool.find_best_match(
        image1, image2, stride=2
    )

    print(f"匹配位置: y={pred_y}, x={pred_x}, 分数={score:.4f}")
    if gt_y is not None:
        print(f"真值位置: y={gt_y}, x={gt_x}, 误差={abs(pred_y-gt_y) + abs(pred_x-gt_x)}")

    # 4. 保存可视化拼图
    save_result_plot(
        image1=image1, 
        template=image2, 
        pred_coords=(pred_y, pred_x), 
        save_dir=save_dir,
        filename="match_result_collage.png"
    )

if __name__ == "__main__":
    main()