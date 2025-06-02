import os
import cv2
from typing import List

# COORDINATE_CONFIG = [
#     {
#         "NAME": (40, 770, 500, 825)
#     },
# ]

COORDINATE_CONFIG = [
    {
        "SA1": (115, 922, 635, 992),
        "SA2": (115, 822, 635, 892)
    },
]

class ImageProcessor:
    def _process_image(self, img_path: str, output_dir: str) -> List[str]:
        """裁剪图片并保存到指定目录，返回保存的文件路径列表"""
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图片：{img_path}")
            return []

        base_name = os.path.splitext(os.path.basename(img_path))[0]
        saved_paths = []

        # 使用COORDINATE_CONFIG2作为裁剪区域配置
        for group_idx, group in enumerate(COORDINATE_CONFIG, 1):
            for field, area in group.items():
                x1, y1, x2, y2 = area
                cropped = img[y1:y2, x1:x2]
                
                # 生成文件名并保存
                new_name = f"{base_name}_group{group_idx}_{field}.png".replace(" ", "_")
                new_path = os.path.abspath(os.path.join(output_dir, new_name))
                cv2.imwrite(new_path, cropped)
                saved_paths.append(new_path)
                
        return saved_paths

def main(input_dir: str, output_dir: str):
    """主处理函数：遍历输入目录，处理所有图片"""
    processor = ImageProcessor()
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_dir, filename)
            saved_files = processor._process_image(img_path, output_dir)
            print(f"已处理 {filename}，生成 {len(saved_files)} 个裁剪图片")

if __name__ == "__main__":
    INPUT_DIR = "D:\\PCR\\OCR\\duihua"
    OUTPUT_DIR = "D:\\PCR\\OCR\\duihua_"
    main(INPUT_DIR, OUTPUT_DIR)