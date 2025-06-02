import os
import cv2
from typing import List, Dict
from module.ocr.ocr import Digit

COORDINATE_CONFIG = [
    {
        "Power": (376, 736, 455, 767),
        "Ranking": (70, 830, 112, 855),
        "CommanderLevel": (72, 801, 111, 817),
        "SynchroLevel": (392, 836, 414, 855)
    },
    {
        "Power": (376, 886, 455, 917),
        "Ranking": (70, 980, 112, 1005),
        "CommanderLevel": (72, 951, 111, 967),
        "SynchroLevel": (392, 986, 414, 1005)
    },
    {
        "Power": (376, 1036, 455, 1067),
        "Ranking": (70, 1130, 112, 1155),
        "CommanderLevel": (72, 1101, 111, 1117),
        "SynchroLevel": (392, 1136, 414, 1155)
    }
]

COORDINATE_CONFIG2 = [
    {
        "Power": (395, 650, 470, 675),
        "Ranking": (85, 765, 120, 790),
        "CommanderLevel": (74, 733, 116, 750),
        "SynchroLevel": (308, 779, 329, 797)
    },
    {
        "Power": (395, 830, 470, 855),
        "Ranking": (85, 945, 120, 970),
        "CommanderLevel": (74, 911, 116, 928),
        "SynchroLevel": (308, 957, 329, 976)
    },
    {
        "Power": (395, 1010, 470, 1035),
        "Ranking": (85, 1125, 120, 1150),
        "CommanderLevel": (74, 1089, 116, 1106),
        "SynchroLevel": (308, 1137, 329, 1155)
    }
]

FIELD_LETTERS = {
    "Power": (107, 107, 107),
    "Ranking": (107, 107, 107),
    "CommanderLevel": (222, 222, 222),
    "SynchroLevel": (255, 255, 255)
}

class ImageProcessor:
    def _process_image(self, img_path: str, output_dir: str) -> List[str]:
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图片：{img_path}")
            return []

        base_name = os.path.splitext(os.path.basename(img_path))[0]
        results = []

        for group_idx, group in enumerate(COORDINATE_CONFIG2, 1):
            for field, area in group.items():
                x1, y1, x2, y2 = area
                cropped = img[y1:y2, x1:x2]
                
                # 修改这里：强制使用下划线并生成绝对路径
                new_name = f"{base_name}_group{group_idx}_{field}.png".replace(" ", "_")
                new_path = os.path.abspath(os.path.join(output_dir, new_name))
                cv2.imwrite(new_path, cropped)
                
                # OCR识别
                value = self._recognize_field(area, img_path)
                new_path = new_path.replace(" ", "_")
                results.append(f"{new_path} {value}")  # 保持单个空格分隔

        return results

    def _recognize_field(self, area: tuple, oldpath: str) -> int:        
        digit_engine = Digit(
            [area],
            name="DynamicField",
            #letter=letter,
            threshold=128,
            lang="cnocr_num"
        )
        img = cv2.imread(oldpath)
        return int(digit_engine.ocr(img))

def main(input_dir: str, output_dir: str, result_path: str):
    processor = ImageProcessor()
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = []
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_dir, filename)
            results = processor._process_image(img_path, output_dir)
            all_results.extend(results)
    
    # 写入结果文件
    with open(result_path, 'w', encoding='utf-8') as f:
        for line in all_results:
            f.write(f"{line}\n")

if __name__ == "__main__":
    INPUT_DIR = "D:\\PCR\\CnOCR-2.3.1\\img\\full2"
    OUTPUT_DIR = "D:\\PCR\\CnOCR-2.3.1\\img\\sp2"
    RESULT_FILE = "D:\\PCR\\CnOCR-2.3.1\\img\\results2.txt"
    main(INPUT_DIR, OUTPUT_DIR, RESULT_FILE)
    