import os
import cv2
from typing import List
import numpy as np
from module.base.base import OCR_MODEL
from functools import cached_property

class FullImageOCR:
    def process_directory(self, input_dir: str) -> List[str]:
        """处理目录中的所有图片并返回识别结果"""
        results = []
        
        for filename in os.listdir(input_dir):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path)
            if img is None:
                print(f"无法读取图片：{img_path}")
                continue
            
            # 识别
            ocr_result = self.ocr_area(img, model='cnocr')
            
            # 处理OCR结果：将每个字符用空格分隔
            processed_text = self.process_ocr_result(ocr_result)
            
            # TSV格式：文件名\t处理后的文本
            result_line = f"{filename}\t{processed_text}"
            results.append(result_line)
        
        return results

    def process_ocr_result(self, ocr_result):
        """处理OCR结果：将每个字符用空格分隔"""
        if ocr_result is None:
            return ""
        
        # 提取所有文本结果
        texts = [item["text"] for item in ocr_result]
        
        # 将所有文本合并为一个字符串
        full_text = " ".join(texts)
        
        # 将每个字符用空格分隔
        return " ".join(list(full_text))

    @cached_property
    def ocr_models(self):
        return OCR_MODEL

    def ocr_area(self, image, area=None, model='cnocr'):
        result = self.ocr_models.__getattribute__(model).ocr(image, area=area)
        if len(result):
            if area:
                # 添加区域偏移
                _result = [item.copy() for item in result]
                offset = np.array([[area[0], area[1]]], dtype=np.float32)
                for item in _result:
                    item["position"] = item["position"] + offset
                return _result
            return result
        else:
            return None

def main(input_dir: str, output_file: str):
    """主处理函数"""
    processor = FullImageOCR()
    results = processor.process_directory(input_dir)
    
    # 写入TSV格式文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in results:
            f.write(f"{line}\n")
    
    print(f"处理完成！共识别 {len(results)} 张图片")
    print(f"结果已保存至: {os.path.abspath(output_file)}")
    print("结果格式: [文件名]\t[字符空格分隔的识别结果]")

if __name__ == "__main__":
    # 配置路径
    IMAGE_DIR = "D:\\PCR\\NIKKECnOCR\\train\\cn\\images"
    RESULT_FILE = "D:\\PCR\\NIKKECnOCR\\train\\cn\\ocr_results.tsv"
    
    main(IMAGE_DIR, RESULT_FILE)