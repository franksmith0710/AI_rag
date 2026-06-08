"""
OCR 图片文字识别模块
负责：用 RapidOCR ONNX 从图片中提取文字
"""

import logging
from typing import List
from core.logging_config import setup_logging

logger = setup_logging("ocr")


class OCRProcessor:
    """OCR 处理器（RapidOCR ONNX 轻量版，全局单例）

    用法:
        ocr = OCRProcessor()
        text = ocr.extract_text("path/to/image.jpg")
    """

    _engine = None  # 类级单例

    @classmethod
    def _get_engine(cls):
        if cls._engine is None:
            try:
                from rapidocr_onnxruntime import RapidOCR
                cls._engine = RapidOCR()
            except Exception as e:
                logger.error(f"RapidOCR 初始化失败: {e}")
                raise
        return cls._engine

    def extract_text(self, image_path: str) -> str:
        """从图片文件提取文字

        Args:
            image_path: 图片文件路径

        Returns:
            提取的文本内容，无文字时返回空字符串
        """
        engine = self._get_engine()
        try:
            result, elapse = engine(image_path)
            if not result:
                logger.info(f"OCR 未识别到文字: {image_path}")
                return ""

            texts: List[str] = []
            for box, text, score in result:
                if text and text.strip():
                    texts.append(text.strip())

            raw = "\n".join(texts)
            logger.info(f"OCR 完成: {len(texts)} 个文字块, 耗时 {elapse}s")
            return raw

        except Exception as e:
            logger.error(f"OCR 识别失败: {e}")
            return ""
