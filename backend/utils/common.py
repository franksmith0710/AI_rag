from typing import Any, Dict, Optional
from datetime import datetime


class ApiResponse:
    @staticmethod
    def success(data: Any = None, message: str = "success") -> Dict:
        return {
            "code": 200,
            "message": message,
            "data": data
        }

    @staticmethod
    def error(message: str, code: int = 500, data: Any = None) -> Dict:
        return {
            "code": code,
            "message": message,
            "data": data
        }


def get_current_time() -> datetime:
    return datetime.now()


def format_time(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")