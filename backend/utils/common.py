from typing import Any, Dict


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