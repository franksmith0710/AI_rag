"""
流式输出测试 - 通过 API 测试
"""
import requests
import json
import time

API_URL = "http://localhost:8000/api/chat"
LOGIN_URL = "http://localhost:8000/api/auth/login"

def get_token():
    """登录获取 token"""
    response = requests.post(LOGIN_URL, json={
        "username": "admin",
        "password": "admin123"
    })
    if response.status_code == 200:
        return response.json().get("access_token")
    return None


def test_current_api():
    """测试当前 API 是否支持流式"""
    print("=" * 60)
    print("Test: Current API Streaming Support")
    print("=" * 60)

    token = get_token()
    if not token:
        print("Login failed!")
        return

    print(f"Token: {token[:30]}...")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    data = {
        "session_id": 1,
        "message": "What is AI?"
    }

    start_time = time.time()

    try:
        response = requests.post(
            API_URL,
            headers=headers,
            json=data,
            stream=True,
            timeout=60
        )

        print(f"Status: {response.status_code}")
        print(f"Content-Type: {response.headers.get('Content-Type')}")
        transfer_enc = response.headers.get('Transfer-Encoding')
        print(f"Transfer-Encoding: {transfer_enc}")

        if transfer_enc == 'chunked':
            print("\n[OK] Streaming (chunked)")
            chunks = []
            for chunk in response.iter_content(chunk_size=None):
                if chunk:
                    chunks.append(chunk)
                    print(f"Chunk: {len(chunk)} bytes")
            full = b"".join(chunks)
            print(f"\nTotal: {len(full)} bytes")
            print(f"Preview: {full[:200]}")
        else:
            print("\n[FAIL] NOT streaming (one-shot)")
            content = response.text
            print(f"Length: {len(content)} chars")
            print(f"Preview: {content[:300]}")

        print(f"\nTime: {time.time() - start_time:.2f}s")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    test_current_api()