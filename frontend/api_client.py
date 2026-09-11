"""
Wrapper بسيط للتواصل مع الـ backend API. لا يوجد أي رابط مكتوب يدويًا (hardcoded)
هنا — الرابط يُقرأ دائمًا من متغير البيئة API_BASE_URL.
"""
import os

import requests
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


class APIError(Exception):
    pass


def check_health() -> bool:
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200 and response.json().get("status") == "ok"
    except requests.RequestException:
        return False


def ask_question(question: str) -> dict:
    try:
        response = requests.post(
            f"{API_BASE_URL}/query",
            json={"question": question},
            timeout=60,
        )
        response.raise_for_status()
        return response.json()
    except requests.HTTPError as exc:
        detail = ""
        try:
            detail = exc.response.json().get("detail", "")
        except Exception:
            pass
        raise APIError(f"خطأ من الخادم ({exc.response.status_code}): {detail}") from exc
    except requests.RequestException as exc:
        raise APIError(f"تعذّر الاتصال بالخادم على {API_BASE_URL}. تأكد إنه شغّال. ({exc})") from exc
