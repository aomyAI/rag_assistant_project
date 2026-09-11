"""
خدمة التوليد (Generation): بناء الـ prompt من القطع المسترجعة، واستدعاء نموذج
Ollama المحلي لتوليد إجابة مبنية على السياق (Grounded Answer) مع ذكر مصادرها.
"""
from __future__ import annotations

import ollama

from app.core.config import settings
from app.utils.logging_config import logger

SYSTEM_PROMPT = (
    "أنت مساعد يجيب فقط بناءً على السياق (Context) المُعطى لك أدناه. "
    "إذا لم يكن السياق كافيًا للإجابة، قل بوضوح إنك لا تملك معلومات كافية في "
    "المستندات المتاحة. لا تستخدم أي معرفة خارج السياق المُعطى. اذكر دائمًا "
    "اسم المصدر الذي استندت إليه في نهاية إجابتك."
)

ollama_client = ollama.Client(host=settings.ollama_host)


def build_prompt(question: str, chunks: list[dict]) -> str:
    context_blocks = []
    for i, c in enumerate(chunks, start=1):
        context_blocks.append(f"[مصدر {i}: {c['source']}]\n{c['text']}")
    context = "\n\n".join(context_blocks)

    return (
        f"السياق المتاح:\n{context}\n\n"
        f"سؤال المستخدم: {question}\n\n"
        "أجب بالعربية بشكل واضح ومباشر بناءً على السياق أعلاه فقط."
    )


def generate_answer(question: str, chunks: list[dict]) -> str:
    if not chunks:
        return "لا توجد مستندات كافية في قاعدة المعرفة للإجابة على هذا السؤال."

    prompt = build_prompt(question, chunks)

    try:
        response = ollama_client.chat(
            model=settings.ollama_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return response["message"]["content"]
    except Exception as exc:  # noqa: BLE001 — نرجع رسالة واضحة بدل الانهيار
        logger.exception("Ollama generation failed")
        raise RuntimeError(
            f"فشل استدعاء نموذج Ollama ({settings.ollama_model}). "
            f"تأكد إنه شغّال محليًا (ollama serve) والموديل متسحّب. التفاصيل: {exc}"
        ) from exc
