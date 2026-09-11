# 📚 RAG Document Assistant — مساعد مذكرات تعلم الآلة

مساعد ذكي يجيب على أسئلتك بالاعتماد فقط على مجموعة مذكرات دراسية في أساسيات
تعلم الآلة (Machine Learning)، باستخدام تقنية **RAG (Retrieval-Augmented
Generation)** مع نموذج لغوي محلي عبر **Ollama** — بدون أي اتصال بالإنترنت وقت
التشغيل، وبدون هلوسة (Hallucination) لأن كل إجابة مبنية على مصدر حقيقي.

> مشروع تخرج — Level 2 Summer Training | **Core Track** (نصوص فقط)

---

## 🖇️ نظرة عامة (Overview)

النظام مقسّم لثلاثة أجزاء رئيسية:

1. **Notebook** (`notebooks/rag_pipeline.ipynb`) — يبني خط الأنابيب: تحميل
   المستندات → تقطيعها → توليد Embeddings → تخزينها في Chroma → اختبار
   الاسترجاع → تقييم → تصدير الـ vector store جاهز للـ backend.
2. **Backend** (`backend/`) — FastAPI يحمّل الـ vector store مرة واحدة عند
   الإقلاع، ويعرض `POST /query` و `GET /health`.
3. **Frontend** (`frontend/`) — واجهة شات بـ Streamlit تكلم الـ backend وتعرض
   الإجابة مع مصادرها.

## 🏗️ المخطط المعماري (Architecture)

```
┌─────────────┐      HTTP       ┌──────────────────┐      ┌──────────────┐
│  Streamlit   │ ───────────────▶│   FastAPI Backend │─────▶│  Chroma       │
│  Frontend    │◀─────────────── │  (retrieval +     │      │  Vector Store │
│  (chat UI)   │   answer+sources│   generation)     │◀─────│  (persisted)  │
└─────────────┘                 └────────┬──────────┘      └──────────────┘
                                          │
                                          ▼
                                  ┌───────────────┐
                                  │ Ollama (local) │
                                  │ LLM (llama3.2) │
                                  └───────────────┘
```

## 🧰 التقنيات المستخدمة (Tech Stack)

| الطبقة | التقنية |
|---|---|
| اللغة | Python 3.10+ |
| Embeddings | `sentence-transformers` (`paraphrase-multilingual-MiniLM-L12-v2`) |
| Vector DB | ChromaDB (persisted محليًا) |
| LLM | Ollama (`llama3.2`, محلي بالكامل) |
| Backend | FastAPI + Pydantic + Uvicorn |
| Frontend | Streamlit |
| الاختبارات | Pytest + FastAPI TestClient |

## 📁 هيكل المشروع (Project Structure)

```
rag-assistant-project/
├── notebooks/
│   └── rag_pipeline.ipynb        # بناء وتقييم خط أنابيب RAG بالكامل
├── sample_data/                  # مذكرات تعلم الآلة (7 ملفات .md)
├── backend/
│   ├── app/
│   │   ├── main.py                # نقطة الدخول + lifespan + CORS
│   │   ├── api/routes/query.py    # /health و /query
│   │   ├── core/config.py         # الإعدادات من .env
│   │   ├── schemas/query.py       # نماذج الطلب/الاستجابة
│   │   ├── services/retrieval.py  # تحميل الـ vector store + الاسترجاع
│   │   ├── services/generation.py # بناء الـ prompt + استدعاء Ollama
│   │   └── services/embeddings.py # دالة الـ embeddings المشتركة
│   ├── data/vector_store/         # يُملأ من النوتبوك (Phase 2.7)
│   ├── tests/test_query.py
│   ├── requirements.txt
│   ├── .env.example
│   └── Dockerfile
├── frontend/
│   ├── app.py
│   ├── api_client.py
│   ├── .env.example
│   └── requirements.txt
├── .gitignore
└── README.md
```

## 🗂️ الدومين والبيانات (Domain & Data)

الدومين المُختار: **مذكرات دراسية في أساسيات تعلم الآلة** — 7 ملفات Markdown
تغطي: دوال التفعيل، الـ Overfitting والتنظيم، الانحدار التدريجي، مقاييس
التقييم، تقسيم البيانات، التعلم بإشراف/بدون إشراف، وأساسيات RAG نفسها.
كل الملفات نصية قابلة للاستخراج المباشر (لا تحتاج OCR).

> يمكنك استبدال محتوى `sample_data/` بأي مستندات (PDF/Markdown) في أي دومين
> تختاره، ثم إعادة تشغيل النوتبوك — الكود عام ولا يفترض دومينًا محددًا.

## ⚙️ الإعداد والتشغيل (Setup)

### المتطلبات
- Python 3.10+
- [Ollama](https://ollama.com) مثبّت محليًا
- Git

### 1) تجهيز Ollama
```bash
ollama pull llama3.2
ollama serve   # يترك يعمل في نافذة طرفية منفصلة
```

### 2) بناء الـ Vector Store (النوتبوك)
```bash
cd notebooks
pip install jupyter pandas numpy chromadb sentence-transformers pypdf python-dotenv ollama
jupyter notebook rag_pipeline.ipynb
```
شغّل كل الخلايا (Kernel → Restart & Run All). للتسليم الفعلي اجعل المتغير
`OFFLINE_DEMO = False` في أول خلية كود (يحتاج إنترنت مرة واحدة لتحميل نموذج
الـ embeddings)، ثم تأكد أن `pipeline_config.json` يطابق الـ vector store.
النسخة المرفقة حاليًا مبنية بوضع offline demo لتعمل بدون تنزيل نموذج embeddings؛
الـ backend يقرأ `pipeline_config.json` ويستخدم نفس الطريقة تلقائيًا.

### 3) تشغيل الـ Backend
```bash
cd backend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```
افتح `http://localhost:8000/docs` وجرّب `/query` من Swagger UI.

اختبارات الـ backend:
```bash
pytest -v
```

### 4) تشغيل الـ Frontend
```bash
cd frontend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```
افتح الرابط اللي هيظهر في الطرفية (عادة `http://localhost:8501`)، واسأل سؤال حقيقي وشوف الإجابة المؤكدة بمصادرها.

## 🔐 متغيرات البيئة (Environment Variables)

**Backend (`.env`):**

| المتغير | الوصف | القيمة الافتراضية |
|---|---|---|
| `OLLAMA_MODEL` | اسم نموذج Ollama | `llama3.2` |
| `OLLAMA_HOST` | عنوان خادم Ollama | `http://localhost:11434` |
| `VECTOR_STORE_PATH` | مسار الـ vector store | `./data/vector_store` |
| `COLLECTION_NAME` | اسم الـ collection في Chroma | `ml_notes` |
| `EMBEDDING_MODEL` | نموذج الـ embeddings (لازم يطابق النوتبوك) | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| `TOP_K` | عدد القطع المسترجعة لكل سؤال | `4` |
| `FRONTEND_ORIGIN` | أصل الفرونت إند المسموح (CORS) | `http://localhost:8501` |
| `OFFLINE_DEMO_EMBEDDINGS` | وضع بديل بدون إنترنت؛ يُطابق artifact المصدّر تلقائيًا | `False` |

**Frontend (`.env`):**

| المتغير | الوصف | القيمة الافتراضية |
|---|---|---|
| `API_BASE_URL` | رابط الـ backend | `http://localhost:8000` |

## 📡 توثيق الـ API

### `GET /health`
```bash
curl http://localhost:8000/health
```
```json
{"status": "ok"}
```

### `POST /query`
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "إيه هي دالة ReLU وليه بتتستخدم كتير؟"}'
```
```json
{
  "answer": "دالة ReLU تُخرج القيمة نفسها إذا كانت موجبة وصفر إذا كانت سالبة... [مصدر: 01_activation_functions.md]",
  "sources": ["01_activation_functions.md"]
}
```

## 📊 نتائج التقييم (من Phase 2.6)

على 10 أسئلة اختبار ضد الـ corpus (7 مستندات، 16 قطعة)، الجدول الكامل
والتحليل موجودين في `notebooks/rag_pipeline.ipynb` (قسم Evaluation). ملخص:

- تم اختبار الاسترجاع (Retrieval) على أسئلة تغطي كل المستندات السبعة.
- الأسئلة ذات المفردات التقنية الفريدة (مثل "ReLU"، "F1-Score") كان استرجاعها
  دقيقًا جدًا.
- الحالات الأصعب كانت الأسئلة اللي بتلامس مفاهيم مشتركة بين أكثر من مستند
  (مثل التقييم مقابل تقسيم البيانات) — تم تخفيفها باقتراح تصغير حجم القطعة
  وزيادة `top_k`.
- **ملاحظة تنفيذية:** النوتبوك تم تنفيذه فعليًا في هذا التسليم بوضع
  `OFFLINE_DEMO=True` (بديل بدون إنترنت). قبل التسليم النهائي شغّله مع
  `OFFLINE_DEMO=False` وأعد تصدير الـ vector store للحصول على embeddings دلالية.

## 🖼️ لقطات الشاشة (Screenshots)

> أضف سكرين شوت من: (1) Swagger UI لـ `/query`، (2) واجهة Streamlit وهي
> تعرض سؤالًا وإجابة مع مصادرها، بعد تشغيل Ollama واختبار التدفق الكامل.

## ⚠️ ملاحظات مهمة

- لو غيّرت `EMBEDDING_MODEL` في الـ backend، **لازم** تعيد تشغيل النوتبوك
  بنفس الاسم بالظبط، وإلا هتحصل على نتائج استرجاع خاطئة (الفضاء المتجهي
  مختلف).
- لا تُلزم نفسك بترتيب تشغيل خلايا معين في النوتبوك غير الترتيب الطبيعي —
  تم اختباره بالكامل عبر `Kernel → Restart & Run All`.
- لا يتم رفع `.env` أو الـ vector store الكبير أو الداتا الخام لـ GitHub (راجع
  `.gitignore`).
