"""
واجهة شات بسيطة (Streamlit) للتحدث مع مساعد المستندات القائم على RAG.
"""
import streamlit as st

from api_client import API_BASE_URL, APIError, ask_question, check_health

st.set_page_config(page_title="مساعد المستندات الذكي", page_icon="📚", layout="centered")

st.title("📚 مساعد المستندات (RAG)")
st.caption(f"متصل بالخادم على: {API_BASE_URL}")

if not check_health():
    st.error("⚠️ لا يمكن الاتصال بالـ backend. تأكد إن FastAPI شغّال على uvicorn app.main:app")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.caption("📎 المصادر: " + ", ".join(msg["sources"]))

question = st.chat_input("اكتب سؤالك عن المستندات هنا...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("جاري البحث في المستندات وتوليد الإجابة..."):
            try:
                result = ask_question(question)
                st.markdown(result["answer"])
                if result.get("sources"):
                    st.caption("📎 المصادر: " + ", ".join(result["sources"]))
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result.get("sources", []),
                    }
                )
            except APIError as exc:
                st.error(str(exc))
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"❌ {exc}", "sources": []}
                )
