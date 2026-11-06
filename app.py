import re
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader
from docx import Document

st.set_page_config(page_title="محلل السيرة الذاتية الذكي", page_icon="📄", layout="wide")
st.markdown("""<style>
/* Keep the app right-to-left without overriding Streamlit's theme colors. */
[data-testid="stAppViewContainer"] { direction: rtl; }
[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] label,
[data-testid="stAppViewContainer"] h1,
[data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3 { text-align: right; }
</style>""", unsafe_allow_html=True)
st.title("📄 محلل السيرة الذاتية الذكي")
st.caption("مشروع تعليمي باستخدام Python و NLP و TF-IDF — إعداد ألين عيد البقمي")
st.info("النتيجة مقياس تشابه نصي وليست قرار توظيف أو حكمًا على كفاءة المرشح. لا ترفعي بيانات حساسة.")

SKILLS = ["python","sql","excel","power bi","machine learning","deep learning","nlp","tensorflow","pytorch","scikit-learn","data analysis","pandas","numpy","git","docker","aws","communication","leadership","project management","تحليل البيانات","تعلم الآلة","الذكاء الاصطناعي","التواصل","إدارة المشاريع","اللغة الإنجليزية","خدمة العملاء","مايكروسوفت إكسل","بايثون"]

def extract_text(upload):
    if upload is None:
        return ""
    if upload.name.lower().endswith(".pdf"):
        return "\n".join(page.extract_text() or "" for page in PdfReader(upload).pages)
    if upload.name.lower().endswith(".docx"):
        return "\n".join(p.text for p in Document(upload).paragraphs)
    return upload.read().decode("utf-8", errors="replace")

def normalize(text):
    text = text.lower().replace("أ","ا").replace("إ","ا").replace("آ","ا").replace("ى","ي")
    return re.sub(r"\s+", " ", text).strip()

def skills_in(text):
    normalized = normalize(text)
    return {s for s in SKILLS if re.search(r"(?<!\w)" + re.escape(normalize(s)) + r"(?!\w)", normalized)}

left, right = st.columns(2)
with left:
    uploaded = st.file_uploader("ارفعي السيرة الذاتية (PDF أو DOCX أو TXT)", type=["pdf","docx","txt"])
    cv_manual = st.text_area("أو الصقي نص السيرة الذاتية هنا", height=220)
with right:
    job = st.text_area("الصقي وصف الوظيفة ومتطلباتها", height=320, placeholder="مثال: نبحث عن محلل بيانات يجيد Python و SQL و Excel وتحليل البيانات...")

if st.button("🔍 تحليل السيرة الذاتية", type="primary", use_container_width=True):
    try:
        cv = (extract_text(uploaded) if uploaded else "") + "\n" + cv_manual
        if len(normalize(cv)) < 20 or len(normalize(job)) < 20:
            st.error("أدخلي سيرة ذاتية ووصف وظيفة يحتوي كل منهما على 20 حرفًا على الأقل.")
        else:
            vec = TfidfVectorizer(analyzer="char", ngram_range=(2,4), min_df=1, sublinear_tf=True)
            matrix = vec.fit_transform([normalize(cv), normalize(job)])
            score = float(cosine_similarity(matrix[0], matrix[1])[0,0]) * 100
            cv_skills = skills_in(cv)
            job_skills = skills_in(job)
            matching = sorted(cv_skills & job_skills)
            missing = sorted(job_skills - cv_skills)
            st.metric("نسبة التشابه النصي مع وصف الوظيفة", f"{score:.1f}%")
            st.progress(min(100, max(0, int(score))))
            st.subheader("✅ المهارات المشتركة")
            st.write("، ".join(matching) if matching else "لم تُكتشف مهارات مشتركة من القائمة المحددة.")
            st.subheader("📚 مهارات مذكورة في الوظيفة وغير موجودة نصيًا في السيرة")
            st.write("، ".join(missing) if missing else "لم تُكتشف فجوات ضمن قائمة المهارات المحددة.")
            st.subheader("💡 توصيات للتحسين")
            st.write("اكتبي الخبرات والإنجازات بأمثلة قابلة للقياس، وأضيفي المهارات التي تتقنينها بالفعل فقط، وراجعي متطلبات الوظيفة يدويًا.")
            st.caption("تنبيه: قد يفوت التحليل مرادفات أو مهارات ضمنية؛ الملفات المصورة تحتاج OCR غير مدعوم في هذه النسخة. لا يُستخدم هذا التقييم لرفض المتقدمين آليًا.")
    except Exception as exc:
        st.error(f"تعذر قراءة الملف أو تحليل النص: {exc}")
