from pathlib import Path
import json
from urllib.parse import quote

import joblib
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models/outcome_status_model.joblib"
METRICS_PATH = ROOT / "reports/model_metrics.json"
DATA_PATH = ROOT / "data/processed/cleaned_dataset.csv"

TEXT = {
    "English": {
        "page_title": "UAE Cancer Data Explorer",
        "language": "Language / اللغة",
        "title": "UAE Cancer Data Explorer",
        "subtitle": "Educational analysis of recorded dataset labels. Not for diagnosis, treatment, or clinical decisions.",
        "provenance_warning": "This dataset has no documented source or license, and its real-world provenance is unverified. Model outputs are exploratory associations only.",
        "missing_artifacts": "Model artifacts are missing. Add the workbook at data/raw/uae_cancer_dataset_200k.xlsx (or set CANCER_DATA_PATH), then run `python train.py`.",
        "missing_data": "Cleaned data is missing. Run `python train.py` to generate it.",
        "tab_dashboard": "Interactive dashboard",
        "tab_prediction": "Try a prediction",
        "tab_details": "Model details",
        "filters": "Dashboard filters",
        "cancer_type": "Cancer type",
        "emirate": "Emirate",
        "gender": "Gender",
        "outcome": "Recorded outcome",
        "diagnosis_year": "Diagnosis year",
        "age": "Age",
        "no_records": "No records match these filters. Broaden one or more selections.",
        "matching_records": "Matching records",
        "records": "Records",
        "dataset_share": "{share} of dataset",
        "median_age": "Median age",
        "cancer_types": "Cancer types",
        "emirates": "Emirates",
        "outcome_counts": "Recorded outcome counts",
        "outcome_by_stage": "Outcome mix by cancer stage",
        "diagnosis_volume": "Diagnosis volume over time",
        "breakdown": "Explore a breakdown",
        "group_by": "Group records by",
        "view_records": "View matching records",
        "showing_records": "Showing up to 1,000 of {count:,} matching records.",
        "download_csv": "Download filtered records as CSV",
        "prediction_intro": "Adjust diagnosis-time attributes and submit to compare the model's class scores. Treatment and follow-up fields are intentionally excluded.",
        "bmi": "BMI (derived from weight and height)",
        "nationality": "Nationality",
        "stage": "Cancer stage",
        "smoking": "Smoking status",
        "comorbidity": "Recorded comorbidity",
        "ethnicity": "Ethnicity",
        "calculate": "Calculate class scores",
        "recorded_class": "Recorded class",
        "model_score": "Model score",
        "most_likely": "Most likely recorded class",
        "score_caption": "Scores are model probabilities for dataset labels, not calibrated patient risks or clinical outcomes.",
        "model_performance": "How the model performed",
        "rows_modeled": "Rows modeled",
        "accuracy": "Held-out accuracy",
        "macro_f1": "Held-out macro F1",
        "model_comparison": "Model comparison",
        "tuning_selection": "Tuning selection: {params} | CV macro F1: {score:.3f} | trials: {trials} | folds: {folds}",
        "target_caption": "The target is a recorded label at an unspecified observation time, not survival or treatment effectiveness.",
        "confusion_matrix": "Held-out confusion matrix",
    },
    "العربية": {
        "page_title": "مستكشف بيانات السرطان في الإمارات",
        "language": "اللغة / Language",
        "title": "مستكشف بيانات السرطان في الإمارات",
        "subtitle": "تحليل تعليمي للتصنيفات المسجلة في مجموعة البيانات. لا يُستخدم للتشخيص أو العلاج أو اتخاذ قرارات سريرية.",
        "provenance_warning": "لا يتوفر مصدر أو ترخيص موثق لمجموعة البيانات، كما لم يتم التحقق من مصدرها الواقعي. مخرجات النموذج استكشافية فقط.",
        "missing_artifacts": "ملفات النموذج غير موجودة. أضف ملف البيانات إلى data/raw/uae_cancer_dataset_200k.xlsx (أو عيّن CANCER_DATA_PATH)، ثم شغّل `python train.py`.",
        "missing_data": "البيانات المنظّفة غير موجودة. شغّل `python train.py` لإنشائها.",
        "tab_dashboard": "لوحة البيانات التفاعلية",
        "tab_prediction": "جرّب التنبؤ",
        "tab_details": "تفاصيل النموذج",
        "filters": "عوامل تصفية لوحة البيانات",
        "cancer_type": "نوع السرطان",
        "emirate": "الإمارة",
        "gender": "الجنس",
        "outcome": "النتيجة المسجلة",
        "diagnosis_year": "سنة التشخيص",
        "age": "العمر",
        "no_records": "لا توجد سجلات تطابق عوامل التصفية. وسّع نطاق اختيار واحد أو أكثر.",
        "matching_records": "السجلات المطابقة",
        "records": "السجلات",
        "dataset_share": "{share} من مجموعة البيانات",
        "median_age": "وسيط العمر",
        "cancer_types": "أنواع السرطان",
        "emirates": "الإمارات",
        "outcome_counts": "عدد النتائج المسجلة",
        "outcome_by_stage": "توزيع النتائج حسب مرحلة السرطان",
        "diagnosis_volume": "عدد التشخيصات بمرور الوقت",
        "breakdown": "استكشاف توزيع البيانات",
        "group_by": "تجميع السجلات حسب",
        "view_records": "عرض السجلات المطابقة",
        "showing_records": "يتم عرض ما يصل إلى 1,000 من أصل {count:,} سجل مطابق.",
        "download_csv": "تنزيل السجلات المصفاة بصيغة CSV",
        "prediction_intro": "عدّل الخصائص المسجلة وقت التشخيص ثم أرسلها لمقارنة درجات فئات النموذج. لا تُستخدم بيانات العلاج والمتابعة.",
        "bmi": "مؤشر كتلة الجسم (محسوب من الوزن والطول)",
        "nationality": "الجنسية",
        "stage": "مرحلة السرطان",
        "smoking": "حالة التدخين",
        "comorbidity": "الأمراض المصاحبة المسجلة",
        "ethnicity": "الأصل العرقي",
        "calculate": "حساب درجات الفئات",
        "recorded_class": "الفئة المسجلة",
        "model_score": "درجة النموذج",
        "most_likely": "الفئة المسجلة الأرجح",
        "score_caption": "تمثل الدرجات احتمالات النموذج لتصنيفات مجموعة البيانات، وليست مخاطر صحية معايرة أو نتائج سريرية.",
        "model_performance": "أداء النموذج",
        "rows_modeled": "عدد السجلات المستخدمة",
        "accuracy": "الدقة على بيانات الاختبار",
        "macro_f1": "مقياس F1 الكلي على بيانات الاختبار",
        "model_comparison": "مقارنة النماذج",
        "tuning_selection": "إعدادات الضبط: {params} | مقياس F1 الكلي للتحقق المتقاطع: {score:.3f} | التجارب: {trials} | الطيات: {folds}",
        "target_caption": "الهدف هو تصنيف نتيجة مسجلة في وقت ملاحظة غير محدد، وليس التنبؤ بالبقاء أو بفعالية العلاج.",
        "confusion_matrix": "مصفوفة الالتباس على بيانات الاختبار",
    },
}

VALUE_AR = {
    "Female": "أنثى", "Male": "ذكر", "Emirati": "إماراتي", "Expatriate": "مقيم",
    "Abu Dhabi": "أبوظبي", "Dubai": "دبي", "Sharjah": "الشارقة", "Ajman": "عجمان",
    "Fujairah": "الفجيرة", "Ras Al Khaimah": "رأس الخيمة", "Umm Al Quwain": "أم القيوين",
    "Breast": "الثدي", "Colorectal": "القولون والمستقيم", "Leukemia": "سرطان الدم",
    "Liver": "الكبد", "Lung": "الرئة", "Ovarian": "المبيض", "Pancreatic": "البنكرياس",
    "Prostate": "البروستاتا", "Non-Smoker": "غير مدخن", "Smoker": "مدخن",
    "Unknown": "غير معروف", "Cardiovascular Disease": "مرض قلبي وعائي", "Diabetes": "السكري",
    "Hypertension": "ارتفاع ضغط الدم", "Arab": "عربي", "African": "أفريقي",
    "East Asian": "شرق آسيوي", "European": "أوروبي", "South Asian": "جنوب آسيوي",
    "Deceased": "متوفى", "Recovered": "متعافٍ", "Under Treatment": "قيد العلاج",
    "Logistic regression": "الانحدار اللوجستي", "Decision tree": "شجرة القرار",
    "balanced": "متوازن",
    "Cancer_Type": "نوع السرطان", "Cancer_Stage": "مرحلة السرطان", "Emirate": "الإمارة",
    "Smoking_Status": "حالة التدخين", "Nationality": "الجنسية", "Outcome": "النتيجة",
    "Records": "السجلات", "Diagnoses": "التشخيصات", "Model": "النموذج",
    "Patient_ID": "معرّف المريض", "Age": "العمر", "Gender": "الجنس",
    "Diagnosis_Date": "تاريخ التشخيص", "Comorbidities": "الأمراض المصاحبة",
    "Ethnicity": "الأصل العرقي", "BMI": "مؤشر كتلة الجسم", "Recorded class": "الفئة المسجلة",
    "Model score": "درجة النموذج",
}

MEDICAL_BACKGROUND_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 440 320">
<g fill="none" stroke="#287f7a" stroke-opacity=".13" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">
<path d="M0 250h106l18-22 19 44 23-70 22 48h58l16-18 16 30 18-52 20 34h124"/>
<g transform="translate(320 45)">
<path d="M4 78V20h24V6h32v14h25v58H4zM29 78V56h30v22"/>
<path d="M15 34h5M15 45h5M70 34h5M70 45h5"/>
</g>
<circle cx="204" cy="70" r="35"/>
<circle cx="204" cy="70" r="48"/>
</g>
<g fill="#287f7a" fill-opacity=".11">
<circle cx="105" cy="140" r="3"/><circle cx="120" cy="140" r="3"/><circle cx="135" cy="140" r="3"/>
<circle cx="105" cy="155" r="3"/><circle cx="120" cy="155" r="3"/><circle cx="135" cy="155" r="3"/>
</g>
</svg>"""
MEDICAL_BACKGROUND_URL = "data:image/svg+xml," + quote(MEDICAL_BACKGROUND_SVG)

st.set_page_config(
    page_title="UAE Cancer Data Explorer | مستكشف بيانات السرطان في الإمارات",
    page_icon="📊",
    layout="wide",
)

language = st.sidebar.selectbox("Language / اللغة", ["English", "العربية"], key="language")
copy = TEXT[language]
is_arabic = language == "العربية"
direction = "rtl" if is_arabic else "ltr"
st.markdown(
    f"""<style>
    .stApp {{
        position: relative;
        isolation: isolate;
        color: #183332;
        background-color: #f6faf9;
        background-image:
            url("{MEDICAL_BACKGROUND_URL}"),
            radial-gradient(ellipse at 92% 8%, rgba(36, 157, 149, .16), transparent 28%),
            radial-gradient(ellipse at 4% 78%, rgba(221, 126, 143, .12), transparent 25%);
        background-size: 440px 320px, auto, auto;
        background-repeat: repeat, no-repeat, no-repeat;
    }}
    /* Quiet hospital cross and awareness ribbon watermarks in the page background. */
    .stApp::after {{
        content: "";
        position: fixed;
        z-index: 0;
        pointer-events: none;
        user-select: none;
    }}
    .stApp::after {{
        content: "🎗️";
        right: 18px;
        bottom: 28px;
        font-size: 170px;
        line-height: 1;
        opacity: .2;
        transform: rotate(-13deg);
        filter: saturate(.72);
    }}
    [data-testid="stAppViewContainer"] {{
        position: relative;
        z-index: 1;
        background: transparent;
    }}
    [data-testid="stHeader"] {{
        position: relative;
        z-index: 20;
        background: rgba(246, 250, 249, .78);
    }}
    [data-testid="stHeader"] button {{
        position: relative;
        z-index: 21;
        pointer-events: auto;
    }}
    html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {{
        direction: {direction};
        text-align: {'right' if is_arabic else 'left'};
    }}
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, rgba(235, 246, 244, .98), rgba(243, 248, 247, .96));
        border-inline-end: 1px solid rgba(39, 121, 116, .12);
    }}
    .block-container {{
        padding-top: 2.3rem;
        padding-bottom: 3rem;
    }}
    h1 {{
        color: #174b49;
        letter-spacing: -.035em;
    }}
    [data-testid="stMetric"] {{
        min-height: 112px;
        padding: 16px 18px;
        border: 1px solid rgba(39, 121, 116, .13);
        border-radius: 16px;
        background: rgba(255, 255, 255, .82);
        box-shadow: 0 8px 24px rgba(25, 71, 68, .045);
    }}
    [data-testid="stMetricLabel"] {{
        color: #58716f;
    }}
    [data-baseweb="tab-list"] {{
        gap: 8px;
    }}
    [data-baseweb="tab"] {{
        border-radius: 10px 10px 0 0;
    }}
    [data-testid="stAlert"] {{
        border-radius: 14px;
    }}
    @media (max-width: 700px) {{
        .stApp::after {{
            right: 8px;
            bottom: 12px;
            font-size: 118px;
            opacity: .15;
        }}
    }}
    </style>""",
    unsafe_allow_html=True,
)


def t(key, **values):
    return copy[key].format(**values)


def display_value(value):
    if is_arabic and pd.notna(value):
        return VALUE_AR.get(str(value), str(value))
    return str(value)


st.title(t("title"))
st.caption(t("subtitle"))
st.warning(t("provenance_warning"))

if not MODEL_PATH.exists() or not METRICS_PATH.exists():
    st.error(t("missing_artifacts"))
    st.stop()


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    classifier = getattr(model, "named_steps", {}).get("model")
    # Newer scikit-learn artifacts may omit this field, while 1.6's
    # predict_proba still reads it. "auto" selects multinomial probabilities
    # for this three-class lbfgs model.
    if isinstance(classifier, LogisticRegression) and not hasattr(classifier, "multi_class"):
        classifier.multi_class = "auto"
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return model, metrics


@st.cache_data(show_spinner="Loading cleaned dataset… / جارٍ تحميل البيانات المنظفة…")
def load_data():
    data = pd.read_csv(DATA_PATH, parse_dates=["Diagnosis_Date"])
    data["Diagnosis_Year"] = data["Diagnosis_Date"].dt.year
    return data


model, metrics = load_artifacts()
if not DATA_PATH.exists():
    st.error(t("missing_data"))
    st.stop()
data = load_data()

tab1, tab2, tab3 = st.tabs([t("tab_dashboard"), t("tab_prediction"), t("tab_details")])
with tab1:
    st.subheader(t("tab_dashboard"))
    with st.sidebar:
        st.header(t("filters"))
        cancer_options = sorted(data["Cancer_Type"].dropna().unique())
        cancer_filter = st.multiselect(t("cancer_type"), cancer_options, default=cancer_options,
                                       format_func=display_value)
        emirate_options = sorted(data["Emirate"].dropna().unique())
        emirate_filter = st.multiselect(t("emirate"), emirate_options, default=emirate_options,
                                        format_func=display_value)
        gender_options = sorted(data["Gender"].dropna().unique())
        gender_filter = st.multiselect(t("gender"), gender_options, default=gender_options,
                                       format_func=display_value)
        outcome_options = sorted(data["Outcome"].dropna().unique())
        outcome_filter = st.multiselect(t("outcome"), outcome_options, default=outcome_options,
                                        format_func=display_value)
        years = data["Diagnosis_Year"].dropna().astype(int)
        selected_years = st.slider(t("diagnosis_year"), int(years.min()), int(years.max()),
                                   (int(years.min()), int(years.max())))
        ages = data["Age"].dropna().astype(int)
        selected_ages = st.slider(t("age"), int(ages.min()), int(ages.max()),
                                  (int(ages.min()), int(ages.max())))

    filtered = data[
        data["Cancer_Type"].isin(cancer_filter)
        & data["Emirate"].isin(emirate_filter)
        & data["Gender"].isin(gender_filter)
        & data["Outcome"].isin(outcome_filter)
        & data["Diagnosis_Year"].between(*selected_years)
        & data["Age"].between(*selected_ages)
    ]
    if filtered.empty:
        st.info(t("no_records"))
    else:
        first, second, third, fourth = st.columns(4)
        first.metric(t("matching_records"), f"{len(filtered):,}",
                     t("dataset_share", share=f"{len(filtered) / len(data):.1%}"))
        second.metric(t("median_age"), f"{filtered.Age.median():.0f}")
        third.metric(t("cancer_types"), f"{filtered.Cancer_Type.nunique()}")
        fourth.metric(t("emirates"), f"{filtered.Emirate.nunique()}")

        left, right = st.columns(2)
        with left:
            st.markdown(f"**{t('outcome_counts')}**")
            outcome_counts = filtered["Outcome"].value_counts().rename_axis(t("outcome")).to_frame(t("records"))
            if is_arabic:
                outcome_counts.index = [display_value(value) for value in outcome_counts.index]
            st.bar_chart(outcome_counts, horizontal=True)
        with right:
            st.markdown(f"**{t('outcome_by_stage')}**")
            stage_mix = pd.crosstab(filtered["Cancer_Stage"], filtered["Outcome"], normalize="index")
            if is_arabic:
                stage_mix.index = [display_value(value) for value in stage_mix.index]
                stage_mix.columns = [display_value(value) for value in stage_mix.columns]
            st.bar_chart(stage_mix, stack="normalize")

        st.markdown(f"**{t('diagnosis_volume')}**")
        yearly = filtered.groupby("Diagnosis_Year").size().rename(t("matching_records"))
        st.line_chart(yearly)

        st.markdown(f"**{t('breakdown')}**")
        breakdown_options = ["Cancer_Type", "Cancer_Stage", "Emirate", "Smoking_Status", "Nationality"]
        breakdown = st.selectbox(t("group_by"), breakdown_options, format_func=display_value)
        breakdown_data = pd.crosstab(filtered[breakdown], filtered["Outcome"])
        breakdown_data = breakdown_data.rename_axis(index=display_value(breakdown), columns=t("outcome"))
        if is_arabic:
            breakdown_data.index = [display_value(value) for value in breakdown_data.index]
            breakdown_data.columns = [display_value(value) for value in breakdown_data.columns]
        st.bar_chart(breakdown_data)
        st.dataframe(breakdown_data, use_container_width=True)

        with st.expander(t("view_records")):
            display_cols = ["Patient_ID", "Age", "Gender", "Emirate", "Diagnosis_Date", "Cancer_Type", "Cancer_Stage", "Outcome", "Smoking_Status"]
            records = filtered[display_cols].head(1000).copy()
            if is_arabic:
                for column in display_cols:
                    if column in records.columns and records[column].dtype == "object":
                        records[column] = records[column].map(display_value)
                records = records.rename(columns=VALUE_AR)
            st.dataframe(records, use_container_width=True, height=350)
            st.caption(t("showing_records", count=len(filtered)))
            st.download_button(t("download_csv"), filtered.to_csv(index=False).encode("utf-8"),
                               file_name="filtered_cancer_records.csv", mime="text/csv")

with tab2:
    st.write(t("prediction_intro"))
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        age = col1.slider(t("age"), 18, 100, 50)
        bmi = col1.slider(t("bmi"), 10.0, 55.0, 24.0, 0.5)

        def localized_selectbox(container, label, options):
            return container.selectbox(label, options, format_func=display_value)

        gender = localized_selectbox(col2, t("gender"), ["Female", "Male"])
        nationality = localized_selectbox(col2, t("nationality"), ["Emirati", "Expatriate"])
        emirate = localized_selectbox(col1, t("emirate"), ["Abu Dhabi", "Dubai", "Sharjah", "Ajman", "Fujairah", "Ras Al Khaimah", "Umm Al Quwain"])
        cancer = localized_selectbox(col1, t("cancer_type"), ["Breast", "Colorectal", "Leukemia", "Liver", "Lung", "Ovarian", "Pancreatic", "Prostate"])
        stage = localized_selectbox(col2, t("stage"), ["I", "II", "III", "IV"])
        smoking = localized_selectbox(col2, t("smoking"), ["Non-Smoker", "Smoker"])
        comorbidity = localized_selectbox(col1, t("comorbidity"), ["Unknown", "Cardiovascular Disease", "Diabetes", "Hypertension"])
        ethnicity = localized_selectbox(col2, t("ethnicity"), ["Arab", "African", "East Asian", "European", "South Asian"])
        submit = st.form_submit_button(t("calculate"), type="primary")
    if submit:
        row = pd.DataFrame([{"Age": age, "BMI": bmi, "Gender": gender,
                             "Nationality": nationality, "Emirate": emirate,
                             "Cancer_Type": cancer, "Cancer_Stage": stage,
                             "Smoking_Status": smoking, "Comorbidities": comorbidity,
                             "Ethnicity": ethnicity}])
        probs = model.predict_proba(row)[0]
        result = pd.DataFrame({"Recorded class": model.classes_, "Model score": probs}).sort_values("Model score", ascending=False)
        best = result.iloc[0]
        st.metric(t("most_likely"), display_value(best["Recorded class"]), f"{best['Model score']:.1%} {t('model_score').lower()}")
        chart_result = result.copy()
        chart_result["Recorded class"] = chart_result["Recorded class"].map(display_value)
        chart_result = chart_result.rename(columns={"Recorded class": t("recorded_class"), "Model score": t("model_score")})
        st.bar_chart(chart_result.set_index(t("recorded_class")), horizontal=True)
        st.dataframe(chart_result.style.format({t("model_score"): "{:.1%}"}), use_container_width=True)
        st.caption(t("score_caption"))

with tab3:
    st.subheader(t("model_performance"))
    a, b, c = st.columns(3)
    a.metric(t("rows_modeled"), f"{metrics['rows']:,}")
    b.metric(t("accuracy"), f"{metrics['test_metrics']['accuracy']:.1%}")
    c.metric(t("macro_f1"), f"{metrics['test_metrics']['macro_f1']:.3f}")
    st.markdown(f"**{t('model_comparison')}**")
    compare = pd.DataFrame(metrics["comparison"]).T.rename_axis(display_value("Model"))
    if is_arabic:
        compare.index = [display_value(value) for value in compare.index]
    st.dataframe(compare.style.format("{:.3f}"), use_container_width=True)
    tuning = metrics["tuning"]
    best_params = tuning["best_params"]
    if is_arabic:
        params_summary = f"أوزان الفئات: {display_value(best_params['model__class_weight'])}، C: {best_params['model__C']}"
    else:
        params_summary = f"class_weight: {best_params['model__class_weight']}, C: {best_params['model__C']}"
    st.write(t("tuning_selection", params=params_summary, score=tuning["best_cv_macro_f1"],
               trials=tuning["trials"], folds=tuning["folds"]))
    st.caption(t("target_caption"))
    st.subheader(t("confusion_matrix"))
    cm = pd.DataFrame(metrics["test_metrics"]["confusion_matrix"],
                      index=metrics["test_metrics"]["class_order"],
                      columns=metrics["test_metrics"]["class_order"])
    if is_arabic:
        cm.index = [display_value(value) for value in cm.index]
        cm.columns = [display_value(value) for value in cm.columns]
    st.dataframe(cm, use_container_width=True)
