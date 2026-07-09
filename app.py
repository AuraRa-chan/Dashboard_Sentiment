import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import plotly.express as px
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
from sklearn.metrics import confusion_matrix
import seaborn as sns
from random import choice
from sklearn.metrics import classification_report

st.set_page_config(
    page_title="Skripsi",
    layout="wide"
)

st.markdown("""
<style>

.main-header{
    text-align:center;
    color:#1E3A8A;
    font-size:42px;
    font-weight:bold;
}

.sub-header{
    text-align:center;
    color:#64748B;
    font-size:18px;
}

.insight-box{
    background-color:#F8FAFC;
    padding:20px;
    border-radius:10px;
    border-left:6px solid #2563EB;
    margin-top:10px;
}

</style>
""", unsafe_allow_html=True)

df = pd.read_csv("Sentimen.csv")

import pandas as pd

hasil_model = pd.read_csv("hasil_model.csv")
hasil_prediksi = pd.read_csv("hasil_prediksi.csv")

report_nb = pd.read_csv("classification_report_nb.csv")
report_svm = pd.read_csv("classification_report_svm.csv")

cm_nb = pd.read_csv("confusion_matrix_nb.csv")
cm_svm = pd.read_csv("confusion_matrix_svm.csv")

kfold = pd.read_csv("kfold_result.csv")

prapemrosesan = pd.read_csv("cleaned_translated_texts.csv")

# ==========================
# FILTER SIDEBAR
# ==========================
with st.sidebar:

    col1, col2 = st.columns([1, 3])

    with col1:
        st.image("logo kampus.png", width=50)

    with col2:
        st.markdown("""
        **UNIVERSITAS SEBELAS APRIL**  
        Informatika
        """)

with st.sidebar:

    st.markdown("---")

    selected_menu = option_menu(
        menu_title=None,
        options=[
            "Overview",
            "Alur Penelitian",
            "Dataset",
            "Distribusi Sentimen",
            "WordCloud",
            "Top Words TF-IDF",
            "Perbandingan Model",
            "Evaluasi Model",
            "Kesimpulan"
        ],
        icons=[
            "house",
            "diagram-3",
            "database",
            "pie-chart",
            "cloud",
            "table",
            "bar-chart",
            "graph-up",
            "file-earmark-text"
        ],
        default_index=0,
        styles={
            "container": {
                "padding": "0!important"
            },
            "icon": {
                "font-size": "18px"
            },
            "nav-link": {
                "font-size": "16px",
                "text-align": "left"
            },
            "nav-link-selected": {
                "background-color": "#2563EB"
            }
        }
    )

# ==========================
# OVERVIEW
# ==========================
if selected_menu == "Overview":

    st.markdown(
        "<h1 style='text-align:center; color:#1E3A8A;'>⌘ Klasifikasi Sentimen</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<h4 style='text-align:center; color:gray;'>Perbandingan Algoritma Naive Bayes dan Support Vector Machine</h4>",
        unsafe_allow_html=True
    )

    st.subheader("Tentang Dashboard")

    st.write("""
    Dashboard Analisis Sentimen Komentar YouTube ini dikembangkan sebagai media
    visualisasi hasil penelitian analisis sentimen terhadap komentar pengguna pada
    video YouTube Windah Basudara. Analisis dilakukan menggunakan algoritma
    Naive Bayes dan Support Vector Machine (SVM) dengan memanfaatkan metode
    TF-IDF sebagai pembobotan kata.

    Melalui dashboard ini, pengguna dapat melihat ringkasan dataset, distribusi
    sentimen, visualisasi kata-kata dominan, hasil pembobotan TF-IDF, performa
    model klasifikasi, serta evaluasi hasil prediksi secara interaktif.
    """)

    st.subheader("Tujuan Dashboard")

    st.write("""
    Dashboard ini bertujuan untuk:

    - Menyajikan hasil analisis sentimen komentar YouTube secara interaktif.
    - Mempermudah interpretasi distribusi sentimen positif, netral, dan negatif.
    - Menampilkan kata-kata dominan pada setiap kategori sentimen menggunakan TF-IDF.
    - Menampilkan hasil klasifikasi menggunakan algoritma Naive Bayes dan Support Vector Machine (SVM).
    - Membandingkan performa kedua algoritma berdasarkan metrik evaluasi seperti Accuracy, Precision, Recall, dan F1-Score.
    """)
    
# ==========================
# Alur Penelitian
# ==========================
elif selected_menu == "Alur Penelitian":

    st.subheader("◇ Alur Penelitian")

    col1, col2, col3 = st.columns([1, 3, 1])

    with col2:
     st.image(
        "Alur Penelitian.png",
        width=700
    )
    
# ==========================
# Dataset
# ==========================
elif selected_menu == "Dataset":

    st.title("Dataset Penelitian")
     
    st.subheader("Sumber Data")

    st.markdown("""
    **Sumber Komentar:**

    - Platform : YouTube
    - Sumber Data : Komentar pada video review Gameplay Love And Deepspace
    - Objek Penelitian : Sentimen pengguna/penonton terhadap Game Love And Deepspace
    - Teknik Akuisisi : Scraping Komentar
    - Jenis Data : Data teks tidak terstruktur (komentar reguler)
    - Periode Pengambilan Data : Januari 2025 - Mei 2026
    - Jumlah Data : 3.363 Komentar
    - Bahasa Data : Campuran (Indonesia, Inggris)
    """)

    st.info("""
    Dataset diperoleh dari komentar pengguna pada platform YouTube
    melalui proses scraping menggunakan pustaka youtube-comment-downloader.

    Data kemudian melalui tahapan preprocessing,
    translasi bahasa, pelabelan sentimen,
    dan klasifikasi menggunakan algoritma
    Naive Bayes serta Support Vector Machine.
    """)

    st.subheader("Informasi Dataset")

    text = pd.read_csv(
     "cleaned_translated_texts.csv"
    )

    comparison = pd.DataFrame({
    "Komentar Asli":
        text["text"]
    })

    st.dataframe(
    comparison.head(10),
    use_container_width=True
    ) 

    st.subheader("Informasi Dataset setelah Pra-Pemrosesan")

    c1, c2, c3 = st.columns(3)

    with c1:
     st.metric(
        "Jumlah Data",
        len(prapemrosesan)
    )

    with c2:
     st.metric(
        "Jumlah Kolom",
        len(prapemrosesan.columns)
    )

    with c3:
     st.metric(
        "Jumlah Kelas",
        df["Label_Asli"].nunique()
    )


    df_asli = pd.read_csv(
     "DATASET 2.csv"
    )

    df_hasil = pd.read_csv(
     "Sentimen.csv"
    )

    comparison = pd.DataFrame({
    "Komentar Asli":
        prapemrosesan["text"],
    "Teks Bersih":
        prapemrosesan["teks_bersih"],
    "Teks Normalisasi":
        prapemrosesan["teks_normal"],
    "Teks Stopwords":
        prapemrosesan["teks_stopwords"],
    "Teks Stemming":
        prapemrosesan["teks_stemmed"],
    "Teks Translasi":
        prapemrosesan["teks_terjemah"]
    })

    st.dataframe(
    comparison.head(10),
    use_container_width=True
    ) 
     
# ==========================
# WordCloud
# ==========================
elif selected_menu == "WordCloud":

    st.title("Word Cloud")

    df_hasil = pd.read_csv("Sentimen.csv")

    label_source = st.selectbox(
    "Pilih Sumber Sentimen",
    [
        "TextBlob",
        "Naive Bayes",
        "Support Vector Machine"
    ]
)

    if label_source == "TextBlob":
     label_col = "Label_Asli"
    elif label_source == "Naive Bayes":
     label_col = "Prediksi_NB"
    else:
     label_col = "Prediksi_SVM"

    selected_wc = st.selectbox(
    "Pilih Sentimen",
    ["Positive", "Negative", "Neutral"]
)

    wc_df = df_hasil[df_hasil[label_col] == selected_wc]

    text = " ".join(
    wc_df["teks_terjemah"].astype(str)
)

    if selected_wc == "Positive":
     color = "Greens"
    elif selected_wc == "Negative":
     color = "Reds"
    else:
     color = "Blues"

    if text.strip() == "":
     st.warning("Tidak ada data untuk ditampilkan.")
    else:
     wc = WordCloud(
        width=1200,
        height=600,
        background_color="white",
        colormap=color,
        collocations=False
    ).generate(text)

    fig, ax = plt.subplots(figsize=(12,6))

    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")

    st.pyplot(fig)

    st.info(f"""
    WordCloud menampilkan kata-kata yang paling sering muncul berdasarkan **{label_source}**
    dengan sentimen **{selected_wc}**.

    Jumlah komentar:
    **{len(wc_df)} komentar**
""")
    
# ==========================
# DISTRIBUSI SENTIMEN
# ==========================
elif selected_menu == "Distribusi Sentimen":

    st.title("Distribusi Sentimen")

    df_hasil = pd.read_csv("Sentimen.csv")

    comparison = pd.DataFrame({
    "Teks Translasi": df_hasil["teks_terjemah"],
    "Sentimen TextBlob": df_hasil["Label_Asli"],
    "Sentimen NB": df_hasil["Prediksi_NB"],
    "Sentimen SVM": df_hasil["Prediksi_SVM"],
})

    st.dataframe(
    comparison.head(10),
    use_container_width=True
)
    
    sentiment_tb = (
    df_hasil["Label_Asli"]
    .value_counts()
    .reset_index()
)
    sentiment_tb.columns = ["Sentiment", "Jumlah"]

    sentiment_nb = (
    df_hasil["Prediksi_NB"]
    .value_counts()
    .reset_index()
)
    sentiment_nb.columns = ["Sentiment", "Jumlah"]

    sentiment_svm = (
    df_hasil["Prediksi_SVM"]
    .value_counts()
    .reset_index()
)
    sentiment_svm.columns = ["Sentiment", "Jumlah"]

    col1, col2, col3 = st.columns(3)

    with col1:
     fig_tb = px.pie(
        sentiment_tb,
        names="Sentiment",
        values="Jumlah",
        title="TextBlob",
        color="Sentiment",
        color_discrete_map={
            "Positive": "#22C55E",
            "Negative": "#EF4444",
            "Neutral": "#F59E0B"
        }
    )
    st.plotly_chart(fig_tb, use_container_width=True)

    with col2:
     fig_nb = px.pie(
        sentiment_nb,
        names="Sentiment",
        values="Jumlah",
        title="Naive Bayes",
        color="Sentiment",
        color_discrete_map={
            "Positive": "#22C55E",
            "Negative": "#EF4444",
            "Neutral": "#F59E0B"
        }
    )
    st.plotly_chart(fig_nb, use_container_width=True)

    with col3:
     fig_svm = px.pie(
        sentiment_svm,
        names="Sentiment",
        values="Jumlah",
        title="Support Vector Machine",
        color="Sentiment",
        color_discrete_map={
            "Positive": "#22C55E",
            "Negative": "#EF4444",
            "Neutral": "#F59E0B"
        }
    )
    st.plotly_chart(fig_svm, use_container_width=True)

    tb_dom = df_hasil["Label_Asli"].value_counts().idxmax()
    nb_dom = df_hasil["Prediksi_NB"].value_counts().idxmax()
    svm_dom = df_hasil["Prediksi_SVM"].value_counts().idxmax()

    col1, col2, col3 = st.columns(3)

    col1.info(f"**TextBlob:** {tb_dom}")
    col2.info(f"**Naive Bayes:** {nb_dom}")
    col3.info(f"**SVM:** {svm_dom}")


# ==========================
# Top Words
# ==========================
elif selected_menu == "Top Words TF-IDF":
    import streamlit as st
    import pandas as pd

    st.title("Top Words TF-IDF")

    pilihan = st.selectbox(
    "Pilih Sentimen",
    ["Positif", "Netral", "Negatif"]
)

    positif = pd.DataFrame({
    "Keyword": [
        "bro", "really", "thanks", "windah", "thank",
        "love", "game", "playing", "cool", "proud",
        "handsome", "play", "good", "like", "sylus",
        "funny", "caleb", "great", "story", "respect"
    ],
    "TF-IDF": [
        101.548190, 53.446720, 49.492995, 44.390174,
        43.562902, 38.419258, 37.505978, 35.195983,
        34.424328, 30.065800, 29.754050, 23.941017,
        19.563513, 19.400405, 17.864212, 16.852814,
        15.480092, 15.363752, 14.744166, 14.709649
    ]
})

    netral = pd.DataFrame({
    "Keyword": [
        "yes", "bro", "yeah", "windah", "thank",
        "caleb", "play", "story", "continue", "gg",
        "sylus", "playing", "zayne", "games", "like",
        "bang", "don", "respect", "uchiha", "rafayel"
    ],
    "TF-IDF": [
        345.707176, 149.662317, 127.735136, 62.119809,
        61.770468, 29.638430, 28.105452, 25.809690,
        23.200546, 22.431684, 22.136315, 21.896436,
        21.401373, 19.817785, 18.924146, 17.966269,
        14.137629, 13.826654, 12.915796, 12.352060
    ]
})
    
    negatif = pd.DataFrame({
    "Keyword": [
        "game", "bro", "playing", "thank", "windah",
        "play", "really", "wrong", "thanks", "crazy",
        "like", "girls", "don", "late", "cool",
        "guys", "watching", "caleb", "bang", "little"
    ],
    "TF-IDF": [
        63.696060, 62.706550, 42.936598, 37.070795,
        26.976946, 23.193837, 18.284827, 17.112768,
        16.774243, 14.616863, 12.707859, 12.494596,
        10.752085, 9.762040, 8.171581, 8.085358,
        7.811406, 7.102635, 7.022773, 6.803624
    ]
})
    if pilihan == "Positif":
     st.success("😊 Top Words Sentimen Positif")
     st.dataframe(positif, use_container_width=True, hide_index=True)

     st.info("""
    **Interpretasi:**
    
    Komentar positif didominasi oleh kata-kata apresiasi seperti **thanks**, **love**, **cool**, dan **great**.
    Selain itu, penyebutan **Windah**, **game**, **playing**, serta karakter seperti **Sylus** dan **Caleb**
    menunjukkan bahwa penonton memberikan respons positif terhadap gameplay dan konten yang disajikan.
    """)

    elif pilihan == "Netral":
     st.warning("😐 Top Words Sentimen Netral")
     st.dataframe(netral, use_container_width=True, hide_index=True)

     st.info("""
    **Interpretasi:**
    
    Komentar netral didominasi oleh kata-kata percakapan seperti **yes**, **yeah**, dan **bro**.
    Selain itu, kata **Windah**, **story**, **play**, **Caleb**, dan **Sylus**
    menunjukkan bahwa komentar netral lebih banyak membahas isi video tanpa menunjukkan emosi yang kuat.
    """)

    else:
     st.error("😠 Top Words Sentimen Negatif")
     st.dataframe(negatif, use_container_width=True, hide_index=True)

     st.info("""
    **Interpretasi:**
    
    Komentar negatif masih didominasi pembahasan mengenai **game** dan aktivitas **playing**.
    Munculnya kata seperti **wrong**, **crazy**, dan **late**
    mengindikasikan adanya kritik atau ketidakpuasan terhadap situasi tertentu dalam video atau gameplay.
    """)


# ==========================
# Perbandingan Model
# ==========================
elif selected_menu == "Perbandingan Model":

    st.title("Perbandingan Prediksi Model")

    fig = px.bar(
    hasil_model,
    x="Model",
    y="Accuracy",
    color="Model",
    text=hasil_model["Accuracy"].map(lambda x: f"{x:.2%}")
)

    fig.update_traces(textposition="outside")

    fig.update_layout(
    yaxis_tickformat=".0%",
    xaxis_title="Model",
    yaxis_title="Accuracy",
    showlegend=False
)

    st.plotly_chart(fig, use_container_width=True)

    import pandas as pd

    hasil_prediksi = pd.read_csv("hasil_prediksi.csv")

    dist_nb = (
    hasil_prediksi["Prediksi NB"]
    .value_counts()
    .reset_index()
)

    dist_nb.columns = ["Sentimen", "Jumlah"]
    dist_nb["Model"] = "Naive Bayes"

    dist_svm = (
    hasil_prediksi["Prediksi SVM"]
    .value_counts()
    .reset_index()
)

    dist_svm.columns = ["Sentimen", "Jumlah"]
    dist_svm["Model"] = "Support Vector Machine"

    dist_all = pd.concat([dist_nb, dist_svm], ignore_index=True)

    fig = px.bar(
    dist_all,
    x="Sentimen",
    y="Jumlah",
    color="Model",
    barmode="group",
    text="Jumlah",
    title="Distribusi Sentimen Hasil Prediksi"
)

    fig.update_traces(textposition="outside")

    st.plotly_chart(fig, use_container_width=True)


# ==========================
# Evaluasi Model
# ==========================
elif selected_menu == "Evaluasi Model":

    st.title("Evaluasi Model")

    tab1, tab2 = st.tabs([
    "Naive Bayes",
    "Support Vector Machine"
    ])

    with tab1:

     st.subheader("Classification Report")

     st.dataframe(
        report_nb,
        use_container_width=True
    )

    with tab2:

     st.subheader("Classification Report")

     st.dataframe(
        report_svm,
        use_container_width=True
    )
     
    st.subheader("Confusion Matrix")

    col1, col2 = st.columns(2)

    with col1:
     st.write("Naive Bayes")

     fig, ax = plt.subplots(figsize=(3.3, 3.3))

     sns.heatmap(
         cm_nb,
         annot=True,
         fmt="d",
         cmap="Blues",
         square=True,
         cbar=False,
         xticklabels=["Neg", "Net", "Pos"],
         yticklabels=["Neg", "Net", "Pos"],
         annot_kws={"size": 9},
         ax=ax
    )
     st.pyplot(fig, use_container_width=False)

    with col2:
     st.write("Support Vector Machine")

     fig, ax = plt.subplots(figsize=(3.3, 3.3))

     sns.heatmap(
         cm_svm,
         annot=True,
         fmt="d",
         cmap="Greens",
         square=True,
         cbar=False,
         xticklabels=["Neg", "Net", "Pos"],
         yticklabels=["Neg", "Net", "Pos"],
         annot_kws={"size": 9},
         ax=ax
    )
     st.pyplot(fig, use_container_width=False)
     

    st.subheader("K-Fold Cross Validation")

    kfold = pd.read_csv("kfold_result.csv")


    nb_mean = kfold["Naive Bayes"].mean()
    svm_mean = kfold["SVM"].mean()

    nb_std = kfold["Naive Bayes"].std()
    svm_std = kfold["SVM"].std()

    col1, col2 = st.columns(2)

    col1.metric("Mean Accuracy NB", f"{nb_mean:.2%}")
    col2.metric("Mean Accuracy SVM", f"{svm_mean:.2%}")

    col3, col4 = st.columns(2)

    col3.metric("Std Dev NB", f"{nb_std:.2%}")
    col4.metric("Std Dev SVM", f"{svm_std:.2%}")

# ==========================
# Tabel Hasil Fold
# ==========================

    st.subheader("Hasil Setiap Fold")

    kfold_table = kfold.copy()

    if "Fold" not in kfold_table.columns:
     kfold_table.insert(
        0,
        "Fold",
        [f"Fold {i+1}" for i in range(len(kfold_table))]
    )
    else:
     kfold_table["Fold"] = kfold_table["Fold"].apply(lambda x: f"Fold {x}")

    kfold_table["Naive Bayes"] = kfold_table["Naive Bayes"].map(lambda x: f"{x:.2%}")
    kfold_table["SVM"] = kfold_table["SVM"].map(lambda x: f"{x:.2%}")

    st.dataframe(
    kfold_table,
    hide_index=True,
    use_container_width=True
)


# ==========================
# KESIMPULAN
# ==========================
elif selected_menu == "Kesimpulan":

    st.title("Kesimpulan Penelitian")

    st.info("""
    - **Support Vector Machine (SVM)** memberikan performa terbaik dengan **akurasi 89%**, lebih tinggi dibandingkan **Naive Bayes (80%)**.
    - Hasil **K-Fold Cross Validation** menunjukkan SVM lebih stabil dengan **standar deviasi 0,46%**, sedangkan Naive Bayes **0,89%**.
    - Distribusi komentar didominasi oleh **sentimen netral**, diikuti sentimen positif dan negatif.
    - Analisis **TF-IDF** menunjukkan komentar tidak hanya membahas *Love and Deepspace*, tetapi juga **Windah Basudara** serta karakter seperti **Sylus, Caleb,** dan **Zayne**.
    """)

    st.markdown("""
    ### Ringkasan Hasil Penelitian

    - Dataset yang digunakan telah melalui tahapan
    preprocessing, translasi, pelabelan sentimen,
    dan ekstraksi fitur menggunakan TF-IDF.

    - Hasil visualisasi menunjukkan bahwa mayoritas
    komentar memiliki sentimen netral.


    - Hasil evaluasi menggunakan 5-Fold Cross Validation
    menunjukkan bahwa algoritma Naive Bayes memperoleh
    rata-rata akurasi sebesar **80.40%** dengan standar
    deviasi **0.89%**.

    - Algoritma Support Vector Machine memperoleh
    rata-rata akurasi sebesar **89.18%** dengan standar
    deviasi **0.49%**.

    - Berdasarkan Confusion Matrix dan hasil evaluasi,
    Support Vector Machine mampu mengklasifikasikan
    sentimen dengan tingkat ketepatan yang lebih tinggi
    dibandingkan Naive Bayes.

    ### Model Terbaik

    Support Vector Machine (SVM) merupakan model terbaik
    pada penelitian ini karena memiliki:

    - Akurasi lebih tinggi (89.30%)
    - Tingkat kesalahan klasifikasi lebih rendah
    - Performa yang lebih stabil pada setiap fold
    - Kemampuan generalisasi yang lebih baik terhadap data


    ### Saran Pengembangan

    - Gunakan dataset yang lebih besar dan lebih beragam, dengan mengambil komentar dari beberapa video maupun kreator konten agar hasil analisis lebih representatif.
    - Bandingkan performa SVM dan Naive Bayes dengan algoritma lain seperti Random Forest, XGBoost, Logistic Regression, LSTM, maupun IndoBERT.
    - Tingkatkan kualitas pelabelan sentimen dengan anotasi manual oleh beberapa anotator serta mengukur tingkat kesepakatan menggunakan Cohen's Kappa atau Fleiss' Kappa.
    - Kembangkan penelitian menggunakan pendekatan lanjutan seperti Aspect-Based Sentiment Analysis (ABSA), Topic Modeling, dan Emotion Classification untuk memperoleh analisis yang lebih mendalam.
    - Analisis lebih lanjut pengaruh kreator konten, karakter favorit, dan interaksi komunitas terhadap pola sentimen pengguna di media sosial.
    """)
