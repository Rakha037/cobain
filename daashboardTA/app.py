import streamlit as st
import pandas as pd
import numpy as np
import joblib
from keras.models import load_model, Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
import os
import random
import tensorflow as tf

# =========================
# Konfigurasi UI & Styling
# =========================
st.set_page_config(
    page_title="Dashboard Peramalan Saham INCO",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Kustom untuk styling yang lebih baik
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f4e79 0%, #2980b9 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .section-header {
        background: linear-gradient(90deg, #34495e 0%, #5d6d7e 100%);
        color: white;
        padding: 0.75rem;
        border-radius: 8px;
        margin: 1rem 0;
        text-align: center;
        font-weight: bold;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3498db;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .success-card {
        background: linear-gradient(90deg, #27ae60 0%, #2ecc71 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .info-card {
        background: linear-gradient(90deg, #3498db 0%, #5dade2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .warning-card {
        background: linear-gradient(90deg, #f39c12 0%, #f4d03f 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .home-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 5px solid #3498db;
    }
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        text-align: center;
    }
    .tabs-container {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stButton > button {
        background: linear-gradient(90deg, #3498db 0%, #2980b9 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(52, 152, 219, 0.4);
    }
    .sidebar-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ==== SET SEED SUPAYA HASIL LEBIH KONSISTEN ====
seed_value = 42
np.random.seed(seed_value)
random.seed(seed_value)
tf.random.set_seed(seed_value)

# =========================
# Konfigurasi Parameter Tuning (Updated)
# =========================
def get_tuning_params(n_predictors):
    """
    Mengembalikan parameter tuning berdasarkan jumlah variabel prediktor
    """
    tuning_config = {
        1: {"lag": 12, "batch_size": 40, "units": 50, "dropout": 0.2, "learning_rate": 0.01, "epochs": 150},
        2: {"lag": 2, "batch_size": 40, "units": 100, "dropout": 0.2, "learning_rate": 0.01, "epochs": 150},
        3: {"lag": 1, "batch_size": 40, "units": 50, "dropout": 0.2, "learning_rate": 0.01, "epochs": 150},
        4: {"lag": 1, "batch_size": 40, "units": 150, "dropout": 0.3, "learning_rate": 0.001, "epochs": 150},
        5: {"lag": 1, "batch_size": 10, "units": 50, "dropout": 0.2, "learning_rate": 0.001, "epochs": 150}
    }
    
    # Jika lebih dari 5 variabel, gunakan parameter untuk 5 variabel
    if n_predictors > 5:
        return tuning_config[5]
    
    return tuning_config.get(n_predictors, tuning_config[4])  # Default ke 4 variabel jika tidak ditemukan

# =========================
# Fungsi Membuat Model Baru (dinamis sesuai parameter tuning)
# =========================
def build_lstm_model(n_features, units, dropout, learning_rate):
    """
    Membangun model LSTM dengan parameter yang dapat dikonfigurasi termasuk learning rate
    """
    # Set seed untuk konsistensi
    tf.random.set_seed(seed_value)
    
    model = Sequential()
    model.add(LSTM(units=units, return_sequences=False, input_shape=(1, n_features)))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    
    # Kompilasi model dengan learning rate kustom
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mse")
    return model

# =========================
# Fungsi untuk membuat lag features
# =========================
def create_lag_features(X, y, lag=1):
    """
    Membuat lag features dari variabel prediktor
    X: variabel prediktor
    y: variabel target
    lag: jumlah lag (periode sebelumnya yang digunakan)
    
    Return:
    X_lag: variabel prediktor dengan lag (X[t-lag] untuk prediksi y[t])
    y_current: variabel target saat ini (y[t])
    """
    X_lag = X[:-lag] if lag > 0 else X  # Ambil X[t-lag] sampai X[T-lag]
    y_current = y[lag:]  # Ambil y[lag] sampai y[T] (target saat ini)
    
    return X_lag, y_current

# =========================
# Fungsi untuk membuat model LSTM univariate (untuk variabel prediktor)
# =========================
def build_univariate_lstm_model(sequence_length=1, units=50, dropout=0.1):
    """
    Membuat model LSTM untuk peramalan univariate (satu variabel)
    """
    # Set seed untuk konsistensi
    tf.random.set_seed(seed_value)
    
    model = Sequential()
    model.add(LSTM(units=units, return_sequences=False, input_shape=(sequence_length, 1)))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    return model

# =========================
# Fungsi untuk membuat sequences untuk univariate time series
# =========================
def create_univariate_sequences(data, sequence_length=1):
    """
    Membuat sequences untuk univariate time series prediction
    """
    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i])
        y.append(data[i])
    return np.array(X), np.array(y)

# =========================
# Fungsi untuk melatih model LSTM untuk variabel prediktor
# =========================
def train_predictor_models(X_original, selected_predictors, sequence_length=1):
    
    predictor_models = {}
    predictor_scalers = {}
    
    for i, predictor in enumerate(selected_predictors):
        # Set seed setiap kali melatih model
        tf.random.set_seed(seed_value + i)
        
        # Data untuk prediktor ini
        predictor_data = X_original[:, i]
        
        # Skip jika data tidak cukup
        if len(predictor_data) <= sequence_length:
            continue
            
        # Scaling
        scaler = MinMaxScaler()
        predictor_scaled = scaler.fit_transform(predictor_data.reshape(-1, 1)).flatten()
        
        # Buat sequences
        X_seq, y_seq = create_univariate_sequences(predictor_scaled, sequence_length)
        
        if len(X_seq) > 0:
            # Reshape untuk LSTM
            X_seq = X_seq.reshape(X_seq.shape[0], X_seq.shape[1], 1)
            
            # Buat dan latih model
            model = build_univariate_lstm_model(sequence_length, units=50, dropout=0.1)
            model.fit(X_seq, y_seq, epochs=50, batch_size=16, verbose=0)
            
            # Simpan model dan scaler
            predictor_models[predictor] = model
            predictor_scalers[predictor] = scaler
    
    return predictor_models, predictor_scalers

# =========================
# Fungsi untuk forecast variabel prediktor
# =========================
def forecast_predictors_lstm(X_original, selected_predictors, predictor_models, predictor_scalers, n_future, sequence_length=5):
    """
    Forecast variabel prediktor menggunakan model LSTM
    """
    forecasted_predictors = {}
    
    for i, predictor in enumerate(selected_predictors):
        if predictor not in predictor_models:
            # Jika tidak ada model, gunakan nilai terakhir
            last_value = X_original[-1, i]
            forecasted_predictors[predictor] = [last_value] * n_future
            continue
        
        model = predictor_models[predictor]
        scaler = predictor_scalers[predictor]
        
        # Ambil data terakhir
        predictor_data = X_original[:, i]
        predictor_scaled = scaler.transform(predictor_data.reshape(-1, 1)).flatten()
        
        # Ambil sequence terakhir
        last_sequence = predictor_scaled[-sequence_length:]
        current_sequence = last_sequence.copy()
        
        predictions = []
        for _ in range(n_future):
            # Reshape untuk prediksi
            input_seq = current_sequence.reshape(1, sequence_length, 1)
            
            # Prediksi
            pred_scaled = model.predict(input_seq, verbose=0)[0][0]
            pred = scaler.inverse_transform([[pred_scaled]])[0][0]
            predictions.append(pred)
            
            # Update sequence (geser dan tambah prediksi baru)
            current_sequence = np.append(current_sequence[1:], pred_scaled)
        
        forecasted_predictors[predictor] = predictions
    
    return forecasted_predictors

# =========================
# Fungsi untuk split data training dan testing
# =========================
def split_data(X, y, train_ratio=0.8):
    """
    Split data menjadi training dan testing
    """
    split_idx = int(len(X) * train_ratio)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    return X_train, X_test, y_train, y_test

# =========================
# Fungsi untuk generate forecast dengan parameter dinamis
# =========================
def generate_forecast(n_future):
    """
    Generate forecast dengan jumlah periode yang ditentukan
    """
    if "model" not in st.session_state:
        st.error("Model belum dilatih. Silakan latih model terlebih dahulu.")
        return
    
    model = st.session_state["model"]
    scaler_X = st.session_state["scaler_X"]
    scaler_y = st.session_state["scaler_y"]
    selected_predictors = st.session_state["selected_predictors"]
    
    # Menggunakan LSTM untuk forecast variabel prediktor
    predictor_models = st.session_state["predictor_models"]
    predictor_scalers = st.session_state["predictor_scalers"]
    X_original = st.session_state["X_original"]
    
    # Forecast variabel prediktor menggunakan LSTM
    forecasted_predictors = forecast_predictors_lstm(
        X_original, selected_predictors, predictor_models, predictor_scalers, n_future, sequence_length=5
    )
    
    # Forecast target menggunakan hasil forecast prediktor
    future_preds = []
    
    for period in range(n_future):
        # Buat input dari hasil forecast variabel prediktor
        current_predictor_values = []
        for predictor in selected_predictors:
            current_predictor_values.append(forecasted_predictors[predictor][period])
        
        # Convert ke numpy array dan scale
        current_input = np.array(current_predictor_values)
        current_input_scaled = scaler_X.transform(current_input.reshape(1, -1))[0]
        
        # Reshape untuk prediksi LSTM target (1 sample, 1 timestep, n_features)
        input_reshaped = current_input_scaled.reshape(1, 1, -1)
        
        # Prediksi target
        pred_scaled = model.predict(input_reshaped, verbose=0)
        pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
        future_preds.append(pred)
    
    return future_preds, forecasted_predictors

# =========================
# INTERFACE APLIKASI UTAMA
# =========================

# Header Utama
st.markdown("""
<div class="main-header">
    <h1>Dashboard Peramalan Saham INCO</h1>
    <p>Sistem Prediksi Harga Saham Berbasis LSTM</p>
</div>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    # Bagian Upload File
    st.markdown("""
    <div class="sidebar-card">
        <h3 style="color: #34495e;">Upload Data</h3>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Unggah File Excel", 
        type=["xlsx", "xls"], 
        key="shared_file_upload",
        help="Unggah file data saham yang berisi kolom 'Saham Inco' dan variabel prediktor"
    )

    # Status File
    if uploaded_file is not None:
        if "uploaded_df" not in st.session_state or st.session_state.get("uploaded_filename", "") != uploaded_file.name:
            try:
                df = pd.read_excel(uploaded_file)
                st.session_state["uploaded_df"] = df
                st.session_state["uploaded_filename"] = uploaded_file.name
                st.markdown("""
                <div class="success-card">
                    <h4>File Berhasil Diunggah!</h4>
                    <p style="margin: 0;">{}</p>
                </div>
                """.format(uploaded_file.name), unsafe_allow_html=True)
            except Exception as e:
                st.markdown(f"""
                <div class="warning-card">
                    <h4>Kesalahan Upload</h4>
                    <p style="margin: 0;">{str(e)}</p>
                </div>
                """, unsafe_allow_html=True)
                st.session_state["uploaded_df"] = None
        else:
            st.markdown("""
            <div class="info-card">
                <h4>Dataset Aktif</h4>
                <p style="margin: 0;">{}</p>
            </div>
            """.format(st.session_state['uploaded_filename']), unsafe_allow_html=True)
            
    menu = st.radio(
        "Pilih Modul:", 
        ["Halaman Utama", "Analisis Data", "Dashboard Peramalan"], 
        index=0
    )

# =========================
# HALAMAN UTAMA
# =========================
if menu == "Halaman Utama":
    st.markdown("""
    <div class="section-header">
        <h2>Selamat Datang di Dashboard Peramalan Saham INCO</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Penjelasan Dashboard
    st.markdown("""
    <div class="home-card">
        <h3>Tentang Dashboard</h3>
        <p>Dashboard ini merupakan sistem prediksi harga saham PT Vale Indonesia Tbk (INCO) yang menggunakan metode
        <strong>Long Short-Term Memory (LSTM)</strong>, sebuah jenis neural network yang khusus dirancang untuk 
        menganalisis data time series seperti harga saham.</p>
        
    </div>
    """, unsafe_allow_html=True)
    
    # Fitur Utama
    st.markdown("""
    <div class="section-header">
        <h3>Fitur-Fitur Dashboard</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4>Analisis Data</h4>
            <p>Eksplorasi mendalam data historis dengan visualisasi interaktif dan analisis korelasi untuk memahami hubungan antar variabel.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h4>Parameter Tuning Otomatis</h4>
            <p>Sistem secara otomatis menyesuaikan parameter model LSTM berdasarkan jumlah variabel prediktor yang dipilih.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4>Peramalan LSTM</h4>
            <p>Prediksi harga saham masa depan menggunakan model LSTM yang telah dioptimalkan.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h4>Visualisasi Hasil</h4>
            <p>Grafik interaktif yang menampilkan hasil prediksi, perbandingan dengan data aktual, dan metrik evaluasi model.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Panduan Penggunaan
    st.markdown("""
    <div class="section-header">
        <h3>Cara Menggunakan Dashboard</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="home-card">
        <h4>Langkah 1: Persiapan Data</h4>
        <ul>
            <li>Siapkan file Excel (.xlsx atau .xls) yang berisi data historis harga saham</li>
            <li>Pastikan file memiliki kolom <strong>'Saham Inco'</strong> sebagai variabel target</li>
            <li>Sertakan variabel prediktor seperti harga saham, harga nikel, nilai tukar, inflasi, bi rate</li>
            <li>Data berupa time series dengan periode yang konsisten</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="home-card">
        <h4>Langkah 2: Upload dan Eksplorasi Data</h4>
        <ul>
            <li>Gunakan fitur upload di sidebar untuk mengunggah file Excel</li>
            <li>Pilih menu <strong>"Analisis Data"</strong> untuk mengeksplorasi dataset</li>
            <li>Analisis korelasi antar variabel untuk memahami hubungan data</li>
            <li>Pilih variabel prediktor yang memiliki korelasi signifikan</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="home-card">
        <h4>Langkah 3: Pelatihan Model</h4>
        <ul>
            <li>Pindah ke menu <strong>"Dashboard Peramalan"</strong></li>
            <li>Pilih variabel prediktor yang akan digunakan untuk prediksi</li>
            <li>Sistem akan otomatis menentukan parameter optimal berdasarkan jumlah variabel</li>
            <li>Klik <strong>"Latih Model LSTM"</strong> dan tunggu proses pelatihan selesai</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="home-card">
        <h4>Langkah 4: Menghasilkan Peramalan</h4>
        <ul>
            <li>Setelah model berhasil dilatih, pilih jumlah periode yang ingin diramalkan</li>
            <li>Klik <strong>"Buat Peramalan LSTM"</strong> untuk menghasilkan prediksi</li>
            <li>Analisis hasil peramalan melalui grafik dan tabel yang disediakan</li>
            <li>Unduh hasil peramalan dalam format CSV untuk analisis lebih lanjut</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    

# =========================
# MODUL ANALISIS DATA 
# =========================
elif menu == "Analisis Data":
    st.markdown("""
    <div class="section-header">
        <h2>Eksplorasi Data</h2>
    </div>
    """, unsafe_allow_html=True)

    if "uploaded_df" in st.session_state and st.session_state["uploaded_df"] is not None:
        df = st.session_state["uploaded_df"]
        
        # Bagian Pratinjau Data
        with st.expander("Pratinjau Data", expanded=True):
            st.subheader("Ringkasan Dataset")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown("""
                <div class="metric-card">
                    <h3>{}</h3>
                    <p>Total Baris</p>
                </div>
                """.format(len(df)), unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="metric-card">
                    <h3>{}</h3>
                    <p>Total Kolom</p>
                </div>
                """.format(len(df.columns)), unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                <div class="metric-card">
                    <h3>{}</h3>
                    <p>Nilai Kosong</p>
                </div>
                """.format(df.isnull().sum().sum()), unsafe_allow_html=True)
            
            with col4:
                st.markdown("""
                <div class="metric-card">
                    <h3>{}</h3>
                    <p>Kolom Numerik</p>
                </div>
                """.format(len(df.select_dtypes(include=[np.number]).columns)), unsafe_allow_html=True)
            
            st.dataframe(df.head(10), use_container_width=True)
        
        # Analisis Time Series
        if "Saham Inco" in df.columns:
            predictor_vars = [col for col in df.columns if col != "Saham Inco"]
            
            if len(predictor_vars) > 0:
                # Pemilihan Variabel
                col1, col2 = st.columns([2, 1])
                with col1:
                    selected_vars_analysis = st.multiselect(
                        "Pilih variabel prediktor untuk analisis:",
                        predictor_vars,
                        default=predictor_vars[:4] if len(predictor_vars) >= 4 else predictor_vars,
                        key="analysis_vars"
                    )
                
                with col2:
                    st.markdown("""
                    <div class="info-card">
                        <h4>Tips Analisis</h4>
                        <p style="margin: 0; font-size: 0.9em;">Pilih variabel yang mungkin mempengaruhi harga saham INCO</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                if selected_vars_analysis:
                    # Grafik Perbandingan
                    st.subheader("Perbandingan Harga Saham vs Prediktor")
                    
                    n_predictors = len(selected_vars_analysis)
                    cols_comp = 2 if n_predictors > 1 else 1
                    rows_comp = (n_predictors + 1) // 2
                    
                    fig2, axes2 = plt.subplots(rows_comp, cols_comp, figsize=(15, 5*rows_comp))
                    fig2.suptitle("Harga Saham INCO vs Variabel Prediktor", fontsize=16, y=1.02)
                    
                    if rows_comp == 1:
                        axes2 = [axes2] if cols_comp == 1 else axes2
                    else:
                        axes2 = axes2.flatten()
                    
                    colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
                    
                    for i, var in enumerate(selected_vars_analysis):
                        if i < len(axes2):
                            ax = axes2[i]
                            ax_twin = ax.twinx()
                            
                            line1 = ax.plot(df.index, df['Saham Inco'], color='red', linewidth=2, 
                                           label='Saham INCO', alpha=0.8)
                            ax.set_xlabel('Periode')
                            ax.set_ylabel('Harga Saham INCO', color='red')
                            ax.tick_params(axis='y', labelcolor='red')
                            
                            color_pred = colors[i % len(colors)]
                            line2 = ax_twin.plot(df.index, df[var], color=color_pred, linewidth=2, 
                                               label=var, alpha=0.8, linestyle='--')
                            ax_twin.set_ylabel(var, color=color_pred)
                            ax_twin.tick_params(axis='y', labelcolor=color_pred)
                            
                            ax.set_title(f'Saham INCO vs {var}', fontweight='bold')
                            ax.grid(True, alpha=0.3)
                            
                            lines1, labels1 = ax.get_legend_handles_labels()
                            lines2, labels2 = ax_twin.get_legend_handles_labels()
                            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
                    
                    for i in range(n_predictors, len(axes2)):
                        axes2[i].set_visible(False)
                    
                    plt.tight_layout()
                    st.pyplot(fig2)
                    
                    # Analisis Korelasi
                    st.subheader("Analisis Korelasi")
                    
                    correlation_vars = ['Saham Inco'] + selected_vars_analysis
                    corr_matrix = df[correlation_vars].corr()
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        fig3, ax3 = plt.subplots(figsize=(10, 8))
                        im = ax3.imshow(corr_matrix, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)
                        
                        ax3.set_xticks(range(len(correlation_vars)))
                        ax3.set_yticks(range(len(correlation_vars)))
                        ax3.set_xticklabels(correlation_vars, rotation=45, ha='right')
                        ax3.set_yticklabels(correlation_vars)
                        
                        for i in range(len(correlation_vars)):
                            for j in range(len(correlation_vars)):
                                text = ax3.text(j, i, f'{corr_matrix.iloc[i, j]:.3f}',
                                              ha="center", va="center", color="black", fontweight='bold')
                        
                        cbar = plt.colorbar(im, ax=ax3)
                        cbar.set_label('Koefisien Korelasi', rotation=270, labelpad=20)
                        
                        ax3.set_title('Matriks Korelasi: Saham INCO vs Prediktor', 
                                     fontweight='bold', pad=20)
                        plt.tight_layout()
                        st.pyplot(fig3)
                    
                    with col2:
                        st.subheader("Ringkasan Korelasi")
                        corr_with_target = df[selected_vars_analysis + ['Saham Inco']].corr()['Saham Inco'].drop('Saham Inco')
                        corr_df = pd.DataFrame({
                            'Variabel': corr_with_target.index,
                            'Korelasi': corr_with_target.values,
                            'Kekuatan': ['Sangat Kuat' if abs(x) >= 0.8 else 
                                         'Kuat' if abs(x) >= 0.6 else
                                         'Sedang' if abs(x) >= 0.4 else
                                         'Lemah' if abs(x) >= 0.2 else
                                         'Sangat Lemah' for x in corr_with_target.values]
                        })
                        
                        st.dataframe(corr_df, use_container_width=True)
                        
                        # Tombol download
                        csv_corr = corr_df.to_csv(index=False)
                        st.download_button(
                            label="Unduh Data Korelasi",
                            data=csv_corr,
                            file_name="analisis_korelasi.csv",
                            mime="text/csv",
                            key="download_correlation_csv"
                        )
                else:
                    st.markdown("""
                    <div class="warning-card">
                        <h4>Pilihan Diperlukan</h4>
                        <p style="margin: 0;">Silakan pilih minimal 1 variabel prediktor untuk analisis time series</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="warning-card">
                    <h4>Tidak Ada Variabel Prediktor</h4>
                    <p style="margin: 0;">Tidak ditemukan variabel prediktor dalam dataset</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="warning-card">
                <h4>Kolom Target Hilang</h4>
                <p style="margin: 0;">Dataset harus memiliki kolom 'Saham Inco' untuk analisis time series</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="warning-card">
            <h4>Tidak Ada Data Tersedia</h4>
            <p style="margin: 0;">Silakan unggah file Excel di sidebar untuk memulai analisis</p>
        </div>
        """, unsafe_allow_html=True)

# =========================
# MODUL DASHBOARD PERAMALAN
# =========================
elif menu == "Dashboard Peramalan":
    st.markdown("""
    <div class="section-header">
        <h2>Dashboard Peramalan Saham Lanjutan</h2>
    </div>
    """, unsafe_allow_html=True)

    if "uploaded_df" in st.session_state and st.session_state["uploaded_df"] is not None:
        df = st.session_state["uploaded_df"]
        st.markdown(f"""
        <div class="success-card">
            <h4>Dataset Siap</h4>
            <p style="margin: 0;">Menggunakan file: {st.session_state['uploaded_filename']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Pratinjau Data Cepat
        with st.expander("Pratinjau Data Cepat"):
            st.dataframe(df.head(), use_container_width=True)

        # Pemilihan Variabel
        all_predictors = [col for col in df.columns if col != "Saham Inco"]
        
        if not all_predictors:
            st.markdown("""
            <div class="warning-card">
                <h4>Variabel Prediktor Hilang</h4>
                <p style="margin: 0;">File Excel harus berisi variabel prediktor selain kolom 'Saham Inco'</p>
            </div>
            """, unsafe_allow_html=True)
        elif "Saham Inco" not in df.columns:
            st.markdown("""
            <div class="warning-card">
                <h4>Kolom Target Hilang</h4>
                <p style="margin: 0;">File Excel harus berisi kolom target 'Saham Inco'</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Bagian Konfigurasi Model
            st.markdown("""
            <div class="section-header">
                <h3>Konfigurasi Model</h3>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                selected_predictors = st.multiselect(
                    "Pilih Variabel Prediktor:", 
                    all_predictors, 
                    default=all_predictors[:4] if len(all_predictors) >= 4 else all_predictors,
                    help="Pilih variabel yang akan digunakan untuk memprediksi harga saham INCO"
                )
            
            with col2:
                if selected_predictors:
                    n_predictors = len(selected_predictors)
                    params = get_tuning_params(n_predictors)
                    
                    st.markdown("""
                    <div class="info-card">
                        <h4>Parameter Tuning Otomatis</h4>
                        <p><strong>Variabel:</strong> {}</p>
                        <p><strong>Lag:</strong> {}</p>
                        <p><strong>Unit LSTM:</strong> {}</p>
                        <p><strong>Bath Size:</strong> {}</p>
                        <p><strong>Dropout:</strong> {}</p>
                        <p><strong>Learning Rate:</strong> {}</p>
                        <p><strong>Epoch:</strong> {}</p>
                    </div>
                    """.format(n_predictors, params["lag"], params["units"], params["batch_size"], 
                              params["dropout"], params["learning_rate"], params["epochs"]), unsafe_allow_html=True)

            if not selected_predictors:
                st.markdown("""
                <div class="warning-card">
                    <h4>Pilihan Diperlukan</h4>
                    <p style="margin: 0;">Silakan pilih minimal 1 variabel prediktor untuk melanjutkan</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Bagian Pelatihan
                st.markdown("""
                <div class="section-header">
                    <h3>Pelatihan Model</h3>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("Latih Model LSTM", type="primary", use_container_width=True):
                        with st.spinner("Melatih model LSTM"):
                            # Set seed untuk konsistensi
                            tf.random.set_seed(seed_value)
                            np.random.seed(seed_value)
                            random.seed(seed_value)
                            
                            # Kode pelatihan model
                            X = df[selected_predictors].values
                            y = df[['Saham Inco']].values.flatten()

                            lag = params["lag"]
                            X_lag, y_current = create_lag_features(X, y, lag)

                            X_train, X_test, y_train, y_test = split_data(X_lag, y_current, train_ratio=0.8)

                            scaler_X = MinMaxScaler()
                            scaler_y = MinMaxScaler()
                            
                            X_train_scaled = scaler_X.fit_transform(X_train)
                            y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
                            
                            X_test_scaled = scaler_X.transform(X_test)
                            y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

                            X_train_lstm = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
                            X_test_lstm = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])

                            model = build_lstm_model(
                                n_features=n_predictors,
                                units=params["units"],
                                dropout=params["dropout"],
                                learning_rate=params["learning_rate"]
                            )

                            history = model.fit(
                                X_train_lstm, y_train_scaled,
                                epochs=params["epochs"],
                                batch_size=params["batch_size"],
                                validation_data=(X_test_lstm, y_test_scaled),
                                verbose=0,
                                shuffle=False
                            )

                            y_train_pred_scaled = model.predict(X_train_lstm)
                            y_test_pred_scaled = model.predict(X_test_lstm)
                            
                            y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).flatten()
                            y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).flatten()

                            
                            predictor_models, predictor_scalers = train_predictor_models(X, selected_predictors, sequence_length=5)

                            # Simpan ke session state
                            st.session_state["scaler_X"] = scaler_X
                            st.session_state["scaler_y"] = scaler_y
                            st.session_state["model"] = model
                            st.session_state["selected_predictors"] = selected_predictors
                            st.session_state["params"] = params
                            st.session_state["last_input"] = X_test_scaled[-1]
                            st.session_state["history"] = history
                            st.session_state["predictor_models"] = predictor_models
                            st.session_state["predictor_scalers"] = predictor_scalers
                            st.session_state["X_original"] = X
                            st.session_state["model_trained"] = True
                            st.session_state["y_train"] = y_train
                            st.session_state["y_test"] = y_test
                            st.session_state["y_train_pred"] = y_train_pred
                            st.session_state["y_test_pred"] = y_test_pred

                            mape_train = mean_absolute_percentage_error(y_train, y_train_pred) * 100
                            mape_test = mean_absolute_percentage_error(y_test, y_test_pred) * 100

                            # Bagian Hasil
                            st.markdown("""
                            <div class="section-header">
                                <h3>Hasil Pelatihan</h3>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            
                            # Grafik Time Series (Aktual vs Prediksi) - Ukuran Lebih Besar
                            st.subheader("Hasil Prediksi vs Data Aktual")
                            
                            # Gabungkan data untuk visualisasi time series yang kontinyu
                            y_actual_full = np.concatenate([y_train, y_test])
                            y_pred_full = np.concatenate([y_train_pred, y_test_pred])
                            
                            fig, ax = plt.subplots(figsize=(18, 8))
                            
                            # Plot data aktual
                            ax.plot(range(len(y_actual_full)), y_actual_full, 
                                   label="Data Aktual", color='blue', linewidth=3, alpha=0.8)
                            
                            # Plot prediksi
                            ax.plot(range(len(y_pred_full)), y_pred_full, 
                                   label="Prediksi LSTM", color='red', linewidth=3, alpha=0.8, linestyle='--')
                            
                            # Tambahkan garis pembatas training-testing
                            split_point = len(y_train)
                            ax.axvline(x=split_point, color='green', linestyle=':', alpha=0.7, linewidth=3, 
                                      label=f'Batas Train-Test (Index: {split_point})')
                            
                            # Tambahkan text box dengan MAPE Testing
                            textstr = f'MAPE: {mape_test:.2f}%'
                            props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.9, edgecolor='darkgreen')
                            ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=14, fontweight='bold',
                                   verticalalignment='top', bbox=props)
                            
                            ax.set_xlabel("Periode", fontsize=12)
                            ax.set_ylabel("Harga Saham INCO", fontsize=12)
                            ax.set_title("Perbandingan Data Aktual vs Prediksi LSTM", fontweight='bold', fontsize=16)
                            ax.legend(loc='upper right', fontsize=12)
                            ax.grid(True, alpha=0.3)
                            
                            plt.tight_layout()
                            st.pyplot(fig)

                            st.markdown("""
                            <div class="success-card">
                                <h4>Pelatihan Model Selesai!</h4>
                                <p style="margin: 0;">Model LSTM Anda siap untuk peramalan. Gulir ke bawah untuk menghasilkan prediksi.</p>
                            </div>
                            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="warning-card">
            <h4>Tidak Ada Dataset Tersedia</h4>
            <p style="margin: 0;">Silakan unggah file Excel di sidebar untuk memulai peramalan</p>
        </div>
        """, unsafe_allow_html=True)

    # Bagian Peramalan (Hanya ditampilkan jika model sudah dilatih)
    if st.session_state.get("model_trained", False):
        st.markdown("---")
        st.markdown("""
        <div class="section-header">
            <h2>Peramalan Periode Kedepan</h2>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            n_future_lstm = st.slider(
                "Periode Peramalan:", 
                min_value=1, 
                max_value=24, 
                value=6, 
                key="lstm_forecast_slider",
                help="Pilih jumlah periode kedepan untuk diramalkan"
            )
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Konfigurasi Peramalan</h4>
                <p><strong>Metode:</strong> Prediksi Auto LSTM</p>
                <p><strong>Periode:</strong> {n_future_lstm}</p>
                <p><strong>Variabel:</strong> {len(st.session_state.get('selected_predictors', []))}</p>
            </div>
            """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Buat Peramalan LSTM", key="lstm_forecast_btn", type="primary", use_container_width=True):
                with st.spinner("Menghasilkan peramalan LSTM..."):
                    future_preds, forecasted_predictors = generate_forecast(n_future_lstm)
                    
                    if future_preds is not None:
                        st.session_state["lstm_results"] = {
                            "future_preds": future_preds,
                            "forecasted_predictors": forecasted_predictors,
                            "n_periods": n_future_lstm
                        }
                        
                        # Tampilan Hasil
                        st.markdown("""
                        <div class="section-header">
                            <h3>Hasil Peramalan LSTM</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Tabel Hasil
                        hasil_data = {"Periode": range(1, n_future_lstm+1), "Peramalan_Saham_INCO": future_preds}
                        hasil_df = pd.DataFrame(hasil_data)
                        
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.subheader("Data Peramalan")
                            st.dataframe(hasil_df, use_container_width=True)
                        
                        with col2:
                            avg_forecast = np.mean(future_preds)
                            trend = "Naik" if future_preds[-1] > future_preds[0] else "Turun"
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4>Ringkasan Peramalan</h4>
                                <p><strong>Rata-rata:</strong> {avg_forecast:.2f}</p>
                                <p><strong>Tren:</strong> {trend}</p>
                                <p><strong>Rentang:</strong> {min(future_preds):.2f} - {max(future_preds):.2f}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Visualisasi Peramalan - Ukuran Lebih Besar
                        st.subheader("Visualisasi Peramalan")
                        fig2, ax2 = plt.subplots(figsize=(16, 8))
                        ax2.plot(range(1, n_future_lstm+1), future_preds, marker='o', color='orange', linewidth=4, markersize=10)
                        ax2.set_xlabel("Periode Masa Depan", fontsize=12)
                        ax2.set_ylabel("Peramalan Harga Saham INCO", fontsize=12)
                        ax2.set_title("Peramalan Harga Saham Berbasis LSTM", fontweight='bold', fontsize=16)
                        ax2.grid(True, alpha=0.3)
                        
                        for i, value in enumerate(future_preds):
                            ax2.annotate(f'{value:.2f}', (i+1, value), textcoords="offset points", xytext=(0,12), ha='center', fontsize=10, fontweight='bold')
                        
                        plt.tight_layout()
                        st.pyplot(fig2)

                        # Opsi Unduh
                        csv_lstm = hasil_df.to_csv(index=False)
                        st.download_button(
                            label="Unduh Peramalan LSTM",
                            data=csv_lstm,
                            file_name=f"peramalan_lstm_{n_future_lstm}_periode.csv",
                            mime="text/csv",
                            key="download_lstm_csv"
                        )
        
        # Tampilkan hasil sebelumnya jika tersedia
        if "lstm_results" in st.session_state:
            st.markdown("""
            <div class="info-card">
                <h4>Hasil Sebelumnya Tersedia</h4>
                <p style="margin: 0;">Peramalan terakhir: {} periode</p>
            </div>
            """.format(st.session_state['lstm_results']['n_periods']), unsafe_allow_html=True)
            
            if st.button("Tampilkan Hasil LSTM Sebelumnya", key="show_lstm_results"):
                results = st.session_state["lstm_results"]
                future_preds = results["future_preds"]
                n_periods = results["n_periods"]
                
                hasil_data = {"Periode": range(1, n_periods+1), "Peramalan_Saham_INCO": future_preds}
                hasil_df = pd.DataFrame(hasil_data)
                st.dataframe(hasil_df, use_container_width=True)