import streamlit as st
import pandas as pd
import numpy as np
import joblib
from keras.models import load_model, Sequential
from keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error

# =========================
# Konfigurasi Parameter Tuning
# =========================
def get_tuning_params(n_predictors):
    """
    Mengembalikan parameter tuning berdasarkan jumlah variabel prediktor
    """
    tuning_config = {
        1: {"lag": 1, "batch_size": 8, "units": 100, "dropout": 0.5, "epochs": 150},
        2: {"lag": 1, "batch_size": 32, "units": 150, "dropout": 0.3, "epochs": 100},
        3: {"lag": 1, "batch_size": 32, "units": 150, "dropout": 0.1, "epochs": 150},
        4: {"lag": 1, "batch_size": 32, "units": 50, "dropout": 0.1, "epochs": 100},
        5: {"lag": 1, "batch_size": 8, "units": 150, "dropout": 0.3, "epochs": 50}
    }
    
    # Jika lebih dari 5 variabel, gunakan parameter untuk 5 variabel
    if n_predictors > 5:
        return tuning_config[5]
    
    return tuning_config.get(n_predictors, tuning_config[4])  # Default ke 4 variabel jika tidak ditemukan

# =========================
# Fungsi Membuat Model Baru (dinamis sesuai parameter tuning)
# =========================
def build_lstm_model(n_features, units, dropout):
    model = Sequential()
    model.add(LSTM(units=units, return_sequences=False, input_shape=(1, n_features)))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
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
def build_univariate_lstm_model(sequence_length=5, units=50, dropout=0.2):
    """
    Membuat model LSTM untuk peramalan univariate (satu variabel)
    """
    model = Sequential()
    model.add(LSTM(units=units, return_sequences=False, input_shape=(sequence_length, 1)))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    return model

# =========================
# Fungsi untuk membuat sequences untuk univariate time series
# =========================
def create_univariate_sequences(data, sequence_length=5):
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
def train_predictor_models(X_original, selected_predictors, sequence_length=5):
    """
    Melatih model LSTM untuk setiap variabel prediktor
    """
    predictor_models = {}
    predictor_scalers = {}
    
    for i, predictor in enumerate(selected_predictors):
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
            model = build_univariate_lstm_model(sequence_length, units=30, dropout=0.2)
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
def generate_forecast(n_future, method="lstm"):
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
    
    if method == "lstm":
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
    
    else:
        # Method asumsi perubahan (original method)
        # Ini akan dijalankan di bagian terpisah dengan input asumsi
        return None, None

# =========================
# Sidebar Menu
# =========================
st.sidebar.title("📊 Dashboard Skripsi")
menu = st.sidebar.radio("Pilih Menu", ["Analisis Data", "Dashboard"])

# =========================
# Analisis Data
# =========================
if menu == "Analisis Data":
    st.header("📂 Upload Data Excel")
    uploaded_file = st.file_uploader("Upload file Excel", type=["xlsx", "xls"])

    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        st.subheader("📊 Preview Data")
        st.dataframe(df.head())
        
        # Informasi dasar dataset
        st.subheader("ℹ️ Informasi Dataset")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Jumlah Baris", len(df))
        with col2:
            st.metric("Jumlah Kolom", len(df.columns))
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        
        # Tampilkan statistik deskriptif
        st.subheader("📈 Statistik Deskriptif")
        st.dataframe(df.describe())
        
        # Cek apakah ada kolom 'Saham Inco' dan variabel prediktor
        if "Saham Inco" in df.columns:
            # Identifikasi variabel prediktor (semua kolom kecuali 'Saham Inco')
            predictor_vars = [col for col in df.columns if col != "Saham Inco"]
            
            if len(predictor_vars) > 0:
                st.subheader("📊 Time Series Analysis")
                st.info(f"💡 Menampilkan hubungan time series antara **Saham Inco** dengan variabel prediktor")
                
                # Pilihan untuk memilih variabel yang ingin ditampilkan
                selected_vars_analysis = st.multiselect(
                    "Pilih variabel prediktor untuk analisis:",
                    predictor_vars,
                    default=predictor_vars[:4] if len(predictor_vars) >= 4 else predictor_vars,
                    key="analysis_vars"
                )
                
                if selected_vars_analysis:
                    # ==========================================
                    # PLOT 1: Time Series Plot Individual
                    # ==========================================
                    st.subheader("📈 Time Series Plot - Individual Variables")
                    
                    # Buat subplot untuk setiap variabel
                    n_vars = len(selected_vars_analysis) + 1  # +1 untuk Saham Inco
                    cols = 2
                    rows = (n_vars + 1) // 2
                    
                    fig1, axes = plt.subplots(rows, cols, figsize=(15, 4*rows))
                    fig1.suptitle("Time Series Plot - Semua Variabel", fontsize=16, y=1.02)
                    
                    # Flatten axes jika perlu
                    if rows == 1:
                        axes = [axes] if cols == 1 else axes
                    else:
                        axes = axes.flatten()
                    
                    # Plot Saham Inco (target variable)
                    axes[0].plot(df.index, df['Saham Inco'], color='red', linewidth=2, alpha=0.8)
                    axes[0].set_title('Saham Inco (Target)', fontweight='bold')
                    axes[0].set_xlabel('Periode')
                    axes[0].set_ylabel('Nilai')
                    axes[0].grid(True, alpha=0.3)
                    
                    # Plot setiap variabel prediktor
                    colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
                    for i, var in enumerate(selected_vars_analysis):
                        ax_idx = i + 1
                        if ax_idx < len(axes):
                            color = colors[i % len(colors)]
                            axes[ax_idx].plot(df.index, df[var], color=color, linewidth=2, alpha=0.8)
                            axes[ax_idx].set_title(f'{var}', fontweight='bold')
                            axes[ax_idx].set_xlabel('Periode')
                            axes[ax_idx].set_ylabel('Nilai')
                            axes[ax_idx].grid(True, alpha=0.3)
                    
                    # Sembunyikan subplot yang tidak digunakan
                    for i in range(n_vars, len(axes)):
                        axes[i].set_visible(False)
                    
                    plt.tight_layout()
                    st.pyplot(fig1)
                    
                    # ==========================================
                    # PLOT 2: Comparison Plot (Saham Inco vs Each Predictor)
                    # ==========================================
                    st.subheader("🔍 Comparison Plot - Saham Inco vs Setiap Prediktor")
                    
                    # Buat subplot untuk perbandingan
                    n_predictors = len(selected_vars_analysis)
                    cols_comp = 2 if n_predictors > 1 else 1
                    rows_comp = (n_predictors + 1) // 2
                    
                    fig2, axes2 = plt.subplots(rows_comp, cols_comp, figsize=(15, 5*rows_comp))
                    fig2.suptitle("Comparison: Saham Inco vs Variabel Prediktor", fontsize=16, y=1.02)
                    
                    # Flatten axes jika perlu
                    if rows_comp == 1:
                        axes2 = [axes2] if cols_comp == 1 else axes2
                    else:
                        axes2 = axes2.flatten()
                    
                    for i, var in enumerate(selected_vars_analysis):
                        if i < len(axes2):
                            ax = axes2[i]
                            
                            # Plot dengan dual y-axis
                            ax_twin = ax.twinx()
                            
                            # Plot Saham Inco
                            line1 = ax.plot(df.index, df['Saham Inco'], color='red', linewidth=2, 
                                           label='Saham Inco', alpha=0.8)
                            ax.set_xlabel('Periode')
                            ax.set_ylabel('Saham Inco', color='red')
                            ax.tick_params(axis='y', labelcolor='red')
                            
                            # Plot variabel prediktor
                            color_pred = colors[i % len(colors)]
                            line2 = ax_twin.plot(df.index, df[var], color=color_pred, linewidth=2, 
                                               label=var, alpha=0.8, linestyle='--')
                            ax_twin.set_ylabel(var, color=color_pred)
                            ax_twin.tick_params(axis='y', labelcolor=color_pred)
                            
                            # Title dan grid
                            ax.set_title(f'Saham Inco vs {var}', fontweight='bold')
                            ax.grid(True, alpha=0.3)
                            
                            # Legend
                            lines1, labels1 = ax.get_legend_handles_labels()
                            lines2, labels2 = ax_twin.get_legend_handles_labels()
                            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
                    
                    # Sembunyikan subplot yang tidak digunakan
                    for i in range(n_predictors, len(axes2)):
                        axes2[i].set_visible(False)
                    
                    plt.tight_layout()
                    st.pyplot(fig2)
                    
                    # ==========================================
                    # PLOT 3: Correlation Heatmap
                    # ==========================================
                    st.subheader("🔥 Correlation Heatmap")
                    
                    # Hitung korelasi
                    correlation_vars = ['Saham Inco'] + selected_vars_analysis
                    corr_matrix = df[correlation_vars].corr()
                    
                    # Plot heatmap
                    fig3, ax3 = plt.subplots(figsize=(10, 8))
                    im = ax3.imshow(corr_matrix, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)
                    
                    # Set ticks dan labels
                    ax3.set_xticks(range(len(correlation_vars)))
                    ax3.set_yticks(range(len(correlation_vars)))
                    ax3.set_xticklabels(correlation_vars, rotation=45, ha='right')
                    ax3.set_yticklabels(correlation_vars)
                    
                    # Tambahkan nilai korelasi pada setiap cell
                    for i in range(len(correlation_vars)):
                        for j in range(len(correlation_vars)):
                            text = ax3.text(j, i, f'{corr_matrix.iloc[i, j]:.3f}',
                                          ha="center", va="center", color="black", fontweight='bold')
                    
                    # Colorbar
                    cbar = plt.colorbar(im, ax=ax3)
                    cbar.set_label('Korelasi', rotation=270, labelpad=20)
                    
                    ax3.set_title('Correlation Matrix: Saham Inco vs Variabel Prediktor', 
                                 fontweight='bold', pad=20)
                    plt.tight_layout()
                    st.pyplot(fig3)
                    
                    # ==========================================
                    # TABEL KORELASI
                    # ==========================================
                    st.subheader("📋 Tabel Korelasi dengan Saham Inco")
                    
                    # Buat tabel korelasi dengan Saham Inco
                    corr_with_target = df[selected_vars_analysis + ['Saham Inco']].corr()['Saham Inco'].drop('Saham Inco')
                    corr_df = pd.DataFrame({
                        'Variabel': corr_with_target.index,
                        'Korelasi dengan Saham Inco': corr_with_target.values,
                        'Kekuatan Korelasi': ['Sangat Kuat' if abs(x) >= 0.8 else 
                                             'Kuat' if abs(x) >= 0.6 else
                                             'Sedang' if abs(x) >= 0.4 else
                                             'Lemah' if abs(x) >= 0.2 else
                                             'Sangat Lemah' for x in corr_with_target.values]
                    })
                    
                    # Format tabel dengan warna
                    def color_correlation(val):
                        if abs(val) >= 0.8:
                            return 'background-color: #ff9999'  # Merah muda
                        elif abs(val) >= 0.6:
                            return 'background-color: #ffcc99'  # Oranye muda
                        elif abs(val) >= 0.4:
                            return 'background-color: #ffff99'  # Kuning muda
                        else:
                            return 'background-color: #ccffcc'  # Hijau muda
                    
                    styled_corr_df = corr_df.style.applymap(color_correlation, subset=['Korelasi dengan Saham Inco'])
                    st.dataframe(styled_corr_df)
                    
                   
                    # ==========================================
                    # DOWNLOAD CORRELATION DATA
                    # ==========================================
                    csv_corr = corr_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Tabel Korelasi (CSV)",
                        data=csv_corr,
                        file_name="correlation_analysis.csv",
                        mime="text/csv",
                        key="download_correlation_csv"
                    )
                    
                else:
                    st.warning("Pilih minimal 1 variabel prediktor untuk analisis time series.")
            else:
                st.warning("Tidak ditemukan variabel prediktor dalam dataset.")
        else:
            st.error("Dataset harus memiliki kolom target 'Saham Inco' untuk analisis time series.")
        

# =========================
# Dashboard Peramalan
# =========================
elif menu == "Dashboard":
    st.header("📈 Dashboard Peramalan")

    uploaded_file = st.file_uploader("Upload file Excel untuk Peramalan", type=["xlsx", "xls"])
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)

        # Pilih variabel prediktor (dinamis sesuai file)
        all_predictors = [col for col in df.columns if col != "Saham Inco"]
        selected_predictors = st.multiselect("Pilih variabel prediktor:", all_predictors, default=all_predictors[:4])

        if not selected_predictors:
            st.warning("Silakan pilih minimal 1 variabel prediktor.")
        elif "Saham Inco" not in df.columns:
            st.error("Excel harus memiliki kolom target 'Saham Inco'.")
        else:
            # Tampilkan parameter tuning yang akan digunakan
            n_predictors = len(selected_predictors)
            params = get_tuning_params(n_predictors)
            
            st.subheader("⚙️ Parameter Tuning yang Digunakan")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Jumlah Variabel", n_predictors)
                st.metric("Lag", params["lag"])
            with col2:
                st.metric("Batch Size", params["batch_size"])
                st.metric("Units LSTM", params["units"])
            with col3:
                st.metric("Dropout", params["dropout"])
                st.metric("Epochs", params["epochs"])

            if st.button("🚀 Jalankan Peramalan"):
                with st.spinner("Sedang melatih model..."):
                    X = df[selected_predictors].values
                    y = df[['Saham Inco']].values.flatten()

                    # Buat lag features
                    lag = params["lag"]
                    X_lag, y_current = create_lag_features(X, y, lag)

                    # Split data menjadi training dan testing
                    X_train, X_test, y_train, y_test = split_data(X_lag, y_current, train_ratio=0.8)

                    # Scaling hanya pada data training
                    scaler_X = MinMaxScaler()
                    scaler_y = MinMaxScaler()
                    
                    X_train_scaled = scaler_X.fit_transform(X_train)
                    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
                    
                    # Transform data testing menggunakan scaler dari training
                    X_test_scaled = scaler_X.transform(X_test)
                    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

                    # Reshape untuk LSTM (samples, timesteps=1, features)
                    X_train_lstm = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
                    X_test_lstm = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])

                    # Bangun model LSTM untuk target
                    model = build_lstm_model(
                        n_features=n_predictors,
                        units=params["units"],
                        dropout=params["dropout"]
                    )

                    # Training model target
                    history = model.fit(
                        X_train_lstm, y_train_scaled,
                        epochs=params["epochs"],
                        batch_size=params["batch_size"],
                        validation_data=(X_test_lstm, y_test_scaled),
                        verbose=0
                    )

                    # Prediksi pada data training dan testing
                    y_train_pred_scaled = model.predict(X_train_lstm)
                    y_test_pred_scaled = model.predict(X_test_lstm)
                    
                    # Inverse transform
                    y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).flatten()
                    y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).flatten()

                    # ========== LATIH MODEL UNTUK VARIABEL PREDIKTOR ==========
                    st.info("🔄 Melatih model LSTM untuk variabel prediktor...")
                    predictor_models, predictor_scalers = train_predictor_models(X, selected_predictors, sequence_length=5)

                    # Simpan ke session_state
                    st.session_state["scaler_X"] = scaler_X
                    st.session_state["scaler_y"] = scaler_y
                    st.session_state["model"] = model
                    st.session_state["selected_predictors"] = selected_predictors
                    st.session_state["params"] = params
                    st.session_state["last_input"] = X_test_scaled[-1]  # Ambil input terakhir untuk forecasting
                    st.session_state["history"] = history
                    st.session_state["predictor_models"] = predictor_models
                    st.session_state["predictor_scalers"] = predictor_scalers
                    st.session_state["X_original"] = X  # Simpan data original untuk forecasting
                    st.session_state["model_trained"] = True  # Flag bahwa model sudah dilatih

                    # Hitung MAPE
                    mape_train = mean_absolute_percentage_error(y_train, y_train_pred) * 100
                    mape_test = mean_absolute_percentage_error(y_test, y_test_pred) * 100

                    # Plot hasil training
                    st.subheader("📊 Hasil Peramalan")
                    
                    # Plot perbandingan actual vs predicted
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                    
                    # Training data
                    ax1.plot(y_train, label="Aktual Training", marker='o', alpha=0.7)
                    ax1.plot(y_train_pred, label="Prediksi Training", marker='x', alpha=0.7)
                    ax1.set_title("Training Data: Aktual vs Prediksi")
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                    
                    # Testing data
                    ax2.plot(y_test, label="Aktual Testing", marker='o', alpha=0.7)
                    ax2.plot(y_test_pred, label="Prediksi Testing", marker='x', alpha=0.7)
                    ax2.set_title("Testing Data: Aktual vs Prediksi")
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Tampilkan metrik
                    col1, col2 = st.columns(2)
                    with col1:
                        st.success(f"MAPE Training: {mape_train:.2f}%")
                    with col2:
                        st.success(f"MAPE Testing: {mape_test:.2f}%")

                    # Plot training history
                    st.subheader("📈 Training History")
                    fig_hist, ax_hist = plt.subplots(figsize=(10, 4))
                    ax_hist.plot(history.history['loss'], label='Training Loss')
                    ax_hist.plot(history.history['val_loss'], label='Validation Loss')
                    ax_hist.set_title('Model Loss')
                    ax_hist.set_xlabel('Epoch')
                    ax_hist.set_ylabel('Loss')
                    ax_hist.legend()
                    ax_hist.grid(True, alpha=0.3)
                    st.pyplot(fig_hist)

                    st.success("✅ Model berhasil dilatih! Sekarang Anda dapat melakukan peramalan di bagian bawah.")

    # =========================
    # BAGIAN FORECASTING TERPISAH - HANYA MUNCUL JIKA MODEL SUDAH DILATIH
    # =========================
    if st.session_state.get("model_trained", False):
        st.markdown("---")
        st.header("🔮 Peramalan ke Depan")
        
        # Tabs untuk berbagai jenis forecasting
        tab1, tab2 = st.tabs(["🤖 LSTM Auto-Forecast", "⚙️ Manual Forecast dengan Asumsi"])
        
        # Tab 1: LSTM Auto-Forecast
        with tab1:
            st.subheader("🤖 Peramalan Otomatis menggunakan LSTM")
            st.info("💡 Metode ini menggunakan model LSTM terpisah untuk meramalkan setiap variabel prediktor secara otomatis")
            
            # Slider untuk LSTM forecast - menggunakan key yang unik
            n_future_lstm = st.slider(
                "Pilih jumlah periode ke depan:", 
                min_value=1, 
                max_value=24, 
                value=6, 
                key="lstm_forecast_slider"
            )
            
            if st.button("🚀 Generate LSTM Forecast", key="lstm_forecast_btn"):
                with st.spinner("Generating LSTM forecast..."):
                    future_preds, forecasted_predictors = generate_forecast(n_future_lstm, method="lstm")
                    
                    if future_preds is not None:
                        # Simpan hasil forecast ke session_state
                        st.session_state["lstm_results"] = {
                            "future_preds": future_preds,
                            "forecasted_predictors": forecasted_predictors,
                            "n_periods": n_future_lstm
                        }
                        
                        # Tampilkan hasil forecast
                        st.subheader("📅 Hasil Ramalan LSTM")
                        
                        # Buat DataFrame hasil yang lebih detail
                        hasil_data = {"Periode": range(1, n_future_lstm+1), "Ramalan_Saham_Inco": future_preds}
                        
                        # Tambahkan hasil forecast untuk setiap variabel prediktor
                        selected_predictors = st.session_state["selected_predictors"]
                        for predictor in selected_predictors:
                            hasil_data[f"Forecast_{predictor}"] = forecasted_predictors[predictor]
                        
                        hasil_df = pd.DataFrame(hasil_data)
                        st.dataframe(hasil_df)
                        
                        # Plot hasil ramalan ke depan untuk target
                        st.subheader("📈 Visualisasi Ramalan Target (Saham Inco)")
                        fig2, ax2 = plt.subplots(figsize=(12, 6))
                        ax2.plot(range(1, n_future_lstm+1), future_preds, marker='o', color='orange', linewidth=2, markersize=8)
                        ax2.set_xlabel("Periode ke Depan")
                        ax2.set_ylabel("Ramalan Saham Inco")
                        ax2.set_title("Forecast Saham Inco ke Depan (Dengan LSTM untuk Prediktor)")
                        ax2.grid(True, alpha=0.3)
                        
                        # Tambahkan nilai pada setiap titik
                        for i, value in enumerate(future_preds):
                            ax2.annotate(f'{value:.2f}', (i+1, value), textcoords="offset points", xytext=(0,10), ha='center')
                        
                        st.pyplot(fig2)

                        # Plot hasil ramalan untuk variabel prediktor
                        st.subheader("📊 Visualisasi Ramalan Variabel Prediktor")
                        
                        # Buat subplot untuk setiap prediktor
                        n_predictors_plot = len(selected_predictors)
                        cols = 2 if n_predictors_plot > 1 else 1
                        rows = (n_predictors_plot + 1) // 2
                        
                        fig3, axes = plt.subplots(rows, cols, figsize=(15, 4*rows))
                        if n_predictors_plot == 1:
                            axes = [axes]
                        elif rows == 1:
                            axes = [axes]
                        else:
                            axes = axes.flatten()
                        
                        for i, predictor in enumerate(selected_predictors):
                            if i < len(axes):
                                ax = axes[i]
                                ax.plot(range(1, n_future_lstm+1), forecasted_predictors[predictor], 
                                       marker='s', color=f'C{i}', linewidth=2, markersize=6)
                                ax.set_xlabel("Periode ke Depan")
                                ax.set_ylabel(f"Forecast {predictor}")
                                ax.set_title(f"Forecast {predictor}")
                                ax.grid(True, alpha=0.3)
                                
                                # Tambahkan nilai pada setiap titik
                                for j, value in enumerate(forecasted_predictors[predictor]):
                                    ax.annotate(f'{value:.2f}', (j+1, value), 
                                              textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
                        
                        # Sembunyikan subplot yang tidak digunakan
                        for i in range(n_predictors_plot, len(axes)):
                            axes[i].set_visible(False)
                        
                        plt.tight_layout()
                        st.pyplot(fig3)

                        # Download hasil LSTM forecast
                        csv_lstm = hasil_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download LSTM Forecast (CSV)",
                            data=csv_lstm,
                            file_name=f"lstm_forecast_{n_future_lstm}_periods.csv",
                            mime="text/csv",
                            key="download_lstm_csv"
                        )
            
            # Tampilkan hasil yang tersimpan jika ada
            elif "lstm_results" in st.session_state:
                st.info(f"📊 Hasil forecast LSTM terakhir ({st.session_state['lstm_results']['n_periods']} periode) masih tersimpan di memori.")
                
                if st.button("👀 Tampilkan Kembali Hasil LSTM", key="show_lstm_results"):
                    results = st.session_state["lstm_results"]
                    future_preds = results["future_preds"]
                    forecasted_predictors = results["forecasted_predictors"]
                    n_periods = results["n_periods"]
                    
                    # Tampilkan hasil yang tersimpan
                    st.subheader("📅 Hasil Ramalan LSTM (Tersimpan)")
                    
                    hasil_data = {"Periode": range(1, n_periods+1), "Ramalan_Saham_Inco": future_preds}
                    selected_predictors = st.session_state["selected_predictors"]
                    for predictor in selected_predictors:
                        hasil_data[f"Forecast_{predictor}"] = forecasted_predictors[predictor]
                    
                    hasil_df = pd.DataFrame(hasil_data)
                    st.dataframe(hasil_df)

        # Tab 2: Manual Forecast dengan Asumsi
        with tab2:
            st.subheader("⚙️ Peramalan Manual dengan Asumsi Perubahan")
            st.info("💡 Metode ini menggunakan asumsi perubahan yang Anda tentukan untuk setiap variabel prediktor")
            
            selected_predictors = st.session_state["selected_predictors"]
            
            # Input asumsi perubahan
            st.subheader("⚙️ Asumsi Perubahan Variabel per Periode")
            asumsi = {}
            for var in selected_predictors:
                asumsi[var] = st.number_input(f"Perubahan {var} per periode (%)", value=0.0, step=0.1, key=f"asumsi_{var}")
            
            # Slider untuk manual forecast
            n_future_manual = st.slider(
                "Pilih jumlah periode ke depan:", 
                min_value=1, 
                max_value=24, 
                value=6, 
                key="manual_forecast_slider"
            )

            if st.button("🚀 Generate Manual Forecast", key="manual_forecast_btn"):
                with st.spinner("Generating manual forecast..."):
                    model = st.session_state["model"]
                    scaler_X = st.session_state["scaler_X"]
                    scaler_y = st.session_state["scaler_y"]
                    
                    # Gunakan data terakhir sebagai starting point  
                    current_input = st.session_state["last_input"].copy()  # Input saat ini (periode terakhir)
                    
                    future_preds = []
                    future_inputs = []
                    
                    for period in range(n_future_manual):
                        # Reshape untuk prediksi (1 sample, 1 timestep, n_features)
                        input_reshaped = current_input.reshape(1, 1, -1)
                        
                        # Prediksi nilai target
                        pred_scaled = model.predict(input_reshaped, verbose=0)
                        pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
                        future_preds.append(pred)
                        
                        # Simpan input saat ini untuk visualisasi (dalam bentuk original)
                        input_original = scaler_X.inverse_transform(current_input.reshape(1, -1))[0]
                        future_inputs.append(input_original)
                        
                        # Update input untuk periode berikutnya berdasarkan asumsi perubahan
                        for i, var in enumerate(selected_predictors):
                            # Konversi persentase ke desimal dan apply perubahan
                            change = asumsi[var] / 100.0
                            current_input[i] = current_input[i] * (1 + change)

                    # Tampilkan hasil forecast
                    st.subheader("📅 Hasil Ramalan Manual")
                    
                    # Buat DataFrame hasil
                    hasil_data = {"Periode": range(1, n_future_manual+1), "Ramalan_Saham_Inco": future_preds}
                    
                    # Tambahkan prediksi variabel input
                    for i, var in enumerate(selected_predictors):
                        hasil_data[f"Prediksi_{var}"] = [inp[i] for inp in future_inputs]
                    
                    hasil_df = pd.DataFrame(hasil_data)
                    st.dataframe(hasil_df)
                    
                    # Plot hasil ramalan ke depan
                    st.subheader("📈 Visualisasi Ramalan")
                    fig2, ax2 = plt.subplots(figsize=(12, 6))
                    ax2.plot(range(1, n_future_manual+1), future_preds, marker='o', color='green', linewidth=2, markersize=8)
                    ax2.set_xlabel("Periode ke Depan")
                    ax2.set_ylabel("Ramalan Saham Inco")
                    ax2.set_title("Forecast Saham Inco ke Depan (Manual dengan Asumsi)")
                    ax2.grid(True, alpha=0.3)
                    
                    # Tambahkan nilai pada setiap titik
                    for i, value in enumerate(future_preds):
                        ax2.annotate(f'{value:.2f}', (i+1, value), textcoords="offset points", xytext=(0,10), ha='center')
                    
                    st.pyplot(fig2)
                    
                    # Download hasil manual forecast
                    csv_manual = hasil_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Manual Forecast (CSV)",
                        data=csv_manual,
                        file_name=f"manual_forecast_{n_future_manual}_periods.csv",
                        mime="text/csv",
                        key="download_manual_csv"
                    )
                    
                    # Simpan hasil manual ke session_state
                    st.session_state["manual_results"] = {
                        "future_preds": future_preds,
                        "future_inputs": future_inputs,
                        "n_periods": n_future_manual,
                        "asumsi": asumsi.copy()
                    }

    # =========================
    # BAGIAN PERBANDINGAN HASIL (Opsional)
    # =========================
    if st.session_state.get("model_trained", False):
        if "lstm_results" in st.session_state and "manual_results" in st.session_state:
            st.markdown("---")
            st.header("🔍 Perbandingan Hasil Forecasting")
            
            lstm_results = st.session_state["lstm_results"]
            manual_results = st.session_state["manual_results"]
            
            # Pastikan jumlah periode sama untuk perbandingan
            min_periods = min(lstm_results["n_periods"], manual_results["n_periods"])
            
            if st.button("📊 Tampilkan Perbandingan", key="comparison_btn"):
                # Data untuk perbandingan
                lstm_preds = lstm_results["future_preds"][:min_periods]
                manual_preds = manual_results["future_preds"][:min_periods]
                
                # Buat DataFrame perbandingan
                comparison_df = pd.DataFrame({
                    "Periode": range(1, min_periods+1),
                    "LSTM_Forecast": lstm_preds,
                    "Manual_Forecast": manual_preds,
                    "Selisih": [abs(l-m) for l, m in zip(lstm_preds, manual_preds)],
                    "Selisih_Persen": [abs(l-m)/((l+m)/2)*100 for l, m in zip(lstm_preds, manual_preds)]
                })
                
                st.subheader("📋 Tabel Perbandingan")
                st.dataframe(comparison_df)
                
                # Plot perbandingan
                st.subheader("📈 Grafik Perbandingan")
                fig_comp, ax_comp = plt.subplots(figsize=(12, 6))
                
                ax_comp.plot(range(1, min_periods+1), lstm_preds, 
                            marker='o', color='orange', linewidth=2, markersize=8, 
                            label='LSTM Auto-Forecast', alpha=0.8)
                ax_comp.plot(range(1, min_periods+1), manual_preds, 
                            marker='s', color='green', linewidth=2, markersize=8, 
                            label='Manual Forecast', alpha=0.8)
                
                ax_comp.set_xlabel("Periode ke Depan")
                ax_comp.set_ylabel("Ramalan Saham Inco")
                ax_comp.set_title("Perbandingan LSTM Auto-Forecast vs Manual Forecast")
                ax_comp.legend()
                ax_comp.grid(True, alpha=0.3)
                
                st.pyplot(fig_comp)
                
                # Download perbandingan
                csv_comparison = comparison_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Perbandingan (CSV)",
                    data=csv_comparison,
                    file_name=f"comparison_forecast_{min_periods}_periods.csv",
                    mime="text/csv",
                    key="download_comparison_csv"
                )
                
                # Insight singkat
                avg_diff = comparison_df["Selisih"].mean()
                avg_diff_pct = comparison_df["Selisih_Persen"].mean()
                
                st.info(f"📊 **Insight Perbandingan:**")
                st.info(f"• Rata-rata selisih absolut: {avg_diff:.2f}")
                st.info(f"• Rata-rata selisih persentase: {avg_diff_pct:.2f}%")