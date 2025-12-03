# app.py

import time
import streamlit as st
from copia_de_ariel_tft_1_2 import entrenar_y_predecir as entrenar_tft
from deepar import entrenar_y_predecir_deepar

st.set_page_config(
    page_title="Forecast CC Dashboard",
    page_icon="📈",
    layout="wide"
)

# ============== HEADER ==============
st.markdown(
    """
    <style>
    .big-title {
        font-size: 32px;
        font-weight: 800;
        color: #1f4e79;
    }
    .subtitle {
        font-size: 16px;
        color: #6c757d;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="big-title">📊 Forecast Dashboard por Cost Center</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Entrena y compara modelos TFT vs DeepAR para un CC concreto, con Optuna bajo demanda.</p>',
    unsafe_allow_html=True
)

st.divider()

# ============== SIDEBAR / CONTROLES ==============
with st.sidebar:
    st.header("⚙️ Configuración")
    cc = st.text_input("Cost Center (CC)", value="TLTL499551", help="Debe coincidir con CC_anon / unique_id en tus ficheros")
    h = st.slider("Horizonte h (meses) para TFT", min_value=1, max_value=12, value=6, step=1)
    modo = st.radio(
        "Modo de ejecución",
        ["Un solo modelo", "Comparar TFT vs DeepAR"],
        index=0
    )
    modelo_unico = st.selectbox("Modelo (si eliges solo uno)", ["TFT", "DeepAR"])
    lanzar = st.button("Entrenar y predecir")

# ============== LÓGICA PRINCIPAL ==============
if lanzar:
    if not cc:
        st.warning("Introduce un CC en la barra lateral.")
    else:
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()

            with st.spinner("Entrenando modelos y calculando predicciones..."):
                # Progreso aproximado
                for i in range(0, 80, 5):
                    progress_bar.progress(i)
                    status_text.text(f"Progreso aproximado: {i}%")
                    time.sleep(0.15)

                resultados = {}

                if modo == "Un solo modelo":
                    if modelo_unico == "TFT":
                        df_tft, met_tft = entrenar_tft(cc, h)
                        resultados["TFT"] = (df_tft, met_tft)
                    else:
                        df_dp, met_dp = entrenar_y_predecir_deepar(cc)
                        resultados["DeepAR"] = (df_dp, met_dp)
                else:  # Comparar TFT vs DeepAR
                    df_tft, met_tft = entrenar_tft(cc, h)
                    resultados["TFT"] = (df_tft, met_tft)

                    progress_bar.progress(90)
                    status_text.text("Entrenando DeepAR...")

                    df_dp, met_dp = entrenar_y_predecir_deepar(cc)
                    resultados["DeepAR"] = (df_dp, met_dp)

                progress_bar.progress(100)
                status_text.text("Entrenamiento completado ✅")

            st.success("Proceso completado correctamente.")

            # ========== PESTAÑAS ==========
            if modo == "Un solo modelo":
                modelo = list(resultados.keys())[0]
                df_pred, metricas = resultados[modelo]

                tab_main, tab_table = st.tabs([f"🔍 Detalle {modelo}", "📄 Datos crudos"])

                with tab_main:
                    st.subheader(f"Resultados {modelo} para CC {cc}")

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("MAE", f"{metricas['MAE']:.4f}")
                    c2.metric("MAPE (%)", f"{metricas['MAPE']:.2f}")
                    c3.metric("RMSE", f"{metricas['RMSE']:.4f}")
                    c4.metric("R²", f"{metricas['R2']:.4f}")

                    with st.expander("Hiperparámetros óptimos"):
                        st.json(metricas.get("best_params", {}))

                    if "best_mape_val" in metricas:
                        st.caption(f"Mejor MAPE validación (DeepAR): {metricas['best_mape_val']:.2f}%")

                    st.markdown("#### Serie real vs predicción (test)")
                    df_plot = df_pred.sort_values("ds").set_index("ds")
                    st.line_chart(df_plot[["y", "y_hat"]])

                with tab_table:
                    st.markdown("#### Predicciones en detalle")
                    st.dataframe(df_pred, use_container_width=True)

            else:
                # COMPARACIÓN LADO A LADO
                st.subheader(f"Comparativa TFT vs DeepAR para CC {cc}")

                col_left, col_right = st.columns(2)

                # --- Izquierda: TFT ---
                if "TFT" in resultados:
                    df_tft, met_tft = resultados["TFT"]
                    with col_left:
                        st.markdown("###TFT")
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("MAE", f"{met_tft['MAE']:.4f}")
                        c2.metric("MAPE (%)", f"{met_tft['MAPE']:.2f}")
                        c3.metric("RMSE", f"{met_tft['RMSE']:.4f}")
                        c4.metric("R²", f"{met_tft['R2']:.4f}")

                        st.markdown("**Curva real vs predicción (test)**")
                        df_plot_tft = df_tft.sort_values("ds").set_index("ds")
                        st.line_chart(df_plot_tft[["y", "y_hat"]])

                # --- Derecha: DeepAR ---
                if "DeepAR" in resultados:
                    df_dp, met_dp = resultados["DeepAR"]
                    with col_right:
                        st.markdown("###DeepAR")
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("MAE", f"{met_dp['MAE']:.4f}")
                        c2.metric("MAPE (%)", f"{met_dp['MAPE']:.2f}")
                        c3.metric("RMSE", f"{met_dp['RMSE']:.4f}")
                        c4.metric("R²", f"{met_dp['R2']:.4f}")

                        st.markdown("**Curva real vs predicción (test)**")
                        df_plot_dp = df_dp.sort_values("ds").set_index("ds")
                        st.line_chart(df_plot_dp[["y", "y_hat"]])

                # Tabla resumen comparativa
                st.markdown("### 📊 Resumen numérico")

                rows = []
                for nombre, (_, met) in resultados.items():
                    rows.append({
                        "Modelo": nombre,
                        "MAE": met["MAE"],
                        "MAPE": met["MAPE"],
                        "RMSE": met["RMSE"],
                        "R2": met["R2"],
                    })
                import pandas as pd
                df_comp = pd.DataFrame(rows)
                st.dataframe(df_comp, use_container_width=True)

        except Exception as e:
            st.error(f"Error: {e}")
