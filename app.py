# app.py

import streamlit as st
from copia_de_ariel_tft_1_2 import entrenar_y_predecir as entrenar_tft
from deepar import entrenar_y_predecir_deepar

st.set_page_config(page_title="Forecast por CC", layout="wide")

st.title("Predicción de facturación por Cost Center")

st.markdown(
    """
    Escribe un CC tal como aparece en los Excels y elige el modelo.
    - TFT usa df_raw_dias_laborables.xlsx y horizonte h seleccionable.
    - DeepAR usa union_CC_por_fecha.xlsx y horizonte interno fijo.
    """
)

col1, col2, col3 = st.columns(3)
with col1:
    cc = st.text_input("Cost Center (CC)", value="TLTL499551", help="Ej: TLTL499551")
with col2:
    h = st.slider("Horizonte h (meses) · TFT", min_value=1, max_value=12, value=6, step=1)
with col3:
    modelo = st.selectbox("Modelo", ["TFT", "DeepAR"])

if st.button("Entrenar y predecir"):
    if not cc:
        st.warning("Introduce un CC.")
    else:
        try:
            st.write(f"DEBUG · Modelo={modelo} · CC='{cc}' · h={h}")
            if modelo == "TFT":
                with st.spinner("Entrenando TFT y calculando predicciones..."):
                    df_pred, metricas = entrenar_tft(cc, h)
            else:  # DeepAR
                with st.spinner("Entrenando DeepAR y calculando predicciones..."):
                    df_pred, metricas = entrenar_y_predecir_deepar(cc)

            st.success("Proceso completado.")

            st.subheader(f"Predicciones (conjunto de test) - {modelo}")
            st.dataframe(df_pred, use_container_width=True)

            st.subheader("Métricas")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("MAE", f"{metricas['MAE']:.4f}")
            c2.metric("MAPE (%)", f"{metricas['MAPE']:.2f}")
            c3.metric("RMSE", f"{metricas['RMSE']:.4f}")
            c4.metric("R²", f"{metricas['R2']:.4f}")

            with st.expander("Hiperparámetros óptimos"):
                st.write(metricas.get('best_params', {}))
                if 'best_mape_val' in metricas:
                    st.caption(f"Mejor MAPE validación (DeepAR): {metricas['best_mape_val']:.2f}%")

            st.subheader("Real vs Predicción (test)")
            df_plot = df_pred.sort_values("ds").set_index("ds")
            st.line_chart(df_plot[["y", "y_hat"]])

        except Exception as e:
            st.error(f"Error: {e}")
