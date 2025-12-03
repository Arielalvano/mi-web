# backend_tft.py

import numpy as np
import pandas as pd
import torch
import optuna

from neuralforecast import NeuralForecast
from neuralforecast.models import TFT
from neuralforecast.losses.pytorch import QuantileLoss
from sklearn.metrics import r2_score

# ====== CONFIG GLOBAL ======
h = 6
input_size = 12
futr_exog_list = ['month', 'quarter', 'year', 'HC', 'dias_laborables_mes']
FREQ = 'M'
EXCEL_PATH = 'df_raw_dias_laborables.xlsx'  # ajusta si está en otra ruta

best_params_cc = {}
final_models_cc = {}


# ====== CARGA Y PREPROCESO ======
def cargar_df_completo():
    df_raw = pd.read_excel(EXCEL_PATH)
    df = (
        df_raw.rename(columns={
            'CC_anon': 'unique_id',
            'Fecha': 'ds',
            'Facturacion': 'y'
        })
        .sort_values(['unique_id', 'ds'])
        .reset_index(drop=True)
    )
    df['ds'] = pd.to_datetime(df['ds'])
    df['month'] = df['ds'].dt.month
    df['quarter'] = df['ds'].dt.quarter
    df['year'] = df['ds'].dt.year - 2017
    # Se asume que df_raw ya tiene 'HC' y 'dias_laborables_mes'
    return df


# ====== OBJECTIVE (COPIADA 100%) ======
def objective(trial, train_df, valid_df, h, input_size, futr_exog_list):
    n_head = trial.suggest_categorical("n_head", [1, 2, 4, 8])
    hidden_size_multiplier = trial.suggest_int("hidden_size_multiplier", 16 // n_head, 128 // n_head)
    hidden_size = hidden_size_multiplier * n_head
    dropout = trial.suggest_float("dropout", 0.05, 0.5)
    attn_dropout = trial.suggest_float("attn_dropout", 0.05, 0.5)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 12, 16, 24, 32])
    max_steps = trial.suggest_int("max_steps", 200, 800)

    model = TFT(
        h=h,
        input_size=input_size,
        hidden_size=hidden_size,
        n_head=n_head,
        dropout=dropout,
        attn_dropout=attn_dropout,
        futr_exog_list=futr_exog_list,
        scaler_type='robust',
        loss=QuantileLoss(q=0.5),
        learning_rate=learning_rate,
        max_steps=max_steps,
        batch_size=batch_size,
        random_seed=42,
        val_check_steps=50,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu'
    )

    nf = NeuralForecast(models=[model], freq='M')
    nf.fit(df=train_df)

    # Genera el DataFrame futuro con las fechas del test
    futr_df = nf.make_future_dataframe(df=train_df)
    # Aquí solo selecciona las fechas y series correspondientes al valid_df
    futr_df = futr_df.merge(valid_df[['unique_id', 'ds']], on=['unique_id', 'ds'], how='inner')

    # Asegura que las exógenas están presentes
    futr_df['month'] = futr_df['ds'].dt.month
    futr_df['quarter'] = futr_df['ds'].dt.quarter
    futr_df['year'] = futr_df['ds'].dt.year - 2017
    futr_df['HC'] = valid_df['HC'].values
    futr_df['dias_laborables_mes'] = valid_df['dias_laborables_mes'].values

    # Predicción
    preds = nf.predict(futr_df=futr_df)
    df_eval = valid_df.merge(preds.reset_index(), on=['unique_id', 'ds'], how='inner')

    # Evitar división por cero
    mask = df_eval['y'] != 0
    if mask.sum() == 0:
        return float('inf')

    mape = (np.abs(df_eval.loc[mask, 'y'] - df_eval.loc[mask, 'TFT_ql0.5']) /
            np.abs(df_eval.loc[mask, 'y']) * 100).mean()
    return mape


# ====== FUNCIÓN PARA LA INTERFAZ ======
def entrenar_y_predecir(cc: str, h_usuario: int | None = None):
    """
    Usa EXACTAMENTE tu objective, pero aplicado solo al CC que entra por parámetro.
    Devuelve:
      - df_pred: ['unique_id', 'ds', 'y', 'y_hat']
      - metricas: dict con MAE, MAPE, RMSE, R2, best_params
    """
    global best_params_cc, final_models_cc

    if h_usuario is not None:
        h_local = h_usuario
    else:
        h_local = h

    df = cargar_df_completo()

    # Filtrar solo el CC pedido
    df_cc = df[df['unique_id'] == cc].copy().reset_index(drop=True)
    if df_cc.empty:
        raise ValueError(f"No se encontraron datos para el CC '{cc}'")

    if len(df_cc) <= h_local:
        raise ValueError(f"La serie '{cc}' tiene <= {h_local} observaciones; ajusta el horizonte.")

    # Split train/test como en tu código original
    test_df = df_cc.tail(h_local).copy().reset_index(drop=True)
    train_df = df_cc.iloc[:-h_local].copy().reset_index(drop=True)

    # Reutilizamos los mismos diccionarios por compatibilidad
    data_train_cc = {cc: train_df}
    data_test_cc = {cc: test_df}

    # Optuna EXACTAMENTE igual, pero solo para ese CC
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(
        lambda trial: objective(trial, train_df, test_df, h_local, input_size, futr_exog_list),
        n_trials=30,
        show_progress_bar=False   # en backend no hace falta barra
    )

    best_params_cc[cc] = study.best_params

    # Entrenar modelo final igual que en tu bucle
    bp = study.best_params
    final_hidden_size = bp['hidden_size_multiplier'] * bp['n_head']
    final_model = TFT(
        h=h_local,
        input_size=input_size,
        hidden_size=final_hidden_size,
        n_head=bp['n_head'],
        dropout=bp['dropout'],
        attn_dropout=bp['attn_dropout'],
        futr_exog_list=futr_exog_list,
        scaler_type='robust',
        loss=QuantileLoss(q=0.5),
        learning_rate=bp['learning_rate'],
        max_steps=bp['max_steps'],
        batch_size=bp['batch_size'],
        random_seed=42,
        val_check_steps=50,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu'
    )
    nf_final = NeuralForecast(models=[final_model], freq=FREQ)
    nf_final.fit(df=train_df)
    final_models_cc[cc] = nf_final

    # Predicción y métricas EXACTAMENTE como en tu parte final
    nf = final_models_cc[cc]
    test_df = data_test_cc[cc]

    predicciones = nf.predict(futr_df=test_df)

    df_test_eval = test_df.merge(
        predicciones.reset_index(),
        on=['unique_id', 'ds'],
        how='inner'
    )

    mae = (df_test_eval['y'] - df_test_eval['TFT_ql0.5']).abs().mean()
    mape = ((df_test_eval['y'] - df_test_eval['TFT_ql0.5']).abs() / df_test_eval['y'] * 100).mean()
    rmse = np.sqrt(((df_test_eval['y'] - df_test_eval['TFT_ql0.5']) ** 2).mean())
    r2 = r2_score(df_test_eval['y'], df_test_eval['TFT_ql0.5'])

    metricas = {
        'MAE': float(mae),
        'MAPE': float(mape),
        'RMSE': float(rmse),
        'R2': float(r2),
        'best_params': best_params_cc[cc]
    }

    df_pred = df_test_eval[['unique_id', 'ds', 'y']].copy()
    df_pred['y_hat'] = df_test_eval['TFT_ql0.5'].values

    return df_pred, metricas
