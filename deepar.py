# backend_deepar.py

import numpy as np
import pandas as pd
import torch, random, warnings
from gluonts.dataset.common import ListDataset
from gluonts.torch.model.deepar import DeepAREstimator
from gluonts.evaluation.backtest import make_evaluation_predictions
import optuna
from sklearn.metrics import r2_score

warnings.filterwarnings('ignore')
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# -------- Aux --------
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true).ravel(), np.array(y_pred).ravel()
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true != 0)
    if not np.any(mask):
        return np.nan
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]
    mape = np.mean(np.abs((y_true_filtered - y_pred_filtered) / y_true_filtered)) * 100
    return mape

def minmax_normalize(series):
    min_val = series.min()
    max_val = series.max()
    if max_val - min_val == 0:
        return series - min_val
    return (series - min_val) / (max_val - min_val)

# -------- Función principal para Streamlit --------
def entrenar_y_predecir_deepar(cc: str):
    # ==== CARGA Y PREPARACIÓN (idéntico a tu código, pero para cc genérico) ====
    data_df = pd.read_excel("union_CC_por_fecha.xlsx")
    data_df['Fecha'] = pd.to_datetime(data_df['Fecha'])
    data_df = data_df.sort_values(['CC_anon', 'Fecha']).reset_index(drop=True)
    data_df['mes'] = data_df['Fecha'].dt.month
    data_df['año'] = data_df['Fecha'].dt.year

    subdf = data_df[data_df["CC_anon"] == cc].copy()
    if subdf.empty:
        raise ValueError(f"No se encontraron datos para el CC '{cc}' en union_CC_por_fecha.xlsx")

    # Normalización
    subdf['Facturacion_norm'] = minmax_normalize(subdf['Facturacion'])
    subdf['HC_norm'] = minmax_normalize(subdf['HC'])
    subdf = subdf.set_index('Fecha').sort_index()

    inferred_freq = pd.infer_freq(subdf.index)
    freq = inferred_freq if inferred_freq else "M"

    # ==== SPLIT ====
    prediction_length = 3
    context_length = prediction_length * 3
    val_length = prediction_length
    test_length = prediction_length

    target = subdf["Facturacion_norm"].values.astype(float)
    hc_feat = subdf["HC_norm"].values.astype(float)
    mes_feat = subdf["mes"].values.astype(float)
    año_feat = subdf["año"].values.astype(float)
    start = pd.Timestamp(subdf.index[0])
    N = len(target)
    train_length = N - val_length - test_length
    if train_length <= 0:
        raise ValueError(f"La serie '{cc}' es demasiado corta para el split DeepAR.")

    # Splits
    train_target = target[:train_length]
    val_target = target[:train_length+val_length]
    test_target = target

    train_feat = [mes_feat[:train_length], año_feat[:train_length], hc_feat[:train_length]]
    val_feat = [mes_feat[:train_length+val_length], año_feat[:train_length+val_length], hc_feat[:train_length+val_length]]
    test_feat = [mes_feat, año_feat, hc_feat]

    train_entry = {"start": start, "target": train_target, "feat_dynamic_real": train_feat}
    val_entry = {"start": start, "target": val_target, "feat_dynamic_real": val_feat}
    test_entry = {"start": start, "target": test_target, "feat_dynamic_real": test_feat}

    train_dataset = ListDataset([train_entry], freq=freq)
    val_dataset = ListDataset([val_entry], freq=freq)
    test_dataset = ListDataset([test_entry], freq=freq)

    # ==== OPTUNA (tu objective, casi igual) ====
    prediction_length = val_length  # para validación

    def objective(trial):
        num_layers = trial.suggest_int("num_layers", 1, 3)
        hidden_size = trial.suggest_int("hidden_size", 16, 128, log=True)
        dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.2)
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        batch_size = 24
        scaling = True
        estimator = DeepAREstimator(
            prediction_length=prediction_length,
            context_length=context_length,
            freq=freq,
            num_layers=num_layers,
            hidden_size=hidden_size,
            dropout_rate=dropout_rate,
            scaling=scaling,
            lr=lr,
            batch_size=batch_size,
            trainer_kwargs=dict(
                accelerator="auto",  # usa GPU si hay
                max_epochs=30,
                devices=1,
                enable_checkpointing=True,
                enable_progress_bar=False,
                log_every_n_steps=20,
                gradient_clip_val=1.0,
            ),
        )
        try:
            predictor = estimator.train(train_dataset)
            forecast_it, ts_it = make_evaluation_predictions(
                dataset=val_dataset, predictor=predictor, num_samples=100
            )
            forecasts = list(forecast_it)
            tss = list(ts_it)
            ts = tss[0]
            pred = forecasts[0]
            y_true = ts.values[-val_length:].ravel()
            y_pred = pred.quantile(0.5)[-val_length:].ravel()

            if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
                return 1e6
            if np.any(np.isinf(y_true)) or np.any(np.isinf(y_pred)):
                return 1e6
            if len(y_true) != len(y_pred) or len(y_true) == 0:
                return 1e6

            mape = mean_absolute_percentage_error(y_true, y_pred)
            if not np.isfinite(mape) or mape < 0 or mape > 1000:
                return 1e6
            return mape
        except Exception:
            return 1e6

    from optuna.pruners import MedianPruner
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    sampler = optuna.samplers.GPSampler()
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name=f"deepar_{cc}"
    )
    study.optimize(
        objective,
        n_trials=25,
        timeout=1800,
        show_progress_bar=False,
        n_jobs=1,
    )
    best_params = study.best_params
    best_mape = study.best_value

    # ==== TRAIN FINAL + TEST ====
    final_train_target = target[:train_length + val_length]
    final_train_feat = [
        mes_feat[:train_length + val_length],
        año_feat[:train_length + val_length],
        hc_feat[:train_length + val_length]
    ]
    final_train_entry = {
        "start": start,
        "target": final_train_target,
        "feat_dynamic_real": final_train_feat
    }
    final_train_dataset = ListDataset([final_train_entry], freq=freq)

    final_estimator = DeepAREstimator(
        prediction_length=test_length,
        context_length=context_length,
        freq=freq,
        num_layers=best_params["num_layers"],
        hidden_size=best_params["hidden_size"],
        dropout_rate=best_params["dropout_rate"],
        scaling=True,
        lr=best_params["lr"],
        batch_size=24,
        trainer_kwargs=dict(
            accelerator="auto",
            max_epochs=50,
            devices=1,
            enable_checkpointing=True,
            enable_progress_bar=False,
            log_every_n_steps=20,
            gradient_clip_val=1.0,
        ),
    )
    predictor = final_estimator.train(final_train_dataset)
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_dataset, predictor=predictor, num_samples=200
    )
    forecasts = list(forecast_it)
    tss = list(ts_it)
    ts = tss[0]
    pred = forecasts[0]

    y_true = np.array(ts.values[-test_length:]).ravel()
    y_pred = np.array(pred.quantile(0.5))[-test_length:].ravel()

    mape_test = mean_absolute_percentage_error(y_true, y_pred)
    r2_test = r2_score(y_true, y_pred)

    # DataFrame para la UI: fechas + valores normalizados
    ts_index = ts.index.to_timestamp() if hasattr(ts.index, "to_timestamp") else pd.to_datetime(ts.index)
    df_pred = pd.DataFrame({
        "ds": ts_index[-test_length:],
        "y": y_true,
        "y_hat": y_pred
    })

    metricas = {
        "MAE": float(np.mean(np.abs(y_true - y_pred))),
        "MAPE": float(mape_test),
        "RMSE": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "R2": float(r2_test),
        "best_params": best_params,
        "best_mape_val": float(best_mape)
    }

    return df_pred, metricas
