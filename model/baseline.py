"""Structured-input ESI research baseline. Free-text mapping is not implemented."""
from pathlib import Path
import json
import math

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, confusion_matrix, f1_score, log_loss,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

LABELS = np.arange(1, 6)
SOURCE_VITALS = [
    'triage_vital_hr', 'triage_vital_sbp', 'triage_vital_dbp',
    'triage_vital_rr', 'triage_vital_o2', 'triage_vital_temp',
]
NUMERIC_FEATURES = ['age'] + SOURCE_VITALS[:-1] + ['temperature_c', 'ambulance']
MODEL_PARAMETERS = {
    'learning_rate': 0.1,
    'max_iter': 80,
    'max_leaf_nodes': 15,
    'min_samples_leaf': 30,
    'l2_regularization': 1.0,
    'max_bins': 63,
    'early_stopping': False,
    'random_state': 42,
}


def normalize_gender(values):
    normalized = values.astype('string').str.strip().str.lower()
    return normalized.map({'female': 'Female', 'male': 'Male'}).fillna('Unknown')


def ambulance_status(values):
    """Unknown arrival is not a confirmed non-ambulance arrival."""
    normalized = values.astype('string').str.strip().str.lower()
    mapping = {
        'ambulance': 1.0,
        'car': 0.0, 'walk-in': 0.0, 'public transportation': 0.0,
        'wheelchair': 0.0, 'police': 0.0,
    }
    return normalized.map(mapping).astype('float32')


def prepare_training_features(raw, complaint_columns):
    """Apply fixed encodings only. No medians, scaling, or labels are learned."""
    required = ['age', 'gender', 'arrivalmode'] + SOURCE_VITALS + list(complaint_columns)
    absent = set(required) - set(raw.columns)
    if absent:
        raise ValueError(f'Missing source columns: {sorted(absent)}')
    numeric = {c: pd.to_numeric(raw[c], errors='raise').astype('float32')
               for c in ['age'] + SOURCE_VITALS[:-1]}
    numeric['temperature_c'] = ((pd.to_numeric(raw['triage_vital_temp'], errors='raise') - 32) * 5 / 9).astype('float32')
    numeric['ambulance'] = ambulance_status(raw['arrivalmode'])
    base = pd.DataFrame(numeric, index=raw.index)
    base['gender'] = normalize_gender(raw['gender'])
    for name in complaint_columns:
        values = raw[name].to_numpy(dtype='float32', copy=False)
        known = values[~np.isnan(values)]
        if (not np.isfinite(known).all() or (known < 0).any()
                or (known != np.floor(known)).any()):
            raise ValueError(f'Invalid complaint count in {name}')
    complaints = raw[list(complaint_columns)].to_numpy(dtype='float32', copy=True)
    missing = np.isnan(complaints).any(axis=1).astype('float32')
    np.clip(complaints, 0, 1, out=complaints)  # NaN remains NaN.
    encoded = pd.DataFrame(complaints, index=raw.index, columns=complaint_columns)
    encoded['complaint_missing'] = missing
    return pd.concat([base, encoded], axis=1)


def make_pipeline(complaint_columns):
    """All learned imputation is fitted by Pipeline.fit on its training rows."""
    preprocessing = ColumnTransformer([
        ('numeric', SimpleImputer(strategy='median', add_indicator=True,
                                  keep_empty_features=True), NUMERIC_FEATURES),
        ('gender', OneHotEncoder(categories=[['Female', 'Male', 'Unknown']],
                                handle_unknown='ignore', sparse_output=False,
                                dtype=np.float32), ['gender']),
        ('complaints', SimpleImputer(strategy='constant', fill_value=0,
                                     keep_empty_features=True),
         list(complaint_columns) + ['complaint_missing']),
    ], remainder='drop', sparse_threshold=0, n_jobs=1)
    return Pipeline([
        ('preprocessing', preprocessing),
        ('classifier', HistGradientBoostingClassifier(**MODEL_PARAMETERS)),
    ])


def asymmetric_cost_matrix(undertriage_penalty):
    """Illustrative squared ordinal cost, not a clinically validated policy."""
    penalty = float(undertriage_penalty)
    if not math.isfinite(penalty) or penalty < 1:
        raise ValueError('Under-triage penalty must be finite and at least 1.')
    gap = LABELS[None, :] - LABELS[:, None]  # predicted minus reference ESI
    return gap.astype(float) ** 2 * np.where(gap > 0, penalty, 1.0)


def predict_with_cost(probabilities, *, undertriage_penalty):
    probabilities = np.asarray(probabilities)
    if probabilities.ndim != 2 or probabilities.shape[1] != 5:
        raise ValueError('Expected probabilities for ESI 1 through 5 in that order.')
    risks = probabilities @ asymmetric_cost_matrix(undertriage_penalty)
    return LABELS[np.argmin(risks, axis=1)]


def metrics(reference, predicted, probabilities=None):
    reference = np.asarray(reference, dtype=int)
    predicted = np.asarray(predicted, dtype=int)
    precision, recall, f1, support = precision_recall_fscore_support(
        reference, predicted, labels=LABELS, zero_division=0)
    high = reference <= 2
    high_predicted = predicted <= 2
    result = {
        'n': int(len(reference)),
        'accuracy': float(accuracy_score(reference, predicted)),
        'qwk': float(cohen_kappa_score(reference, predicted, labels=LABELS, weights='quadratic')),
        'macro_f1': float(f1_score(reference, predicted, labels=LABELS, average='macro', zero_division=0)),
        'undertriage_rate': float(np.mean(predicted > reference)),
        'overtriage_rate': float(np.mean(predicted < reference)),
        'undertriage_by_two_or_more_rate': float(np.mean(predicted - reference >= 2)),
        'high_acuity_recall': float(high_predicted[high].mean()) if high.any() else None,
        'high_acuity_miss_rate': float((~high_predicted[high]).mean()) if high.any() else None,
        'high_acuity_precision': float(high[high_predicted].mean()) if high_predicted.any() else None,
        'confusion_matrix': confusion_matrix(reference, predicted, labels=LABELS).tolist(),
        'per_class': {
            str(label): {'precision': float(precision[i]), 'recall': float(recall[i]),
                         'f1': float(f1[i]), 'support': int(support[i])}
            for i, label in enumerate(LABELS)
        },
    }
    if probabilities is not None:
        probabilities = np.asarray(probabilities, dtype=float)
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
        observed = np.eye(5)[reference - 1]
        result['log_loss'] = float(log_loss(reference, probabilities, labels=LABELS))
        result['multiclass_brier_score'] = float(np.mean(np.sum((probabilities - observed) ** 2, axis=1)))
        high_probability = probabilities[:, :2].sum(axis=1)
        result['high_acuity_brier_score'] = float(np.mean((high_probability - high.astype(float)) ** 2))
        confidence = probabilities.max(axis=1)
        correct = LABELS[probabilities.argmax(axis=1)] == reference
        bins = np.minimum((confidence * 10).astype(int), 9)
        reliability = []
        for i in range(10):
            selected = bins == i
            if selected.any():
                reliability.append({
                    'lower': i / 10, 'upper': (i + 1) / 10,
                    'n': int(selected.sum()),
                    'mean_confidence': float(confidence[selected].mean()),
                    'accuracy': float(correct[selected].mean()),
                })
        result['argmax_reliability_bins'] = reliability
        result['argmax_ece_10_bins'] = float(sum(
            r['n'] / len(reference) * abs(r['accuracy'] - r['mean_confidence'])
            for r in reliability))
    return result


def intake_features(intake, complaint_labels, complaint_columns, *, complaints_missing=False):
    """Require explicit mapped labels; never infer categories from raw narrative."""
    if complaint_labels is None:
        if not complaints_missing:
            raise ValueError('Mapped complaint labels are required; the NLP mapper is not implemented.')
        complaint_labels = []
    if isinstance(complaint_labels, str):
        raise TypeError('Provide a list of category names, not a raw complaint sentence.')
    labels = set(complaint_labels)
    unknown = labels - set(complaint_columns)
    if unknown:
        raise ValueError(f'Unrecognized mapped complaint labels: {sorted(unknown)}')
    if complaints_missing and labels:
        raise ValueError('Missing complaints cannot also have mapped categories.')
    if not complaints_missing and not labels:
        raise ValueError('Use an explicit mapped category or explicitly mark complaints missing.')

    def number(key, low, high, *, required=False):
        value = intake.get(key)
        if value is None or value == '':
            if required:
                raise ValueError(f'{key} is required.')
            return np.nan
        value = float(value)
        if not math.isfinite(value) or value < low or value > high:
            raise ValueError(f'{key} must be between {low} and {high}.')
        return value

    age = number('age', 18, 120, required=True)
    row = {
        'age': age,
        'triage_vital_hr': number('hr', 0, 350),
        'triage_vital_sbp': number('sbp', 0, 300),
        'triage_vital_dbp': number('dbp', 0, 200),
        'triage_vital_rr': number('rr', 0, 100),
        'triage_vital_o2': number('spo2', 0, 100),
        'temperature_c': number('temp', 20, 50),
        'ambulance': float(ambulance_status(pd.Series([intake.get('arrival')])).iloc[0]),
        'gender': normalize_gender(pd.Series([intake.get('gender')])).iloc[0],
        **{name: np.nan if complaints_missing else float(name in labels) for name in complaint_columns},
        'complaint_missing': float(complaints_missing),
    }
    return pd.DataFrame([row])


def score_intake(intake, complaint_labels, *, model_dir=None,
                 complaints_missing=False, undertriage_penalty=None):
    """Research inference with explicit categories. The web prototype is not connected."""
    model_dir = Path(model_dir) if model_dir else Path(__file__).resolve().parent
    manifest = json.loads((model_dir / 'manifest.json').read_text(encoding='utf-8'))
    x = intake_features(intake, complaint_labels, manifest['complaint_columns'],
                        complaints_missing=complaints_missing)
    model = joblib.load(model_dir / 'model.joblib')
    probabilities = model.predict_proba(x)
    argmax = int(LABELS[probabilities[0].argmax()])
    chosen = argmax if undertriage_penalty is None else int(predict_with_cost(
        probabilities, undertriage_penalty=undertriage_penalty)[0])
    coverage_notes = []
    for name in NUMERIC_FEATURES:
        value = float(x[name].iloc[0])
        bounds = manifest['observed_feature_ranges'][name]
        if np.isfinite(value) and bounds['min'] is not None and (
                value < bounds['min'] or value > bounds['max']):
            coverage_notes.append(f'{name} is outside the observed training range.')
    if x['gender'].iloc[0] == 'Unknown':
        coverage_notes.append('Unknown/Other gender has no observed training examples.')
    if pd.isna(x['ambulance'].iloc[0]):
        coverage_notes.append('Arrival method is represented as unknown, not non-ambulance.')
    return {
        'recorded_esi_prediction': chosen,
        'argmax_prediction': argmax,
        'probabilities': {str(k): float(v) for k, v in zip(LABELS, probabilities[0])},
        'decision_rule': 'argmax' if undertriage_penalty is None else 'illustrative_asymmetric_squared_cost',
        'undertriage_penalty': undertriage_penalty,
        'probabilities_calibrated': False,
        'unused_prototype_fields': ['pain', 'complaint narrative'],
        'coverage_notes': coverage_notes,
        'scope': 'Research baseline for adults; category labels must be supplied explicitly.',
    }
