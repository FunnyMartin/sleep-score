"""
app_web.py - Sleep Score Predictor, lokalni webova aplikace
Martin Silar, SPSE Jecna C4c

Spusteni:
    python app_web.py
Dostupne na http://localhost:5000
"""

import os
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'sleep_model.pkl')

_model_data = None


def load_model():
    global _model_data
    if not os.path.exists(MODEL_PATH):
        return False
    try:
        _model_data = joblib.load(MODEL_PATH)
        return True
    except Exception as e:
        print(f"Nacteni modelu selhalo: {e}")
        return False


def build_feature_vector(hr_list, steps, kcal, day_of_week):
    """
    Sestavi feature vektor shodny s MODEL_FEATURES v export_model.py.
    hr_list: [prev1, prev2, ..., prev7] (vcera az 7 dni zpet)
    """
    hr = np.array(hr_list, dtype=float)
    rolling3 = float(np.mean(hr[:3]))
    rolling7 = float(np.mean(hr))
    dow_sin = float(np.sin(2 * np.pi * day_of_week / 7.0))
    dow_cos = float(np.cos(2 * np.pi * day_of_week / 7.0))
    is_weekend = 1 if day_of_week >= 5 else 0

    return [[
        hr[0], hr[1], hr[2], hr[3], hr[4], hr[5], hr[6],
        rolling7, rolling3,
        dow_sin, dow_cos, is_weekend,
        float(steps), float(kcal),
    ]]


@app.route('/')
def index():
    loaded = _model_data is not None
    boundary = _model_data['boundary'] if loaded else 75.6
    return render_template('index.html', model_loaded=loaded, boundary=boundary)


@app.route('/predict', methods=['POST'])
def predict():
    if _model_data is None:
        return jsonify({'error': 'Model neni nacten. Spustte export_model.py.'}), 503

    try:
        d = request.get_json(force=True)

        hr_list = [float(d[f'hr{i}']) for i in range(1, 8)]
        steps = float(d.get('steps', 0))
        kcal = float(d.get('kcal', 0))
        dow = int(d['dow'])

        for i, v in enumerate(hr_list, 1):
            if not (30 <= v <= 130):
                return jsonify({'error': f'Noc {i}: HR {v} je mimo rozsah (30-130 BPM).'}), 400
        if steps < 0 or steps > 80000:
            return jsonify({'error': 'Kroky: zadej hodnotu 0-80000.'}), 400
        if kcal < 0 or kcal > 5000:
            return jsonify({'error': 'Kalorie: zadej hodnotu 0-5000.'}), 400

        feats = build_feature_vector(hr_list, steps, kcal, dow)
        clf = _model_data['model']
        thr = _model_data['threshold']

        proba = float(clf.predict_proba(feats)[0][1])
        prediction = 1 if proba >= thr else 0

        return jsonify({
            'prediction':  prediction,
            'label':       'Dobrá noc' if prediction == 1 else 'Špatná noc',
            'probability': round(proba * 100, 1),
            'threshold':   round(thr * 100, 1),
            'boundary':    _model_data['boundary'],
        })

    except (KeyError, ValueError) as e:
        return jsonify({'error': f'Neplatny vstup: {e}'}), 400
    except Exception as e:
        return jsonify({'error': f'Chyba: {e}'}), 500


@app.route('/model/info')
def model_info():
    if _model_data is None:
        return jsonify({'loaded': False})
    return jsonify({
        'loaded':    True,
        'threshold': round(_model_data['threshold'], 3),
        'boundary':  _model_data['boundary'],
        'accuracy':  round(_model_data.get('accuracy', 0) * 100, 1),
        'n_train':   _model_data.get('n_train'),
    })


if __name__ == '__main__':
    ok = load_model()
    if ok:
        print(f"Model nacten ({MODEL_PATH})")
        print(f"  Threshold: {_model_data['threshold']:.3f}")
        print(f"  Hranice skore: {_model_data['boundary']}")
    else:
        print(f"Model nenalezen ({MODEL_PATH})")
        print("Nejdrive spustte: python export_model.py")
    print("Server: http://localhost:5000")
    app.run(host='127.0.0.1', port=5000, debug=False)
