"""
Zakladni testy
Martin Silar, SPSE Jecna C4c
"""

import sys
import os
import unittest
from src.app_web import app, build_feature_vector
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class FeatureTests(unittest.TestCase):

    def test_output_shape(self):
        hr = [58.0, 59.0, 57.5, 60.0, 58.0, 61.0, 59.0]
        result = build_feature_vector(hr, 8000, 400, 0)
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]), 14)

    def test_rolling7_value(self):
        hr = [55.0] * 7
        feats = build_feature_vector(hr, 0, 0, 1)
        rolling7 = feats[0][7]
        self.assertAlmostEqual(rolling7, 55.0, places=4)

    def test_rolling3_uses_first_three(self):
        hr = [60.0, 60.0, 60.0, 50.0, 50.0, 50.0, 50.0]
        feats = build_feature_vector(hr, 0, 0, 0)
        rolling3 = feats[0][8]
        self.assertAlmostEqual(rolling3, 60.0, places=4)

    def test_weekend_saturday(self):
        hr = [58.0] * 7
        feats = build_feature_vector(hr, 5000, 300, 5)
        self.assertEqual(feats[0][11], 1)

    def test_weekday_monday(self):
        hr = [58.0] * 7
        feats = build_feature_vector(hr, 5000, 300, 0)
        self.assertEqual(feats[0][11], 0)

    def test_steps_in_features(self):
        hr = [57.0] * 7
        feats_low = build_feature_vector(hr, 1000, 200, 2)
        feats_high = build_feature_vector(hr, 20000, 200, 2)
        self.assertNotEqual(feats_low[0][12], feats_high[0][12])

    def test_kcal_in_features(self):
        hr = [57.0] * 7
        feats_a = build_feature_vector(hr, 8000, 100, 3)
        feats_b = build_feature_vector(hr, 8000, 900, 3)
        self.assertNotEqual(feats_a[0][13], feats_b[0][13])


class RouteTests(unittest.TestCase):

    def setUp(self):
        app.config['TESTING'] = True
        self.c = app.test_client()

    def test_homepage_loads(self):
        r = self.c.get('/')
        self.assertEqual(r.status_code, 200)

    def test_model_info_responds(self):
        r = self.c.get('/model/info')
        self.assertEqual(r.status_code, 200)
        body = r.get_json()
        self.assertIn('loaded', body)

    def test_predict_without_model(self):
        import src.app_web as aw
        prev = aw._model_data
        aw._model_data = None
        payload = {f'hr{i}': 58.0 for i in range(1, 8)}
        payload.update({'steps': 8000, 'kcal': 400, 'dow': 0})
        r = self.c.post('/predict', json=payload)
        self.assertEqual(r.status_code, 503)
        aw._model_data = prev

    def test_predict_bad_hr(self):
        import src.app_web as aw
        if aw._model_data is None:
            self.skipTest("model neni nacten")
        payload = {f'hr{i}': 58.0 for i in range(1, 8)}
        payload['hr1'] = 999.0
        payload.update({'steps': 8000, 'kcal': 400, 'dow': 0})
        r = self.c.post('/predict', json=payload)
        self.assertEqual(r.status_code, 400)


if __name__ == '__main__':
    unittest.main(verbosity=2)
