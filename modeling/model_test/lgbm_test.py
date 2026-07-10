import unittest
from dataclasses import dataclass, field
import numpy as np
from base.lgbm_q import LGBMQuantile  # 👈 Replace 'my_module' with your actual file name

# 1. Create a mock Configuration class to feed into your __init__
@dataclass
class MockConfig:
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    seed: int = 42
    lgbm_params: dict = field(default_factory=lambda: {
        "n_estimators": 10,  # Kept small (10 trees) so the tests run lightning-fast
        "learning_rate": 0.1,
        "num_leaves": 15,
        "verbose": -1
    })

class TestLGBMQuantile(unittest.TestCase):
    
    def setUp(self):
        """Set up synthetic data and configuration before each test."""
        self.cfg = MockConfig()
        
        # Generate dummy data: 200 samples, 3 features (e.g., Radiation, Temp, Humidity)
        np.random.seed(self.cfg.seed)
        self.X_train = np.random.rand(200, 3).astype(np.float32)
        # Linear relationship with noise to simulate solar output
        self.y_train = (self.X_train[:, 0] * 50 + self.X_train[:, 1] * 20 + np.random.randn(200) * 5).astype(np.float32)
        
        # Validation set (eval_set)
        self.X_val = np.random.rand(50, 3).astype(np.float32)
        self.y_val = (self.X_val[:, 0] * 50 + self.X_val[:, 1] * 20 + np.random.randn(50) * 5).astype(np.float32)
        self.eval_set = (self.X_val, self.y_val)
        
        # Test/Future set for prediction
        self.X_test = np.random.rand(10, 3).astype(np.float32)

    def test_initialization(self):
        """Test if attributes are correctly assigned from config during __init__."""
        model = LGBMQuantile(self.cfg)
        
        self.assertEqual(model.quantiles, self.cfg.quantiles)
        self.assertEqual(model.seed, self.cfg.seed)
        self.assertEqual(model.params, self.cfg.lgbm_params)
        self.assertEqual(model.models_, {})  # Closet must start empty

    def test_fit_without_eval_set(self):
        """Test training when no validation set is supplied."""
        model = LGBMQuantile(self.cfg)
        returned_model = model.fit(self.X_train, self.y_train, eval_set=None)
        
        # Check method chaining: fit should return self
        self.assertIsInstance(returned_model, LGBMQuantile)
        # Verify a model was trained and stored for every registered quantile
        self.assertEqual(len(model.models_), len(self.cfg.quantiles))
        for q in self.cfg.quantiles:
            self.assertIn(q, model.models_)

    def test_fit_with_eval_set(self):
        """Test training with an eval_set (activates early stopping callbacks)."""
        model = LGBMQuantile(self.cfg)
        model.fit(self.X_train, self.y_train, eval_set=self.eval_set)
        
        self.assertEqual(len(model.models_), len(self.cfg.quantiles))

    def test_predict_output_structure_and_type(self):
        """Test if predict returns a dictionary of float32 arrays with correct shapes."""
        model = LGBMQuantile(self.cfg).fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)
        
        self.assertIsInstance(predictions, dict)
        self.assertEqual(len(predictions), len(self.cfg.quantiles))
        
        for q in self.cfg.quantiles:
            self.assertIn(q, predictions)
            # Ensure the output data type is forced to float32 as specified in your code
            self.assertEqual(predictions[q].dtype, np.float32)
            # Row count must match test input row count
            self.assertEqual(predictions[q].shape, (self.X_test.shape[0],))

    def test_feature_importance(self):
        """Test if feature_importance outputs a dictionary containing scores for each feature."""
        model = LGBMQuantile(self.cfg).fit(self.X_train, self.y_train)
        importance = model.feature_importance()
        
        self.assertIsInstance(importance, dict)
        self.assertEqual(len(importance), len(self.cfg.quantiles))
        
        num_features = self.X_train.shape[1]
        for q in self.cfg.quantiles:
            self.assertIn(q, importance)
            # Importance array length must match number of input columns (3)
            self.assertEqual(len(importance[q]), num_features)

if __name__ == "__main__":
    unittest.main()