from aim5005.features import MinMaxScaler, StandardScaler, LabelEncoder
import numpy as np
import unittest
from unittest.case import TestCase

### DO NOT MODIFY EXISTING TESTS

class TestFeatures(TestCase):
    def test_initialize_min_max_scaler(self):
        scaler = MinMaxScaler()
        assert isinstance(scaler, MinMaxScaler), "scaler is not a MinMaxScaler object"
        
        
    def test_min_max_fit(self):
        scaler = MinMaxScaler()
        data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
        scaler.fit(data)
        assert (scaler.maximum == np.array([1., 18.])).all(), "scaler fit does not return maximum values [1., 18.] "
        assert (scaler.minimum == np.array([-1., 2.])).all(), "scaler fit does not return maximum values [-1., 2.] " 
        
        
    def test_min_max_scaler(self):
        data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
        expected = np.array([[0., 0.], [0.25, 0.25], [0.5, 0.5], [1., 1.]])
        scaler = MinMaxScaler()
        scaler.fit(data)
        result = scaler.transform(data)
        assert (result == expected).all(), "Scaler transform does not return expected values. All Values should be between 0 and 1. Got: {}".format(result.reshape(1,-1))
        
    def test_min_max_scaler_single_value(self):
        data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
        expected = np.array([[1.5, 0.]])
        scaler = MinMaxScaler()
        scaler.fit(data)
        result = scaler.transform([[2., 2.]]) 
        assert (result == expected).all(), "Scaler transform does not return expected values. Expect [[1.5 0. ]]. Got: {}".format(result)
        
    def test_standard_scaler_init(self):
        scaler = StandardScaler()
        assert isinstance(scaler, StandardScaler), "scaler is not a StandardScaler object"
        
    def test_standard_scaler_get_mean(self):
        scaler = StandardScaler()
        data = [[0, 0], [0, 0], [1, 1], [1, 1]]
        expected = np.array([0.5, 0.5])
        scaler.fit(data)
        assert (scaler.mean == expected).all(), "scaler fit does not return expected mean {}. Got {}".format(expected, scaler.mean)
        
    def test_standard_scaler_transform(self):
        scaler = StandardScaler()
        data = [[0, 0], [0, 0], [1, 1], [1, 1]]
        expected = np.array([[-1., -1.], [-1., -1.], [1., 1.], [1., 1.]])
        scaler.fit(data)
        result = scaler.transform(data)
        assert (result == expected).all(), "Scaler transform does not return expected values. Expect {}. Got: {}".format(expected.reshape(1,-1), result.reshape(1,-1))
        
    def test_standard_scaler_single_value(self):
        data = [[0, 0], [0, 0], [1, 1], [1, 1]]
        expected = np.array([[3., 3.]])
        scaler = StandardScaler()
        scaler.fit(data)
        result = scaler.transform([[2., 2.]]) 
        assert (result == expected).all(), "Scaler transform does not return expected values. Expect {}. Got: {}".format(expected.reshape(1,-1), result.reshape(1,-1))

    # TODO: Add a test of your own below this line

    def test_standard_scaler_custom(self):
        scaler = StandardScaler()
        data = [[2, 4], [4, 8], [6, 12], [8, 16]]
        scaler.fit(data)
        result = scaler.transform(data)
        expected = np.array([[-1.3416, -1.3416], [-0.4472, -0.4472], [0.4472, 0.4472], [1.3416, 1.3416]])
        
        assert np.allclose(result, expected, atol=1e-3), "Custom test for StandardScaler failed."
        
    def test_label_encoder(self):
        encoder = LabelEncoder()
        labels = ["cat", "dog", "fish", "cat", "dog"]
        encoder.fit(labels)
        result = encoder.transform(labels)
        expected = [0, 1, 2, 0, 1]
        
        assert np.array_equal(result, expected), f"LabelEncoder failed. Expected {expected}, got {result}"

    def test_label_encoder_fit(self):
        le = LabelEncoder()
        y = np.array(["cat", "dog", "fish", "dog", "cat", "fish"])
        le.fit(y)

        assert set(le.classes_) == {"cat", "dog", "fish"}
        assert le.class_to_index == {"cat": 0, "dog": 1, "fish": 2}


    def test_label_encoder_transform(self):
        le = LabelEncoder()
        y = np.array(["cat", "dog", "fish", "dog", "cat", "fish"])
        le.fit(y)
        y_encoded = le.transform(y)

        assert y_encoded.tolist() == [0, 1, 2, 1, 0, 2]  # Matches assigned indices


    def test_label_encoder_fit_transform(self):
        le = LabelEncoder()
        y = np.array(["apple", "banana", "apple", "orange", "banana"])
        y_encoded = le.fit_transform(y)

        assert len(le.classes_) == 3
        assert y_encoded.tolist() == [0, 1, 0, 2, 1]  # Checks for consistency


    def test_label_encoder_unseen_value(self):
        le = LabelEncoder()
        y_train = np.array(["red", "blue", "green"])
        le.fit(y_train)

        y_test = np.array(["yellow", "red"])
        try:
            le.transform(y_test)
            assert False, "Expected error for unseen category"
        except KeyError:
            pass 


    def test_label_encoder_empty_input(self):
        le = LabelEncoder()
        y = np.array([])

        le.fit(y)
        assert le.classes_.size == 0 

        y_encoded = le.transform(y)
        assert y_encoded.size == 0 


    def test_label_encoder_numeric_labels(self):
        le = LabelEncoder()
        y = np.array([10, 20, 10, 30, 20, 30])
        y_encoded = le.fit_transform(y)

        assert len(le.classes_) == 3
        assert y_encoded.tolist() == [0, 1, 0, 2, 1, 2]  
        
    
if __name__ == '__main__':
    unittest.main()
