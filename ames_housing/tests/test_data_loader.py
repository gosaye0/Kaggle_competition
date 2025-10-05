import pytest
import pandas as pd
from ames_housing.data_loader import load_train_data, load_test_data

def test_load_train_data():
    """
    Tests for data loading functionality.

    Tests verify that training and test data are loaded correctly
    with proper structure and expected columns.
    """
    df = load_train_data()  
    assert isinstance(df, pd.DataFrame)
    assert "Id" not in df.columns
    assert "SalePrice" in df.columns

def test_load_test_data():
    df = load_test_data() 
    assert isinstance(df, pd.DataFrame)
    assert "Id" not in df.columns
    assert "SalePrice" not in df.columns