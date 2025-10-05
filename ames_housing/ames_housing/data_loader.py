from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).parent.parent 

def load_train_data():
    """
    Load the Ames Housing training dataset.
    
    Returns:
        pd.DataFrame: Training data with 'Id' as index
    """
    path = BASE_DIR / "data" / "train.csv"
    train = pd.read_csv(path, index_col="Id")
    return train

def load_test_data():
    """
    Load the Ames Housing test dataset.
    
    Returns:
        pd.DataFrame: test data with 'Id' as index
    """
    path = BASE_DIR / "data" / "test.csv"
    test = pd.read_csv(path, index_col="Id")
    return test
