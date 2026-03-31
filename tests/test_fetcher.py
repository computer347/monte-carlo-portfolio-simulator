import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from data.fetcher import fetch_data

@pytest.fixture
def mock_yf_normal():
    # 150 rows of data (more than 100)
    dates = pd.date_range('2020-01-01', periods=150)
    data = {
        'Open': np.random.rand(150),
        'High': np.random.rand(150),
        'Low': np.random.rand(150),
        'Close': np.random.rand(150),
        'Adj Close': np.random.rand(150),
        'Volume': np.random.randint(100, 1000, 150)
    }
    return pd.DataFrame(data, index=dates)

@pytest.fixture
def mock_yf_small():
    # 50 rows of data (less than 100)
    dates = pd.date_range('2020-01-01', periods=50)
    data = {'Close': np.random.rand(50)}
    return pd.DataFrame(data, index=dates)

@pytest.fixture
def mock_yf_with_nans():
    dates = pd.date_range('2020-01-01', periods=150)
    df = pd.DataFrame({'Close': np.random.rand(150)}, index=dates)
    # Inject NaNs
    df.iloc[10] = np.nan
    df.iloc[20] = np.nan
    # First row NaN to test dropping after ffill
    df.iloc[0] = np.nan
    return df

@patch('data.fetcher.yf.download')
def test_fetch_data_normal(mock_download, mock_yf_normal):
    mock_download.return_value = mock_yf_normal
    df = fetch_data('SPY', '2020-01-01', '2020-05-30')
    assert len(df) == 150
    mock_download.assert_called_once()
    assert mock_download.call_args[0][0] == 'SPY'

@patch('data.fetcher.yf.download')
def test_fetch_data_fallback_kc(mock_download, mock_yf_small, mock_yf_normal):
    # First call returns small, second returns normal
    mock_download.side_effect = [mock_yf_small, mock_yf_normal]
    
    df = fetch_data('KC=F', '2020-01-01', '2020-05-30')
    assert len(df) == 150
    assert mock_download.call_count == 2
    assert mock_download.call_args_list[0][0][0] == 'KC=F'
    assert mock_download.call_args_list[1][0][0] == 'KT=F'

@patch('data.fetcher.yf.download')
def test_fetch_data_forward_fill_drop_nan(mock_download, mock_yf_with_nans):
    mock_download.return_value = mock_yf_with_nans
    df = fetch_data('SPY', '2020-01-01', '2020-05-30')
    assert df.isna().sum().sum() == 0
    # First row gets dropped because ffill doesn't fill the first row if it's NaN
    assert len(df) == 149
