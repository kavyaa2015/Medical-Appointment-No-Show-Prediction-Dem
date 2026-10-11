"""Smoke tests for Streamlit app and dependencies."""

import os
import pytest


def test_app_file_exists():
    assert os.path.exists('app.py')
    with open('app.py', 'r', encoding='utf-8') as f:
        content = f.read()
    assert 'st.set_page_config' in content
    assert 'No-Show Predictor' in content
    assert 'Demand Forecaster' in content


def test_required_directories():
    assert os.path.exists('src')
    assert os.path.exists('models')
    assert os.path.exists('outputs')
    assert os.path.exists('notebooks')
    assert os.path.exists('requirements.txt')
