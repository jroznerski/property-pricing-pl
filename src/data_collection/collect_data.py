"""
Data collection module for Polish real estate pricing system.
Future: scraping / API integration / external datasets.
"""

import kagglehub

# Download latest version
path = kagglehub.dataset_download("krzysztofjamroz/apartment-prices-in-poland")

print("Path to dataset files:", path)