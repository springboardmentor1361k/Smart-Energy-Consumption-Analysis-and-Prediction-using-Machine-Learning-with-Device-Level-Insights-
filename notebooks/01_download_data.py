"""
STEP 1: Download UCI Household Power Consumption Dataset
"""

import os
import requests
import zipfile

print("=" * 70)
print("📥 DOWNLOADING UCI DATASET")
print("=" * 70)

# Create directory
os.makedirs('data/raw', exist_ok=True)

# Download URL
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip"
zip_path = "data/raw/household_power_consumption.zip"
txt_path = "data/raw/household_power_consumption.txt"

# Check if already downloaded
if os.path.exists(txt_path):
    print(f"✅ Dataset already exists at {txt_path}")
    print(f"   Size: {os.path.getsize(txt_path) / 1024 / 1024:.1f} MB")
else:
    print(f"📥 Downloading from: {url}")
    print("   This may take a few minutes...")
    
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = (downloaded / total_size) * 100
                    print(f"\r   Progress: {pct:.1f}%", end='', flush=True)
        
        print(f"\n✅ Download complete!")
        
        # Extract
        print("📦 Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall('data/raw/')
        print("✅ Extraction complete!")
        
        # Clean up zip
        os.remove(zip_path)
        print("🗑️ Removed zip file")
        
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print("\n📋 MANUAL DOWNLOAD INSTRUCTIONS:")
        print("1. Go to: https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption")
        print("2. Click 'Download' button")
        print("3. Extract the zip file")
        print("4. Copy 'household_power_consumption.txt' to 'data/raw/' folder")
        exit(1)

# Verify
if os.path.exists(txt_path):
    file_size = os.path.getsize(txt_path) / 1024 / 1024
    print(f"\n✅ SUCCESS! Dataset ready:")
    print(f"   Path: {txt_path}")
    print(f"   Size: {file_size:.1f} MB")
    
    # Preview
    print("\n📊 Dataset Preview:")
    with open(txt_path, 'r') as f:
        for i, line in enumerate(f):
            if i < 5:
                print(f"   {line.strip()}")
            else:
                break
else:
    print(f"\n❌ ERROR: Dataset not found at {txt_path}")