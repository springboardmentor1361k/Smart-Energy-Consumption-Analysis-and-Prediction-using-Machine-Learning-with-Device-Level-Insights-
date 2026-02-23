"""
COMPLETE PROJECT SETUP SCRIPT
Run this ONCE to create the entire project structure
"""

import os
import json

print("=" * 70)
print("🏗️  SMART ENERGY PROJECT SETUP")
print("=" * 70)

# ============================================================================
# CREATE DIRECTORY STRUCTURE
# ============================================================================
print("\n📁 Creating directory structure...")

directories = [
    'data/raw',
    'data/processed',
    'data/processed/splits',
    'data/models',
    'notebooks',
    'app/templates',
    'app/static/css',
    'app/static/js',
    'app/static/images',
    'reports/figures',
    'config',
]

for dir_path in directories:
    os.makedirs(dir_path, exist_ok=True)
    print(f"  ✅ Created: {dir_path}")

# ============================================================================
# CREATE __init__.py FILES
# ============================================================================
init_files = ['app/__init__.py', 'config/__init__.py']
for init_file in init_files:
    with open(init_file, 'w') as f:
        f.write('"""Package initialization."""\n')
    print(f"  ✅ Created: {init_file}")

# ============================================================================
# CREATE .env FILE
# ============================================================================
env_content = """SECRET_KEY=smart-energy-secret-key-2024
DEBUG=True
FLASK_ENV=development
"""

with open('.env', 'w') as f:
    f.write(env_content)
print("  ✅ Created: .env")

# ============================================================================
# CREATE requirements.txt
# ============================================================================
requirements = """# Core
flask==3.0.0
flask-cors==4.0.0
python-dotenv==1.0.0
werkzeug==3.0.1

# Data Processing
pandas==2.0.3
numpy==1.24.3

# ML - Using sklearn only (no TensorFlow issues)
scikit-learn==1.3.0
xgboost==2.0.3
joblib==1.3.2

# Visualization
matplotlib==3.7.2
seaborn==0.12.2

# Utilities
pytz==2023.3
requests==2.31.0
"""

with open('requirements.txt', 'w') as f:
    f.write(requirements)
print("  ✅ Created: requirements.txt")

# ============================================================================
# CREATE DEVICE MAPPING
# ============================================================================
device_mapping = {
    "device_names": {
        "Sub_metering_1": "Kitchen",
        "Sub_metering_2": "Laundry", 
        "Sub_metering_3": "Climate_Control",
        "Sub_metering_4": "Other_Appliances"
    },
    "device_rooms": {
        "Aggregate": "Whole_House",
        "Kitchen": "Kitchen",
        "Laundry": "Utility",
        "Climate_Control": "HVAC",
        "Other_Appliances": "Mixed"
    },
    "device_descriptions": {
        "Kitchen": "Dishwasher, oven, microwave, hot-plates",
        "Laundry": "Washing machine, tumble-dryer, refrigerator, light",
        "Climate_Control": "Electric water-heater, air-conditioner",
        "Other_Appliances": "All other unmeasured appliances"
    }
}

os.makedirs('data/processed', exist_ok=True)
with open('data/processed/device_mapping.json', 'w') as f:
    json.dump(device_mapping, f, indent=2)
print("  ✅ Created: data/processed/device_mapping.json")

# ============================================================================
# CREATE .gitignore
# ============================================================================
gitignore = """# Virtual environment
venv/
env/
.venv/

# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python

# Data files
data/raw/*.txt
data/raw/*.zip
*.csv

# Models
*.pkl
*.h5
*.keras

# IDE
.vscode/
.idea/

# Environment
.env

# OS
.DS_Store
Thumbs.db
"""

with open('.gitignore', 'w') as f:
    f.write(gitignore)
print("  ✅ Created: .gitignore")

print("\n" + "=" * 70)
print("✅ PROJECT STRUCTURE CREATED SUCCESSFULLY!")
print("=" * 70)
print("""
📁 Project Structure:
smart-energy-new/
├── data/
│   ├── raw/           (for downloaded dataset)
│   ├── processed/     (for cleaned data)
│   └── models/        (for trained models)
├── notebooks/         (for milestone scripts)
├── app/               (Flask application)
│   ├── templates/
│   └── static/
├── reports/figures/   (for visualizations)
├── requirements.txt
└── .env

Next steps:
1. Create virtual environment: python -m venv venv
2. Activate: venv\\Scripts\\activate
3. Install: pip install -r requirements.txt
4. Run download script (create it next)
""")