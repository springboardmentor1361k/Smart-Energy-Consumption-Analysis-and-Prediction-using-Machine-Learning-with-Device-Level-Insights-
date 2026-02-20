from flask import Flask
from .routes import main_bp

def create_app():
    app = Flask(__name__, 
                template_folder='../frontend/templates',
                static_folder='../frontend/static',
                static_url_path='/static')
    
    app.secret_key = 'smart-energy-secret-2024'
    app.register_blueprint(main_bp)
    
    return app
