from . import create_app
from flask_login import login_manager
from .routes import users_db

app = create_app()
login_manager.init_app(app)
login_manager.login_view = 'main.index'

if __name__ == '__main__':
    app.run(debug=True, port=5000)
