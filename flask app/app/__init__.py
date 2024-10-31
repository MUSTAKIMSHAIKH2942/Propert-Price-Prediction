from flask import Flask

def create_app():
    app = Flask(__name__)
    
    # Load configurations
    app.config.from_object('app.config.Config')
    
    # Register blueprints or routes
    from . import routes  # Assuming you have routes in routes.py
    app.register_blueprint(routes.bp)

    return app
