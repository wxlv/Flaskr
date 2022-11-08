import os

from flask import Flask, render_template, g, redirect, url_for
from . import db, auth, blog

def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(SECRET_KEY='dev', DATABASE=os.path.join(
        app.instance_path, 'flaskr.sqlite'))

    if test_config is None:
        app.config.from_pyfile('config.py', silent=True)
    else:
        app.config.from_mapping(test_config)

    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # # application route conf
    @app.route('/hello')
    def hello():
        return 'Hello,World!'
    # @app.route('/')
    # def index():
    #     if g.user is None:
    #         return redirect(url_for('auth.login'))
    #     else:
    #         return render_template('index.html')
    
    # initalized application database
    db.init_app(app)

    # register app auth
    app.register_blueprint(auth.bp)
    app.register_blueprint(blog.bp)

    # register app default rule, the rule resitered in blog.py file
    app.add_url_rule('/', endpoint='index')

    # return flask app instance
    return app
