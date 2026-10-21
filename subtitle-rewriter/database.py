from flask import Flask
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///jobs.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["MAX_CONTENT_LENGTH"] = 3 * 1024 * 1024 * 1024  # 3GB max upload

from models import db

db.init_app(app)

with app.app_context():
    db.create_all()
