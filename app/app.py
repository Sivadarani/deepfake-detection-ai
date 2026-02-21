from flask import Flask, render_template, request, redirect, url_for, session
import os
import cv2
import sqlite3
import numpy as np
import tensorflow as tf
from functools import wraps
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "deepfake_secret_key_change_me")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Upload folder
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# SQLite database
DB_PATH = os.path.join(BASE_DIR, "database.db")

# Load TFLite model using TensorFlow (NOT tflite-runtime)
TFLITE_MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "deepfake_model.tflite")

interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

IMG_SIZE = 224
DEFAULT_FAKE_THRESHOLD = 0.40

# -------------------- AUTH DECORATOR --------------------
def login_required(route_func):
    @wraps(route_func)
    def wrapper(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login", next=request.path))
        return route_func(*args, **kwargs)
    return wrapper

# -------------------- DATABASE --------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            filename TEXT NOT NULL,
            result TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def create_user(username, password):
    password_hash = generate_password_hash(password)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    try:
        cur.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (username, password_hash))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def validate_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return False
    return check_password_hash(row[0], password)

def save_prediction(username, filename, result):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO predictions (username, filename, result) VALUES (?, ?, ?)",
        (username, filename, result),
    )
    conn.commit()
    conn.close()

def get_prediction_history(username):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT id, filename, result, timestamp
        FROM predictions
        WHERE username = ?
        ORDER BY id DESC
    """, (username,))
    rows = cur.fetchall()
    conn.close()
    return rows

# -------------------- PREDICTION --------------------
def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])

    # If model outputs [real, fake]
    if output.shape[-1] >= 2:
        fake_prob = float(output[0][1])
    else:
        fake_prob = float(output[0][0])

    return fake_prob

# -------------------- ROUTES --------------------
@app.route("/")
def landing():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("landing.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if validate_user(username, password):
            session["user"] = username
            next_url = request.args.get("next")
            return redirect(next_url or url_for("landing"))
        else:
            error = "Invalid username or password."

    return render_template("login.html", error=error)

@app.route("/signup", methods=["GET", "POST"])
def signup():
    error = None
    success = None

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if len(username) < 3:
            error = "Username must be at least 3 characters."
        elif len(password) < 4:
            error = "Password must be at least 4 characters."
        else:
            created = create_user(username, password)
            if created:
                success = "Account created. You can now login."
            else:
                error = "Username already exists."

    return render_template("signup.html", error=error, success=success)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    result = None
    confidence = None

    if request.method == "POST":
        file = request.files.get("image")
        if file and file.filename:
            filename = secure_filename(file.filename)
            path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(path)

            fake_prob = predict_image(path)
            result = "FAKE" if fake_prob >= DEFAULT_FAKE_THRESHOLD else "REAL"
            confidence = fake_prob if result == "FAKE" else (1.0 - fake_prob)

            save_prediction(session["user"], filename, result)

    return render_template("upload.html", result=result, confidence=confidence)

@app.route("/dashboard")
@login_required
def dashboard():
    history = get_prediction_history(session["user"])
    return render_template("dashboard.html", history=history)

# -------------------- MAIN --------------------
if __name__ == "__main__":
    init_db()
    app.run(debug=True)