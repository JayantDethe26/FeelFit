from flask import Flask, render_template, request, redirect, url_for, jsonify
from dotenv import load_dotenv
import os
import random
import json
from datetime import date, timedelta

# Initialize Flask app
app = Flask(__name__)

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN", "")
GROQ_KEY = os.getenv("GROQ_API_KEY", "")
PROGRESS_FILE = "progress_log.csv"
STREAK_FILE = "streak.json"

# Helper function to track progress and streaks
def update_and_show_streak():
    today = date.today().isoformat()
    if os.path.exists(STREAK_FILE):
        data = json.load(open(STREAK_FILE))
    else:
        data = {"last_date": "", "streak": 0}
    last = date.fromisoformat(data["last_date"]) if data["last_date"] else None
    if last == date.today() - timedelta(days=1):
        data["streak"] += 1
    elif last != date.today():
        data["streak"] = 1
    data["last_date"] = today
    json.dump(data, open(STREAK_FILE, "w"))
    return data["streak"]

# Route for homepage
@app.route('/')
def home():
    streak = update_and_show_streak()
    return render_template("index.html", streak=streak)

# Route for chatbot page
@app.route('/chat', methods=["GET", "POST"])
def chat():
    if request.method == "POST":
        user_input = request.form['user_input']
        response = chat_fitness(user_input)  # Call your chatbot function here
        return jsonify({'response': response})
    
    return render_template("chatbot.html")

# Chat function (similar to your chatbot backend logic)
def chat_fitness(user_input: str) -> str:
    # Example response, replace with actual chatbot logic
    # You can call the chatbot backend function here
    return f"Chatbot says: {user_input}"

# Start the Flask app
if __name__ == "__main__":
    app.run(debug=True)
