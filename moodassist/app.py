import os
import json
from flask import Flask, request, render_template, send_file, make_response, request, session, abort, redirect, flash
import pyrebase
from flask import jsonify
import time
#from flask_bootstrap import Bootstrap

config = {
    "apiKey": "AIzaSyBxdTCKUa1pa1z-fuBwJoc8QEe3BZyeBxI",
    "authDomain": "friendlychat-3a53a.firebaseapp.com",
    "databaseURL": "https://friendlychat-3a53a.firebaseio.com",
    "projectId": "friendlychat-3a53a",
    "storageBucket": "friendlychat-3a53a.appspot.com",
    "messagingSenderId": "208699988378",
    "appId": "1:208699988378:web:0e5c36b40424a903e08223"
}

firebase = pyrebase.initialize_app(config)
app = Flask(__name__)
auth = firebase.auth()
#######################################

@app.route('/', methods=["POST", "GET"])
def login():
    if not session.get('logged_in'):
        return render_template('index.html')
    
    if request.method == "POST":
        email = request.form["login_email"]
        password = request.form["login_password"]
        try:
            user = auth.sign_in_with_email_and_password(email, password)
            session['logged_in'] = True
            return redirect(url_for('admin'))
        except:
            flash("Incorrect Password!")

@app.route('/admin')
def admin():
    return render_template("admin.html")

if __name__ == '__main__':
    app.jinja_env.auto_reload = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(debug=True, host='0.0.0.0')