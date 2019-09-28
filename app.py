from flask import Flask, render_template, redirect, url_for,request
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField
from wtforms.validators import InputRequired, Email, Length
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

cred=credentials.Certificate("C:\\Users\\Administrator\\Desktop\\Singapore-India Hackathon\\flask-bootstrap trial 1\\friendlychat-3a53a-firebase-adminsdk.json")
firebase_admin.initialize_app(cred)
db=firestore.client()

#def student_table():
#    docs=db.collection(u'students').stream()
#    for doc in docs:
#        for 
database_dict = {1: [1033,"simrita",98,"A","B",21,"y"], 2:[1097,"john",89,"B","C",23,"n"]}


app = Flask(__name__)
app.config['SECRET_KEY'] = 'thisissupposedtobesecret'
Bootstrap(app)

class SignUpForm(FlaskForm):
    username = StringField('UserName', validators = [InputRequired(), Length(min =4, max=15)])
    password = PasswordField('Password', validators = [InputRequired(), Length(min=8, max = 20)])
    remember = BooleanField('RememberMe!')

@app.route('/index.html', methods=["POST","GET"])
def index():
    if request.method == "GET":
        return render_template('index.html', database_dict = database_dict)
    else:
        grade_file = request.files['grades']
        attendance_file = request.files['attendance']
    return render_template('index.html', database_dict = database_dict,msg="success")

@app.route('/login.html' ,methods=["POST","GET"])
def login():
    if request.method == 'GET':
        return render_template('login.html')
    else:
        username = request.form['username']
        password = request.form['password']
        return redirect(url_for("index"))
    return render_template('login.html')

@app.route('/register.html' ,methods=["POST","GET"])
def register():
    if request.method == 'GET':
        return render_template('register.html')
    else:
        name = request.form['name']
        password = request.form['password']
        return redirect(url_for("login"))
    return render_template('index.html')

@app.route('/signup', methods = ['GET','POST'])
def signup():
    form = SignUpForm()

    if form.validate_on_submit():
        return redirect(url_for('index'))
        
    return render_template('signup.html', form=form)


if __name__ == '__main__':
    app.run(debug=True)
