from flask import Flask, render_template, redirect, url_for,request
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField
from wtforms.validators import InputRequired, Email, Length
import firestore_api as fb

database_dict = fb.display_table()

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
        filename= secure_filename(grade_file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
        fb.update_grade(filename)
        attendance_file = request.files['attendance']
        filename1= secure_filename(attendance_file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename1))
        fb.update_attendance(filename1)
        
    return render_template('index.html', database_dict = database_dict,msg="success")

@app.route('/login.html' ,methods=["POST","GET"])
def login():
    if request.method == 'GET':
        return render_template('login.html')
    else:
        username = request.form['username']
        password = request.form['password']
        val = fb.check_user(username,password)
        if val==0:
            return redirect(url_for("index"))
        else:
            return redirect(url_for("login"))
    return render_template('login.html')

@app.route('/register.html' ,methods=["POST","GET"])
def register():
    if request.method == 'GET':
        return render_template('register.html')
    else:
        name = request.form['name']
        username = request.form['username']
        password = request.form['password']
        fb.add_user(name,username,password)
        return redirect(url_for("index"))
    return render_template('index.html')

@app.route('/signup', methods = ['GET','POST'])
def signup():
    form = SignUpForm()

    if form.validate_on_submit():
        return redirect(url_for('index'))
        
    return render_template('signup.html', form=form)


if __name__ == '__main__':
    app.run(debug=True)
