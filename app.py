from flask import Flask, render_template
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField
from wtforms.validators import InputRequired, Email, Length

app = Flask(__name__)
app.config['SECRET_KEY'] = 'thisissupposedtobesecret'
Bootstrap(app)

class SignUpForm(FlaskForm):
    username = StringField('UserName', validators = [InputRequired(), Length(min =4, max=15)])
    password = PasswordField('Password', validators = [InputRequired(), Length(min=8, max = 20)])
    remember = BooleanField('RememberMe!')
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signup')
def signup():
    form = SignUpForm()
    return render_template('signup.html', form=form)

if __name__ == '__main__':
    app.run(debug=True)
