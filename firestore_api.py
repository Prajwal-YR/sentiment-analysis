import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import google.cloud
import pandas as pd

cred = credentials.Certificate('C:\\Learn\\learn4\\friendlychat-3a53a-firebase-adminsdk.json')
firebase_admin.initialize_app(cred)

db = firestore.client()

def add_user(name,user,psw):
    doc_ref=db.collection('administrator').stream()
    next_id=max([int(x.id) for x in doc_ref])+1
    print(next_id)
    json={
        "name":name,
        "email":user,
        "password":psw
    }
    doc_ref=db.collection('administrator').document(str(next_id)).set(json)
    return
		
def check_user(email,psw):
    doc_ref=db.collection('administrator').where(u'username',u'==',email).stream()
    #if len(list(doc_ref)) == 0:
        #print("Empty")
        #user DNE
        #return 1
    #doc=None
    for doc in doc_ref:
        id=doc.id
        print(id)
    doc_ref=db.collection('administrator').document(str(doc.id))
    doc=doc_ref.get()
    if doc.to_dict()['password'] == psw:
            return 0
    #password incorrect    
    return 2

def update_grade(file):
    df =  pd.read_csv(file)
    for data in df.values:
        sid=data[0]
        grade=str(data[1])
        doc_ref = db.collection(u'students').document(str(sid))
        doc_ref.update({u'grades': firestore.ArrayUnion([grade])})
	return
def update_attendance(file):
    df=pd.read_csv(file)
    for data in df.values:
        sid=data[0]
        attendance=str(data[1])
        doc_ref = db.collection(u'students').document(str(sid))
        doc_ref.update({u'attendance': attendance})
    return

def display_table(email):
    docs_ref = db.collection('administrator').where('email','==','email').stream()
    dictionary={}
    no_of_dep=0
    for doc in docs_ref:
        dic=doc.to_dict()
        id=doc.id
        sid=dic['students']
        
        tweet=db.collection('tweets').where('student_id','==',str(id)).stream()
        for t in tweet:
            no_of_dep=len(t.to_dict()['tweets'])
         
        dictionary[id]=id,dic['name'],dic['attendance'],dic['grades'][-1],dic['predicted_grade'],no_of_dep
    return dictionary

def student_details(sid):
    docs_ref=db.collection('tweets').where('student_id','==',str(sid)).stream()
    stud=db.collection('students').document(str(sid)).get()
    dictionary={}
    sname=None
    tlist=[]
    times=[]
    sname=stud.to_dict()['name']
    for doc in docs_ref:
        twitter_username=doc.id
        dic=doc.to_dict()
        no_of_dep=len(dic['tweets'])
        for i in dic['tweets']:
            tlist.append(i['text'])
            times.append(i['timestamp'])
    return sname,tlist,times

def add_student(form):
    did=form.pop('id',None)
    doc_ref=db.collection('students').document(str(did)).set(form)
    tweet_ref=db.collection('tweets').document(str(form['twitter'])).set({'student_id':str(did)})

