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
        "username":user,
        "password":psw
    }
    doc_ref=db.collection('administrator').document(str(next_id)).set(json)
    return
		
def check_user(name,psw):
    doc_ref=db.collection('administrator').where(u'username',u'==',name).stream()
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

def display_table():
    docs_ref=db.collection('students').stream()
    dictionary={}
    no_of_dep=0
    for doc in docs_ref:
        dic=doc.to_dict()
        id=doc.id
        print(id)
        tweet=db.collection('tweets').where('student_id','==',str(id)).stream()
        for t in tweet:
            no_of_dep=len(t.to_dict()['tweets'])
            print("Count = ",no_of_dep)
         
        dictionary[id]=dic['name'],dic['attendance'],dic['grades'][-1],dic['predicted_grade'],no_of_dep
    return dictionary
		
