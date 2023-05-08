from flask import Flask, render_template, request
import pickle
import numpy as np



model = pickle.load(open('model.pkl', 'rb'))
test_data_accuracy = pickle.load(open('accuracy.pkl', 'rb'))
random_forest=pickle.load(open("model1.pkl",'rb'))
test_data_accuracy1 = pickle.load(open('accuracy1.pkl', 'rb'))
app = Flask(__name__,static_folder='templates')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_placement():
    input_data1 = request.form.get("input_data1")
    i1 = np.array([float(val) for val in input_data1.split(',')])
    result = model.predict(i1.reshape(1, -1))
    a1=test_data_accuracy
    a2=test_data_accuracy1
    r1=random_forest.predict(i1.reshape(1, -1))






    if result[0] == 'R':
        result = 'The given object has been detected as a Rock'
    else:
        result = 'The given object has been detected as a Mine'



    if r1[0] == 'R':
        r1 = 'The given object has been detected as a Rock'
    else:
        r1 = 'The given object has been detected as a Mine'

    return render_template('index.html',a2=a2,a1=a1,result=result,r1=r1,input_data1=input_data1)

if __name__ == '__main__':
    app.run(host='127.16.0.1', port=5500)
