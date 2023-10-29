from flask import Flask, request, url_for, redirect, render_template
import pickle
import numpy as np


app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index1.html')

            # 0    no_of_dependents  3415 non-null   int64
            # 1    education         3415 non-null   int64
            # 2    self_employed     3415 non-null   int64
            # 3    income_annum      3415 non-null   int64
            # 4    loan_amount       3415 non-null   int64
            # 5    loan_term         3415 non-null   int64
            # 6    cibil_score       3415 non-null   int64
            # 7   Movable_assets     3415 non-null   int64
            # 8   Immovable_assets   3415 non-null   int64 

@app.route('/predict',methods=['POST'])
def predict():
        if request.method == 'POST':
            Dependencies = int(request.form['dependencies'])
            Education =int(request.form['education'])
            Selfemp =int(request.form['self_employed'])
            incomeanumn=int(request.form['incomeanumn'])
            loananumn=int(request.form['loanammount'])
            loanterm=int(request.form['loanterm'])
            cibilscore=int(request.form['cibilscore'])
            masset=int(request.form['masset'])
            imasset=int(request.form['imasset'])
            #final_features = [Dependencies,Education,Selfemp,incomeanumn,loananumn,cibilscore,masset,imasset]
            final_features = np.array([Dependencies,Education,Selfemp,incomeanumn,loananumn,loanterm,cibilscore,masset,imasset])
            print(final_features)

            final_features = final_features.reshape(1,-1)
            prediction = model.predict(final_features)
            print(prediction)
            # proba = max(model.predict_proba(prediction))
            # print(proba[0])
            
            # top_proba=proba[:,1]
            # print(top_proba)
            #output = round(prediction[0], 2)
            output = prediction.astype('int')
            print(output)
            if (output > 0.7):
                return render_template('index1.html', prediction_text='you have high chance of getting loan approved. \n Probability of loan approval is {}'.format(output))
            else:
                return render_template('index1.html',prediction_text='you have low chance of getting loan approved.  \n Probability of loan approval is {}'.format(output))
            # if (prediction==1):
            #     return render_template('index1.html', prediction_text='you have high chance of getting loan approved.\n Probability of loan approval is {}'.format(proba))
            # else:
            #      return render_template('index1.html',prediction_text='you have low chance of getting loan approved.\n Probability of loan approval is {}'.format(proba))

    # int_features=[int(x) for x in request.form.values()]
    # final=[np.array(int_features)]
    # print(int_features)
    # print(final)
    # prediction=model.predict_proba(final)
    # output='{0:.{1}f}'.format(prediction[0][1], 2)
    # if output>str(0.7):
    #     return render_template('index1.html',pred='Your Forest is in Danger.\nProbability of fire occuring is {}'.format(output),bhai="kuch karna hain iska ab?")
    # else:
    #     return render_template('index1.html',pred='Your Forest is safe.\n Probability of fire occuring is {}'.format(output),bhai="Your Forest is Safe for now")


if __name__ == "__main__":
    app.run(debug=True)