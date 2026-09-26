import streamlit as st
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title='Netflix Churn Decision Tool', page_icon='📺', layout='wide')

@st.cache_data
 def load_data(path='netflix_customer_churn.csv'):
    return pd.read_csv(path)

@st.cache_resource
def train_models(df):
    X=df.drop(columns=['customer_id','churned'])
    y=df['churned']
    cats=X.select_dtypes(include='object').columns.tolist()
    nums=X.select_dtypes(exclude='object').columns.tolist()
    pre=ColumnTransformer([('num',StandardScaler(),nums),('cat',OneHotEncoder(handle_unknown='ignore'),cats)])
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=42,stratify=y)
    log=Pipeline([('pre',pre),('model',LogisticRegression(max_iter=2000,class_weight='balanced'))])
    lin=Pipeline([('pre',pre),('model',LinearRegression())])
    log.fit(Xtr,ytr); lin.fit(Xtr,ytr)
    p=log.predict_proba(Xte)[:,1]; pred=(p>=.5).astype(int)
    lp=np.clip(lin.predict(Xte),0,1)
    metrics={'Accuracy':accuracy_score(yte,pred),'AUC':roc_auc_score(yte,p),'Linear MAE':mean_absolute_error(yte,lp),'Linear RMSE':mean_squared_error(yte,lp)**.5,'Linear R²':r2_score(yte,lp)}
    return log,lin,metrics,cats,nums

st.title('📺 Netflix Customer Churn Decision Tool')
st.caption('A managerial decision-support prototype using logistic and linear regression. Predictions are aids to judgement, not automatic decisions.')
try:
    df=load_data()
except Exception as e:
    st.error('Upload or place netflix_customer_churn.csv in the same folder as this app.')
    st.stop()
log,lin,metrics,cats,nums=train_models(df)

with st.sidebar:
    st.header('Customer profile')
    age=st.slider('Age',13,80,35)
    gender=st.selectbox('Gender',sorted(df.gender.unique()))
    subscription_type=st.selectbox('Subscription type',sorted(df.subscription_type.unique()))
    watch_hours=st.number_input('Watch hours',0.0,100.0,20.0,.1)
    last_login_days=st.number_input('Days since last login',0,365,7)
    region=st.selectbox('Region',sorted(df.region.unique()))
    device=st.selectbox('Device',sorted(df.device.unique()))
    monthly_fee=st.number_input('Monthly fee',0.0,100.0,13.99,.01)
    payment_method=st.selectbox('Payment method',sorted(df.payment_method.unique()))
    number_of_profiles=st.number_input('Number of profiles',1,10,2)
    avg_watch_time_per_day=st.number_input('Average watch time per day',0.0,24.0,1.0,.01)
    favorite_genre=st.selectbox('Favourite genre',sorted(df.favorite_genre.unique()))
    run=st.button('Predict churn risk',type='primary')

profile=pd.DataFrame([{'age':age,'gender':gender,'subscription_type':subscription_type,'watch_hours':watch_hours,'last_login_days':last_login_days,'region':region,'device':device,'monthly_fee':monthly_fee,'payment_method':payment_method,'number_of_profiles':number_of_profiles,'avg_watch_time_per_day':avg_watch_time_per_day,'favorite_genre':favorite_genre}])

if run:
    prob=float(log.predict_proba(profile)[0,1]); lp=float(np.clip(lin.predict(profile)[0],0,1))
    risk='High' if prob>=.70 else 'Medium' if prob>=.40 else 'Low'
    c1,c2,c3=st.columns(3)
    c1.metric('Logistic churn probability',f'{prob:.1%}')
    c2.metric('Linear-regression score',f'{lp:.1%}')
    c3.metric('Risk band',risk)
    st.subheader('Managerial interpretation')
    if risk=='High': st.warning('Prioritise retention outreach: investigate inactivity, content fit, payment friction and a targeted re-engagement offer.')
    elif risk=='Medium': st.info('Monitor engagement and consider personalised content, reminders or a low-cost retention intervention.')
    else: st.success('No immediate high-risk signal. Continue normal engagement monitoring.')
    st.write('The logistic model is the primary classifier. The linear model is shown as a probability-like benchmark and should not be interpreted as a calibrated probability.')

st.divider()
st.subheader('Model performance')
st.dataframe(pd.DataFrame({'Metric':list(metrics.keys()),'Value':[round(v,4) for v in metrics.values()]}),hide_index=True,use_container_width=True)
st.caption('Evaluation uses a stratified 75/25 train-test split. Metrics describe this dataset and should be revalidated before operational use.')

st.subheader('Managerial user study')
st.write('Use the tool, then collect feedback from managers, team leaders, supervisors, entrepreneurs and other experienced decision-makers.')
with st.form('feedback'):
    role=st.selectbox('Your role', ['Manager','Working professional','Team leader','Supervisor','Entrepreneur','Other experienced decision-maker'])
    usefulness=st.slider('How useful is the tool?',1,5,3)
    trust=st.slider('How much do you trust the prediction?',1,5,3)
    would_use=st.selectbox('Would you use it in your work?', ['Yes','Maybe','No'])
    explainability=st.slider('How important is explainability?',1,5,5)
    comments=st.text_area('What do you like, dislike, or want improved?')
    conflict=st.text_area('If the prediction conflicts with your experience, what would you do?')
    submitted=st.form_submit_button('Submit feedback')
if submitted:
    row=pd.DataFrame([{'role':role,'usefulness':usefulness,'trust':trust,'would_use':would_use,'explainability':explainability,'comments':comments,'conflict_response':conflict}])
    try:
        old=pd.read_csv('manager_feedback.csv'); row=pd.concat([old,row],ignore_index=True)
    except FileNotFoundError: pass
    row.to_csv('manager_feedback.csv',index=False)
    st.success('Feedback recorded locally in manager_feedback.csv.')

st.subheader('Study questions')
st.markdown('- How useful is the tool for managers?\n- Would you use it in your work, and why?\n- What do you like or dislike?\n- Do you trust the prediction? Why?\n- What would increase trust?\n- What would you do if experience and AI disagree?\n- Does explainability affect adoption?\n- Does fear of a wrong decision reduce use?\n- What organisational or personal factors discourage adoption?')
