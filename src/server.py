# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 15:47:10 2021

@author: Administrator
"""

from config import *
from db_utils import Issue,load_dataset_into_db


def prepare_model_for_serving():
    
    # df = pd.read_csv("../data/finaldf.csv")
    #for docker 
    df = pd.read_csv(TRAINING_FILE)
    #we select only the rows we decided to keep for training the model
    selected_features = ['team_count','days_in_current_status','issue_type' ,\
                            'count_month_of_year','transictions_so_far','count_year']
        
        
    #cols to encode
    cols = ['issue_type']
    Encoder = preprocessing.OrdinalEncoder()
    for col in cols:
        Encoder.fit(df[col].values.reshape(-1, 1))
        df[col] = Encoder.transform(df[col].values.reshape(-1, 1))
    #define features and target
    y = df.logsec_to_sol.values
    X = df[selected_features].values
    
    #standardzation 
    Scaler = StandardScaler()
    X = Scaler.fit_transform(X,y)
    #fitting the model
    clf = ensemble.RandomForestRegressor()
    # xgb.XGBRegressor()
    clf.fit(X,y)
    return clf,Encoder,Scaler

load_dataset_into_db()
clf,Encoder,Scaler = prepare_model_for_serving()




# num = [([n], [StandardScaler()]) for n in selected_features]
# mapper = DataFrameMapper(num, df_out=True)

# clf = xgb.XGBRegressor()
# #     clf = LinearRegression()
# pipeline = Pipeline([
#     ("le",preprocessing.LabelEncoder),
#     ('preprocess', mapper),
#     ('clf', clf)
# ], verbose=True)






@app.route(ENTRY_POINT + "/release/<date>/resolved-since-now", methods=["GET"])
def get_resolved_issues_until_date(date):
    try:
        dt = datetime.datetime.strptime(date, "%Y-%m-%dT%H:%M:%S.%f%z")
        results = Issue.query.filter(Issue.predicted_resolution_date <= dt).all()
    except ValueError:
        if '-' in date:

            dt = datetime.datetime.strptime(date,'%Y-%m-%d')
            results = Issue.query.filter(Issue.predicted_resolution_date <= dt).all()
            
        else:
            return 'error in date format'
        
    l = []
    for r in results:
        result = {}
        result['issue'] = r.key
        result['predicted_resolution_date'] = r.predicted_resolution_date
        l.append(result)
    return jsonify(l)



@app.route(ENTRY_POINT + "/issue/<issue_key>/resolve-prediction", methods=["GET"])
def predict(issue_key):
    
    
    issue = db.session.query(Issue).filter_by(key=issue_key).order_by(desc('when')).first()
    #to check again
    if(issue is None):
        abort(404)

    if(issue.resolutiondate is not None):
        resolution = {'issue': issue_key,\
                      'resolution_date': issue.resolutiondate}
        return jsonify(resolution)
    
    else:
        
        encodedIssueType = Encoder.transform(np.array([issue.issue_type]).reshape(1, -1))
        result = { 'team_count':[issue.team_count],'days_in_current_status':[issue.days_in_current_status],\
                  'issue_type': [encodedIssueType],'count_month_of_year':[issue.count_month_of_year],\
                  'transictions_so_far':[issue.transictions_so_far],'count_year':[issue.count_year]}
        temp = pd.DataFrame.from_dict(result)
        
        result = Scaler.transform(temp.values)
        # print(result)
        prediction = clf.predict(result)
        
        when = datetime.datetime.timestamp(issue.when)
        forecastedTimestamp = when + int(np.exp(prediction))
        
        forecastedDatetime = datetime.datetime.fromtimestamp(forecastedTimestamp) 
        resolution = {'issue': issue_key,\
                      'predicted_resolution_date': forecastedDatetime}
        issue.predicted_resolution_date = forecastedDatetime
        db.session.commit()
        return jsonify(resolution)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
    

    
    