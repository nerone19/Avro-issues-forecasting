from config import *

class Issue(db.Model):
    __tablename__ = 'issues_api'
    _id = db.Column(db.Integer, primary_key=True)
    key = db.Column(db.String(80), unique=False, nullable=False)
    status = db.Column(db.String(120), unique=False, nullable=False)
    priority = db.Column(db.String(120), unique=False, nullable=False)
    issue_type = db.Column(db.String(120), unique=False, nullable=False)
    when = db.Column(db.DateTime(), unique=False, nullable=False)
    created = db.Column(db.String(120), unique=False, nullable=False)
    reporter = db.Column(db.String(120), unique=False, nullable=False)
    resolutiondate = db.Column(db.DateTime(), unique=False, nullable=True)
    predicted_resolution_date = db.Column(db.DateTime(), unique=False, nullable=True)
    # vote_count = db.Column(db.Integer, unique=False, nullable=False)
    days_in_current_status = db.Column(db.Float, unique=False, nullable=False)
    # project = db.Column(db.String(120), unique=False, nullable=False)
    # assignee = db.Column(db.String(120), unique=False, nullable=False)
    # comment_count = db.Column(db.Integer, unique=False, nullable=False)
    # descrption_length = db.Column(db.String(120), unique=False, nullable=False)
    # summary_length = db.Column(db.String(120), unique=False, nullable=False)
    # who = db.Column(db.String(120), unique=False, nullable=False)
    # to_status = db.Column(db.String(120), unique=False, nullable=False)
    # from_status = db.Column(db.String(120), unique=False, nullable=False)
    # watch_count = db.Column(db.Integer, unique=False, nullable=False)
    # weekday = db.Column(db.Integer, unique=False, nullable=False)
    # week_of_year = db.Column(db.Integer, unique=False, nullable=False)
    # month = db.Column(db.Integer, unique=False, nullable=False)
    # year = db.Column(db.Integer, unique=False, nullable=False)
    # day = db.Column(db.Integer, unique=False, nullable=False)
    # count = db.Column(db.Integer, unique=False, nullable=False)
    count_year = db.Column(db.Integer, unique=False, nullable=False)
    transictions_so_far = db.Column(db.Float, unique=False, nullable=False)
    count_month_of_year = db.Column(db.Integer, unique=False, nullable=False)
    team_count = db.Column(db.Float, unique=False, nullable=False)

    def __init__(self,key,status,priority,issue_type,created,reporter,when,days_in_current_status,resolutiondate,team_count,count_month_of_year,\
                 transictions_so_far,count_year):
        self.key = key
        self.status = status
        self.priority = priority
        self. issue_type = issue_type
        self.created = created 
        self.reporter = reporter
        self.when = when
        self.resolutiondate = resolutiondate
        self.days_in_current_status = days_in_current_status
        self.team_count =team_count
        self.count_month_of_year = count_month_of_year
        self.transictions_so_far = transictions_so_far
        self.count_year = count_year
        self.predicted_resolution_date = None
        
    def __repr__(self):
        return '<key %r>' % self.key


def load_data(file_name):
    data = genfromtxt(file_name, dtype=str,delimiter=',', skip_header=1)
    return data.tolist()


def load_dataset_into_db(): 
    # filename = "../data/complete_df.csv" 
    #for docker
    filename = DATABASE_FILE
    data = load_data(filename) 
    #drop everything from the db
    db.drop_all()
    #create everything again
    db.create_all()
    #clear database from previous rows
    db.session.query(Issue).delete()
    db.session.commit()
    
    print('loading csv dataset into postgres dataset')
    print("-" *64)
    
    try:
        for d in data:
    
            res = None
            if d[6] != '':
                res = datetime.datetime.strptime(d[6], "%Y-%m-%dT%H:%M:%S.%f%z")
            record = Issue(
                status = d[0],
                priority = d[1],
                issue_type = d[3],
                reporter = d[4],
                created = d[5],
                key = d[11],
                resolutiondate = res,
                team_count = d[37],
                days_in_current_status = d[19],
                count_month_of_year = d[35],
                transictions_so_far = d[31],
                count_year = d[36],
                when = datetime.datetime.strptime(d[18], "%Y-%m-%dT%H:%M:%S.%f%z")
            )
            db.session.add(record)
    
        db.session.commit() #Attempt to commit all the records
    except:
        print('something unexpected happened during the loading process')
        db.session.rollback() #Rollback the changes on error
        logging.error("exception ",exc_info=1) #or
    finally:
        db.session.close() #Close the connection
    
    print('finished loading data')
    print("-" *64)