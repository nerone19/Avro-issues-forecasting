import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import dateutil.parser as dp


def parse_iso_to_seconds(time,toCompare):
    parsed_t = dp.parse(time)
    t_in_seconds = parsed_t.timestamp()
    return t_in_seconds

df = pd.read_csv (r'./data/avro-issues.csv')
dfTransition = pd.read_csv (r'./data/avro-transitions.csv')

print (df)

#651 rows are NAN. Shall we delete them?
df['resolutiondate'].isna().sum()


#resolution is redundant  for resolutiondate or may affect time or resolution?
df['resolution'].isna().sum()
#how many values?
df['resolutiondate'].describe()

#2166 unique keys out of 6260 in total. it means there are chains from open to resolved issues
dfTransition['key'].value_counts()

#df transition has value 'closed' for resolved issues whereas issues has just value equal to resolved
#closed and resolved are the same thing?
topFreq = dfTransition[dfTransition['key'] == 'AVRO-1614']
topFreq.iloc[topFreq['days_since_open'].argmax()]
#look for resolved and not closed issues
dfTransition[dfTransition['status'] == 'Resolved'].describe()

#!
transitionsResolved = dfTransition[dfTransition['to_status'] == 'Resolved']
#for each resolved issued,get the number of transition before the solution and the number of days taken before completion
transitionsResolved['key'].value_counts()


#------------------------------------------------------------------

transitionsResolved.sort_values(by='when',ascending =True) 
#we create a new features for counting how many transitions there were before the solution and which was the previous step
history_resolved_transitions = pd.DataFrame(columns=['num_transitions', 'last_transition'])
for index, row in transitionsResolved.iterrows():

    key = row['key']
    #extract all the transitions for the specified issues with the relative key
    t = dfTransition[dfTransition['key'] == key]
    parsed_t = dp.parse(row['when'])
    t_in_seconds_to_compare = parsed_t.timestamp()
    
    count = 0
    for i,el in t.iterrows(): 
        parsed_t = dp.parse(el['when'])
        t_in_seconds = parsed_t.timestamp()
        if(t_in_seconds < t_in_seconds_to_compare):
            last_transition_idx = i
            count += 1
    history_resolved_transitions = history_resolved_transitions.append({'num_transitions':count, \
                                                                        'last_transition': t.loc[last_transition_idx]['to_status']}\
                                                                       , ignore_index=True)


#frequency for last transition
for i in history_resolved_transitions.columns:
    sns.barplot(history_resolved_transitions[i].value_counts().index,history_resolved_transitions[i].value_counts()).set_title(i)
    plt.show()
    


#frequenct num transition for last transition
for col in history_resolved_transitions.last_transition.unique():
    d = history_resolved_transitions[history_resolved_transitions['last_transition'] == col]
    ax = sns.countplot(x="num_transitions",  data=d).set_title(col)
    plt.show()




#todo: what is the frequency of the number of transitions required to get the solution? Do we need additional feature engineering?

dfTransition['status'].value_counts()


#------------------------------------------------------------------
#what is the usual time taken to solve issues?

ax = sns.distplot(transitionsResolved['days_since_open'])
plt.show()
#usually we take around 1 day to solve issues. We may use minutes for predicting time?