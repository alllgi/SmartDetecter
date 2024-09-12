import sys

from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn import preprocessing
lbl = preprocessing.LabelEncoder()
from sklearn.model_selection import StratifiedKFold
import joblib
from tkinter import _flatten
from sklearn import preprocessing
from sklearn.metrics import f1_score


def culatescore(predict, real):
    scores=[]
    f1 = f1_score(real, predict, average='macro')
    scores.append(f1)
    return scores

def caculate(predict):
    ff = open('final_result.txt','w')
    input='./pairs_final.txt'
    f=open(input,'r')
    lines=f.readlines()
    TP=0
    TN=0
    FP=0
    FN=0
    index=0
    for i in lines:
        groundtruth=i.split('\t')[-1].replace('\n','')
        # predict=i.split('\t')[1]
        # print(index)
        if '1' in groundtruth:
           if predict[index]==1:
                TP+=1
                ff.write('1\n')
           else:
                FN+=1
                ff.write('0\n')
        else:
            if predict[index]==1:
                FP+=1
                ff.write('1\n')
            else:
                TN+=1
                ff.write('0\n')
        index+=1
    print(TP,TN,FP,FN)
    acc=(TP+TN)/(TP+TN+FP+FN)
    pre=(TP)/(TP+FP)
    recall=TP/(TP+FN)
    f1=(2*pre*recall)/(pre+recall)
    print("Precision:"+str(pre)+"\tRecall:"+str(recall)+"\tF1-score:"+str(f1))

def detect_similarity(test_data,lss):
    index=0
    a = []
    lens = len(lss)
    results = ""
    predicts = []
    test_data.columns = ['fid1', 'fid2', 'type', 'name', 'names', 'value', 'unit', 'operater', 'memberName', 'other']
    for j in lss:
        fid1 = test_data['fid1'][index].split('_')[0]
        if index < lens - 1:
            next = test_data['fid1'][index+1].split('_')[0]
        else:
            next = ""
        a.append(j)
        if fid1 != next:
            y = int(test_data['fid2'][index].split('_')[1])
            a = np.array(a)
            a = a.reshape(y, -1)
            rows = a.shape[0]
            columns = a.shape[1]
            countX = 0
            countY = 0
            for row in range(a.shape[0]):
                if 1 in a[row, :]:
                    countX += 1
            for col in range(a.shape[1]):
                if 1 in a[:, col]:
                    countY += 1
            simX = countX / rows
            simY = countY / columns
            if simX >= 0.5 and simY >= 0.5:
                fid2 = test_data['fid2'][index].split('_')[0] + '_' + test_data['fid2'][index].split('_')[1]
                predicts.append(1)
                return a.tolist(),True
            else:
                predicts.append(0)
                return '',False
            a = []
        index += 1
def gen_report(test_data,results):
    source=''
    results=list(_flatten(results))
    test_data.columns = ['fid1', 'fid2', 'type', 'name', 'names', 'value', 'unit', 'operater', 'memberName', 'other']
    for i in range(0,len(results)):
        if results[i]==1:
            f1='testContracts/'+test_data['fid1'][i].split('_')[0]+'.sol'
            struct1='SR'+test_data['fid1'][i].split('_')[1]+':'
            start1=int(test_data['fid1'][i].split('_')[2])
            end1=int(test_data['fid1'][i].split('_')[3])
            f2 = 'testContracts/'+test_data['fid2'][i].split('_')[0]+'.sol'
            struct2 = 'SR' + test_data['fid2'][i].split('_')[1]+':'
            start2 = int(test_data['fid2'][i].split('_')[2])
            end2 = int(test_data['fid2'][i].split('_')[3])
            c1=open(f1,'r')
            lines1=c1.readlines()
            c2=open(f2,'r')
            lines2=c2.readlines()
            s1=''
            s2=''
            for j in range(start1-1,end1):
                s1=s1+lines1[j]
            for z in range(start2-1,end2):
                s2=s2+lines2[z]
            source=source+'<tr><td>'+struct1+s1.replace('<',' &lt;').replace('>',' &gt;')+'</td>\n'
            source = source + '<td>' +struct2+ s2.replace('<',' &lt;').replace('>',' &gt;') + '</td></tr>\n'
    source = source+'</table></div></body></html>'
    f3=open('data/template.html','r')
    source=f3.read()+'<tr><td style="text-align:center;background-color: rgba(161, 161, 161, 0.6);">' +\
           test_data['fid1'][0].split('_')[0]+ \
             '</td><td style="text-align:center;background-color: rgba(161, 161, 161, 0.6);">' +\
           test_data['fid2'][0].split('_')[0]+ '</td></tr>'\
           +source
    report = open('testContracts/report.html','w')
    report.write(source)
# load or create your dataset

def train():
    print('Load data...')
    lbl = preprocessing.LabelEncoder()
    train_data = pd.read_csv('data/train.csv', header=None)
    train_data.columns = ['fid1','fid2','type','name','names','value','unit','operater','memberName','other','label']
    train_label=train_data['label']
    train_data.drop(['label','fid1','fid2'],axis=1,inplace=True)
    train_data['type'] = train_data['type'].astype('category')
    train_data['name'] = train_data['name'].astype('category')
    train_data['names'] = train_data['names'].astype('category')
    train_data['value'] = train_data['value'].astype('category')
    train_data['unit'] = train_data['unit'].astype('category')
    train_data['operater'] = train_data['operater'].astype('category')
    train_data['memberName'] = train_data['memberName'].astype('category')
    train_data['other'] = train_data['other'].astype('category')

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': 31,
        'max_bin': 50,
        'max_depth': 6,
        "learning_rate": 0.02,
        "colsample_bytree": 0.8,
        "bagging_fraction": 0.8,
        'min_child_samples': 25,
        'n_jobs': -1,
        'silent': True,
        'seed': 1000,
        'force_col_wise': True,
    }

    results = []
    bigtestresults = []
    smalltestresults = []
    scores = []
    cat = ['type', 'name','names','value','unit','operater','memberName','other']
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=123)
    x, y = pd.DataFrame(train_data), pd.DataFrame(train_label)
    for i, (train_index, valid_index) in enumerate(kf.split(x, y)):
        results=[]
        print("NO. ", i + 1, " Times")
        x_train, y_train = x.iloc[train_index], y.iloc[train_index]
        x_valid, y_valid = x.iloc[valid_index], y.iloc[valid_index]
        lgb_train = lgb.Dataset(x_train, y_train, categorical_feature=cat, silent=True)
        lgb_eval = lgb.Dataset(x_valid, y_valid, reference=lgb_train, categorical_feature=cat, silent=True)
        gbm = lgb.train(params, lgb_train, num_boost_round=400, valid_sets=[lgb_train, lgb_eval], categorical_feature=cat,
                        verbose_eval=100,
                        early_stopping_rounds=200)
        vaild_preds = gbm.predict(x_valid, num_iteration=gbm.best_iteration)
        threshold=0.7
        index=0
        for pred in vaild_preds:
            result = 1 if pred > threshold else 0
            results.append(result)
            index+=1
        c = culatescore(results, y_valid)
        print(c)
    joblib.dump(gbm, r'data/model_1.pkl')
    print('---cross validation score---')
    print(np.average(c))
    print(pd.DataFrame({'category':cat,'importance': gbm.feature_importance()}))
def test():
    test_data = pd.read_csv('testContracts/test_pairs.csv', header=None)
    test_data.columns = ['fid1', 'fid2', 'type', 'name', 'names', 'value', 'unit', 'operater', 'memberName', 'other']
    test_data['type'] = test_data['type'].astype('category')
    test_data['name'] = test_data['name'].astype('category')
    test_data['names'] = test_data['names'].astype('category')
    test_data['value'] = test_data['value'].astype('category')
    test_data['unit'] = test_data['unit'].astype('category')
    test_data['operater'] = test_data['operater'].astype('category')
    test_data['memberName'] = test_data['memberName'].astype('category')
    test_data['other'] = test_data['other'].astype('category')
    cat = ['type', 'name', 'names', 'value', 'unit', 'operater', 'memberName', 'other']
    gbm=joblib.load('data/model.pkl')
    test_pre = gbm.predict(test_data.iloc[:, 2:11], num_iteration=gbm.best_iteration, predict_disable_shape_check='true')
    threshold = 0.5 # set threshold
    smalltestresults = []
    bigtestresults = []
    for w in test_pre:
        temp = 1 if w > threshold else 0
        smalltestresults.append(temp)
    bigtestresults.append(smalltestresults)
    results = []
    finalpres = pd.DataFrame(bigtestresults)
    finaltask = []
    lss = []  # This is the final result
    for i in finalpres.columns:
        temp1 = finalpres.iloc[:, i].value_counts().index[0]
        lss.append(temp1)
    res,sim=detect_similarity(test_data,lss)
    if sim:
        print('Warning!!CodeClone')
        print('See the testContractes/report.html for details')
        gen_report(test_data,res)
    else:
        print('No CodeClone exists here')
if __name__ == "__main__":
    if sys.argv[1] == "--train":
        train()
    elif sys.argv[1] == "--test":
        test()
