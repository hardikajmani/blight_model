'''
  Many countries are affected by the graffitis and various arts by the people on the walls. For which if the wall is private owners refuse to get 
  it fixed, which is eventually charged by the government as a penalty.

  Now this is many a times paid but most of the times not paid by the owner of that wall.

  So, this code deals with the same, based upon the previous instances and many features it allows us to predict the probability whether the 
  test users would pay thier fines or not.

  This question was the final project of the course Applied ML in Python, so based on the dataset given to me , I chosed best 2 features to train and
  get the required roc_auc score.

  Also their are various plots in the end to depict its accuracy in the end.

  As the final output the function predicts the probability of the test users given in a different datasets and returns its in form of a series


                                                                                           -- Hardik Ajmani


'''





import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt


def blight_model():
    
    # Your code here

    # Read dataset
    df = pd.read_csv("train.csv", encoding = 'ISO-8859-1', low_memory = False)
    df = df[np.isfinite(df['compliance'])]
    df2 = pd.read_csv("test.csv")
    add_df = pd.read_csv("addresses.csv")
    lat_df = pd.read_csv("latlons.csv")


    # assign y,X for training and X2 for testing

    y = df['compliance'].values
    X = df[list(['judgment_amount','late_fee'])].values
    X2 = df2[list(['judgment_amount','late_fee'])].values

    
    # Split dataset into train and test/dev dataset using an inbuilt function of sklearn

    X_train,X_test, y_train, y_test = train_test_split(X,y, random_state = 0 )


    # training the data using Gradient Booster classifier 
     
    clf = GradientBoostingClassifier().fit(X_train, y_train)
    
    # Predicting various score on the trained model

    y_score = clf.decision_function(X_test)
    print('Score training Set: ' + str(clf.score(X_train,y_train)))
    print('Score test set: ' + str(clf.score(X_test,y_test)))
    print('ROC_AUC Score:' + str(roc_auc_score(y_test, y_score)) + "\n")


    # Testing is important ,b ut measuring that is even more thus plotting precision recall curve for the model
    # Now the curve depicts that or model is more precision oriented than recall
    
    precision, recall, thresholds = precision_recall_curve(y_test, y_score)
    closest_zero = np.argmin(np.abs(thresholds))
    closest_zero_p = precision[closest_zero]
    closest_zero_r = recall[closest_zero]

    plt.figure()
    plt.xlim([0.0, 1.01])
    plt.ylim([0.0, 1.01])
    plt.plot(precision, recall, label='Precision-Recall Curve')
    plt.plot(closest_zero_p, closest_zero_r, 'o', markersize = 12, fillstyle = 'none', c='r', mew=3)
    plt.xlabel('Precision', fontsize=16)
    plt.ylabel('Recall', fontsize=16)
    plt.axes().set_aspect('equal')
    plt.show()


    # plotting roc_auc graph 

    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_score)
    roc_auc_lr = auc(fpr_lr, tpr_lr)
    plt.figure()
    plt.xlim([-0.01, 1.00])
    plt.ylim([-0.01, 1.01])
    plt.plot(fpr_lr, tpr_lr, lw=3, label='LogRegr ROC curve (area = {:0.2f})'.format(roc_auc_lr))
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('ROC curve ', fontsize=16)
    plt.legend(loc='lower right', fontsize=13)
    plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
    plt.axes().set_aspect('equal')
    plt.show()
    



    ind = df2['ticket_id'].values
    #print(ind)
    pred2 = clf.predict_proba(X2)[:,1]
    
    #print(pred)
    #print(np.shape(pred))
    #print(np.shape(ind))
    
    
    ans = pd.Series(pred2, ind, dtype='float64')
    
    return ans
#blight_model()


#
bm = blight_model()
res = 'Data type Test: '
res += ['Failed: type(bm) should Series\n','Passed\n'][type(bm)==pd.Series]
res += 'Data shape Test: '
res += ['Failed: len(bm) should be 61001\n','Passed\n'][len(bm)==61001]
res += 'Data Values Test: '
res += ['Failed: all values should be in [0.,1.]\n','Passed\n'][all((bm<=1.) & (bm>=0.))]
res += 'Data Values type Test: '
res += ['Failed: bm.dtype should be float\n','Passed\n'][str(bm.dtype).count('float')>0]
res += 'Index type Test: '
res += ['Failed: type(bm.index) should be Int64Index\n','Passed\n'][type(bm.index)==pd.Int64Index]
res += 'Index values type Test: '
res += ['Failed: type(bm.index[0]) should be int64\n','Passed\n'][str(type(bm.index[0])).count("int64")>0]

res += 'Output index shape test:'
res += ['Failed, bm.index.shape should be (61001,)\n','Passed\n'][bm.index.shape==(61001,)]

res += 'Output index test: '
if bm.index.shape==(61001,):
    res +=['Failed\n','Passed\n'][all(pd.read_csv('test.csv',usecols=[0],index_col=0).sort_index().index.values==bm.sort_index().index.values)]
else:
    res+='Failed'
print(res)