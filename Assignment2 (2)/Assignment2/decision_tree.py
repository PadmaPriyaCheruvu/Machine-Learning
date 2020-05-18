import os
import graphviz
import matplotlib.pyplot as plt
import math
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
from sklearn import tree
from IPython.display import Image
import pydotplus



def partition(x):
    
    createInd = dict()
    l = len(x)
    for i in range(0, l):
        temp = x[i]
        if temp in createInd:
            createInd[x[i]].append(i)
        else:
            createInd[x[i]] = []
            createInd[x[i]].append(i)
    return createInd

    
def entropy(y):
    n = 0
    y1 = np.array(y)
    for i in set(y1):
        z=(y1==i)
        l=len(y1)
        tmp= z.sum()/l
        n = n + (tmp * math.log2(tmp))
    n=n*-1
    return n
    


def mutual_information(x, y):
    
    val = partition(x)
    Entropy1 = entropy(y)
    Entropy2 = 0
    for j in val:
        temp = []
        for i in val[j]:
            temp.append(y[i])
        length=len(x)
        temp2 = x.count(j)/length
        temp2=temp2*entropy(temp)
        Entropy2 = Entropy2 + temp2
    Entropy= Entropy1-Entropy2
    return Entropy


def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=5):
    if attribute_value_pairs == None :
        attribute_value_pairs = []
        shapeX=x.shape[1]
        for i in range(0, shapeX):
            for j in set(x[:, i]):
                attribute_value_pairs.append((i,j))

    if len(attribute_value_pairs) == 0 or depth == max_depth :
        cnt = np.bincount(np.array(y))
        val= np.argmax(cnt)
        return val
    
    elif all(t == y[0] for t in y) :
        return y[0]
    
    else :
        MaxVal = 0
        for item in attribute_value_pairs:
            temp = []
            k = item[0]
            length=len(x)
            for i in range(0, length):
                a = x[i][k]
                if a == item[1]:
                    temp.append(1)
                else:
                    temp.append(0)
            value = mutual_information(temp,y)
            if value >= MaxVal:
                MaxVal = value
                split_find = item
        
        a = split_find[1]
        b = split_find[0]
        temp = []
        for i in range(0, len(x)):
            temp.append(x[i][b])
            
        final =partition(temp)[a]
        
        Tx = []
        Fx = []
        Ty = []
        Fy = []
        
        length=len(x)
        for i in range(0, length):
            temp = np.asarray(x[i])
            if i in final:
                Tx.append(temp)
                Ty.append(y[i])
            else:
                Fx.append(temp)
                Fy.append(y[i])
        
        T = attribute_value_pairs.copy()
        F = attribute_value_pairs.copy()
        T.remove(split_find)
        F.remove(split_find)
        
        tree=dict()
        tree.update({(split_find[0], split_find[1], True): id3(Tx, Ty, T, depth+1, max_depth)})
        tree.update({(split_find[0], split_find[1], False): id3(Fx, Fy, F, depth+1, max_depth)})
        return tree


def predict_example(x, tree):
    
    try:
        len(tree.keys())
        
    except Exception:
        return tree
    
    keys=tree.keys()
    item = list(keys)[0]
    
    if x[item[0]] == item[1]:
        return predict_example(x, tree[(item[0], item[1], True)])
    else:
        return predict_example(x, tree[(item[0], item[1], False)])


def compute_error(y_true, y_pred):

    count = 0
    length=len(y_true)
    for i in range(0, length):
        if y_true[i] == y_pred[i]:
            count = count + 0
        else:
            count=count+1
    return count/length


def pretty_print(tree, depth=0):
    if depth == 0:
        print('TREE')

    for index, split_criterion in enumerate(tree):
        sub_trees = tree[split_criterion]

        print('|\t' * depth, end='')
        print('+-- [SPLIT: x{0} = {1} {2}]'.format(split_criterion[0], split_criterion[1], split_criterion[2]))

        # Print the children
        if type(sub_trees) is dict:
            pretty_print(sub_trees, depth + 1)
        else:
            print('|\t' * (depth + 1), end='')
            print('+-- [LABEL = {0}]'.format(sub_trees))


def render_dot_file(dot_string, save_file, image_format='png'):
    if type(dot_string).__name__ != 'str':
        raise TypeError('visualize() requires a string representation of a decision tree.\nUse tree.export_graphviz()'
                        'for decision trees produced by scikit-learn and to_graphviz() for decision trees produced by'
                        'your code.\n')
    os.environ["PATH"] += os.pathsep + 'D:/ANACONDA 3/pkgs/graphviz-2.38-hfd603c8_2/Library/bin/graphviz' 
    graph = graphviz.Source(dot_string)
    graph.format = image_format
    graph.render(save_file, view=True)


def to_graphviz(tree, dot_string='', uid=-1, depth=0):
    uid += 1       
    node_id = uid  

    if depth == 0:
        dot_string += 'digraph TREE {\n'

    for split_criterion in tree:
        sub_trees = tree[split_criterion]
        attribute_index = split_criterion[0]
        attribute_value = split_criterion[1]
        split_decision = split_criterion[2]

        if not split_decision:
            dot_string += '    node{0} [label="x{1} = {2}?"];\n'.format(node_id, attribute_index, attribute_value)

        if type(sub_trees) is dict:
            if not split_decision:
                dot_string, right_child, uid = to_graphviz(sub_trees, dot_string=dot_string, uid=uid, depth=depth + 1)
                dot_string += '    node{0} -> node{1} [label="False"];\n'.format(node_id, right_child)
            else:
                dot_string, left_child, uid = to_graphviz(sub_trees, dot_string=dot_string, uid=uid, depth=depth + 1)
                dot_string += '    node{0} -> node{1} [label="True"];\n'.format(node_id, left_child)

        else:
            uid += 1
            dot_string += '    node{0} [label="y = {1}"];\n'.format(uid, sub_trees)
            if not split_decision:
                dot_string += '    node{0} -> node{1} [label="False"];\n'.format(node_id, uid)
            else:
                dot_string += '    node{0} -> node{1} [label="True"];\n'.format(node_id, uid)

    if depth == 0:
        dot_string += '}\n'
        return dot_string
    else:
        return dot_string, node_id, uid



def loading_function(i):
    Test_Path = "./monks-" + str(i) + ".test"
    M = np.genfromtxt(Test_Path, missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]

    
    Train_Path = "./monks-" + str(i) + ".train"
    M = np.genfromtxt(Train_Path, missing_values=0, skip_header=0, delimiter=',', dtype=int)
    Y_train = M[:, 0]
    X_train = M[:, 1:]


    
    trnError = dict()
    tstError = dict()
    
    for j in range(1, 11):
        decision_tree = id3(X_train, Y_train, max_depth=j)
        trainy_pred = [predict_example(x, decision_tree) for x in X_train]
        trn_err = compute_error(Y_train, trainy_pred)
        testy_pred = [predict_example(x, decision_tree) for x in Xtst]
        tst_err = compute_error(ytst, testy_pred)
        
        trnError[j] = trn_err
        tstError[j] = tst_err
    plt.figure() 
    plt.plot(list(trnError.keys()), list(trnError.values()), marker='o', linewidth=2, markersize=10) 
    plt.plot(list(tstError.keys()), list(tstError.values()), marker='s', linewidth=2, markersize=10) 
    plt.xlabel('Depth', fontsize=15) 
    plt.ylabel('Error', fontsize=15) 
    plt.xticks(list(trnError.keys()), fontsize=18) 
    plt.legend(['Training Error', 'Test Error'], fontsize=16) 
    plt.xscale('log')
    plt.yscale('log')
    plt.title("Monks Dataset-"+str(i))
    
    

    
for i in range(1, 4):
    loading_function(i)

M = np.genfromtxt('./monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
Y_train = M[:, 0]
X_train = M[:, 1:]

M = np.genfromtxt('./monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
ytst = M[:, 0]
Xtst = M[:, 1:]

for i in range(1, 6, 2):
    Final_Tree = id3(X_train, Y_train, max_depth=i)
    pretty_print(Final_Tree)
    dot_str = to_graphviz(Final_Tree)
    render_dot_file(dot_str, './monks1learn-'+str(i))
    y_pred = [predict_example(x, Final_Tree) for x in Xtst]
    
    print("Monks Dataset: Confusion matrix: Depth ",i )
    print(pd.DataFrame(confusion_matrix(ytst, y_pred), 
                       columns=['Predicted Positives', 'Predicted Negatives'],
                       index=['True Positives', 'True Negatives']
                       ))

for i in range(1, 6, 2):
    Data_feature_names = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6'] 
    Final_Tree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=i)
    Final_Tree.fit(X_train, Y_train)
    dot_data = tree.export_graphviz(Final_Tree, out_file=None, feature_names=Data_feature_names,
                            filled=True, rounded=True, special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png('monks1sklearn-'+str(i)+'.png')
    Image(filename='monks1sklearn-'+str(i)+'.png')
    y_pred = Final_Tree.predict(Xtst)
    
    print("MONKS DATASET: Confusion matrix (SKLearn Decision Tree) for depth ",i )
    print(pd.DataFrame(confusion_matrix(ytst, y_pred), 
                       columns=['Predicted Positives', 'Predicted Negatives'],
                       index=['True Positives', 'True Negatives']
                       ))
   
M = np.genfromtxt('./census-income.data', missing_values=0, skip_header=1, delimiter=',', dtype=int)
Train = M[:, 2:6]

M = np.genfromtxt('./census-income.data', missing_values=0, skip_header=1, delimiter=',', dtype=int)
Test = M[:, 2:6]

def Bin(Val):
    for j in range(0, Val.shape[1]):
        sum = 0
        cnt = 0
        for i in range(0, Val.shape[0]):
            sum = sum + Val[i][j]
            cnt = cnt + 1
        mean = sum/cnt
        for i in range(0, Val.shape[0]):
            if Val[i][j] <= mean:
                Val[i][j] = 0
            else:
                Val[i][j] = 1
Train=Bin(Train)
Test=Bin(Test)
                
           
for i in range(1, 6, 2):
    Final_Tree = id3(X_train, Y_train, max_depth=i)
    pretty_print(Final_Tree)
    dot_str = to_graphviz(Final_Tree)
    render_dot_file(dot_str, './censuslearn-'+str(i))
    y_pred = [predict_example(x, Final_Tree) for x in Xtst]

    print("CENSUS DATASET: Confusion matrix (Learned Decision Tree) for depth ",i )
    print(pd.DataFrame(confusion_matrix(ytst, y_pred), 
                       columns=['Predicted Positives', 'Predicted Negatives'],
                       index=['True Positives', 'True Negatives']
                       ))
    
for i in range(1, 6, 2):
    names = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6'] 
    Final_Tree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=i)
    Final_Tree.fit(X_train, Y_train)
    dot_data = tree.export_graphviz(Final_Tree, out_file=None, feature_names=names,
                            filled=True, rounded=True, special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png('censussklearn-'+str(i)+'.png')
    Image(filename='censussklearn-'+str(i)+'.png')
    y_pred = Final_Tree.predict(Xtst)

    print("CENSUS DATASET: Confusion matrix (SKLearn Decision Tree) for depth ",i )
    print(pd.DataFrame(confusion_matrix(ytst, y_pred), 
                       columns=['Predicted Positives', 'Predicted Negatives'],
                       index=['True Positives', 'True Negatives']
                       ))   
 

    
