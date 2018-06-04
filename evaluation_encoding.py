import networkx as nx
import numpy as np
#import matplotlib.pyplot as plt
#from sklearn import tree
#from sklearn import metrics

#from sklearn.ensemble import RandomForestClassifier
#from sklearn import svm
#from sklearn.naive_bayes import GaussianNB
#from sklearn import linear_model
#from sklearn.linear_model import SGDClassifier

CHEM_EL = {1:{'name':'H','color':'yellow'},
2:{'name':'He','color':'grey'},
3:{'name':'Li','color':'grey'},
4:{'name':'Be','color':'grey'},
5:{'name':'B','color':'grey'},
6:{'name':'C','color':'orange'},
7:{'name':'N','color':'red'},
8:{'name':'O','color':'green'},
9:{'name':'F','color':'blue'},
10:{'name':'Ne','color':'grey'},
11:{'name':'Na','color':'grey'},
12:{'name':'Mg','color':'grey'},
13:{'name':'Al','color':'grey'},
14:{'name':'Si','color':'grey'},
15:{'name':'P','color':'grey'},
16:{'name':'S','color':'violet'},
17:{'name':'Cl','color':'brown'},
18:{'name':'Ar','color':'grey'},
19:{'name':'K','color':'grey'},
20:{'name':'Ca','color':'grey'},
21:{'name':'Sc','color':'grey'},
22:{'name':'Ti','color':'grey'},
23:{'name':'V','color':'grey'},
24:{'name':'Cr','color':'grey'},
25:{'name':'Mn','color':'grey'},
26:{'name':'Fe','color':'grey'},
27:{'name':'Co','color':'grey'},
28:{'name':'Ni','color':'grey'},
29:{'name':'Cu','color':'grey'},
30:{'name':'Zn','color':'grey'},
31:{'name':'Ga','color':'grey'},
32:{'name':'Ge','color':'grey'},
33:{'name':'As','color':'grey'},
34:{'name':'Se','color':'grey'},
35:{'name':'Br','color':'grey'},
36:{'name':'Kr','color':'grey'},
37:{'name':'Rb','color':'grey'},
38:{'name':'Sr','color':'grey'},
39:{'name':'Y','color':'grey'},
40:{'name':'Zr','color':'grey'},
41:{'name':'Nb','color':'grey'},
42:{'name':'Mo','color':'grey'},
43:{'name':'Tc','color':'grey'},
44:{'name':'Ru','color':'grey'},
45:{'name':'Rh','color':'grey'},
46:{'name':'Pd','color':'grey'},
47:{'name':'Ag','color':'grey'},
48:{'name':'Cd','color':'grey'},
49:{'name':'In','color':'grey'},
50:{'name':'Sn','color':'grey'},
51:{'name':'Sb','color':'grey'},
52:{'name':'Te','color':'grey'},
53:{'name':'I','color':'grey'},
54:{'name':'Xe','color':'grey'},
55:{'name':'Cs','color':'grey'},
56:{'name':'Ba','color':'grey'},
57:{'name':'Hf','color':'grey'},
58:{'name':'Ta','color':'grey'},
59:{'name':'W','color':'grey'},
60:{'name':'Re','color':'grey'},
61:{'name':'Os','color':'grey'},
62:{'name':'Ir','color':'grey'},
63:{'name':'Pt','color':'grey'},
64:{'name':'Au','color':'grey'},
65:{'name':'Hg','color':'grey'},
66:{'name':'Tl','color':'grey'},
67:{'name':'Pb','color':'grey'},
68:{'name':'Bi','color':'grey'},
69:{'name':'Po','color':'grey'},
70:{'name':'At','color':'grey'},
71:{'name':'Rn','color':'grey'},
72:{'name':'Fr','color':'grey'},
73:{'name':'Ra','color':'grey'},
74:{'name':'Pt','color':'grey'},
75:{'name':'Ac','color':'grey'},
76:{'name':'La','color':'grey'},
77:{'name':'U','color':'grey'},
78:{'name':'Sm','color':'grey'},
79:{'name':'Ce','color':'grey'},
80:{'name':'Nd','color':'grey'},
81:{'name':'Eu','color':'grey'},
82:{'name':'Gd','color':'grey'},
83:{'name':'Dy','color':'grey'},
84:{'name':'Er','color':'grey'},
85:{'name':'Rh','color':'grey'}}
#%%
def isomorph_test(G1,G2):
    em = nx.algorithms.isomorphism.categorical_edge_match('weight', 1)
    nm = nx.algorithms.isomorphism.categorical_node_match('label', 'C')
    GM = nx.algorithms.isomorphism.GraphMatcher(G1,G2,node_match=nm,edge_match=em) #G1-main graph, G2-subgraph
    
    return GM.subgraph_is_isomorphic()
#%%
# parameters:
num_substructures = 100
num_samplings = 200
#total_num_features = 192

# Open file for classification:
input_file = 'S9'#'S10','S11'

f = open('evaluation//'+input_file+'_hless.gsp','r')
M_list = []

##count = 0
for line in f:
    try:
        if '#' in line:
            G1 = nx.Graph()
            #labels1 = {}
            #node_colors1 = []
            line = f.next()
            while 'v' in line:
                data = line.split(' ')
                node = np.int(data[1])
                element = np.int(data[2])
                #labels1[node] = CHEM_EL[element]['name']
                #node_colors1.append(CHEM_EL[element]['color'])
                #G1.add_node(node,label=labels1[node])
                G1.add_node(node,label=CHEM_EL[element]['name'])
                line = f.next()
            while 'e' in line:
                data = line.split(' ')
                vertice1 = np.int(data[1])
                vertice2 = np.int(data[2])
                G1.add_edge(vertice1,vertice2,weight=np.int(data[3]))
                line = f.next()
            M_list.append(G1)
            ##if isomorph_test(G1,G2):
            ##    bin_matrix[count] = True
            ##count += 1
    except ValueError:
        print "Invalid input:", line
f.close()

# number of molecules for classification:
num_molecules_classify = len(M_list)
 
# dimensions of input data:
pred_X = np.zeros((num_samplings,num_molecules_classify,num_substructures),bool)

for i in range(num_samplings):
    print input_file,'sampling',i+1
    
    # Open top discriminative substructures for encoding the input data:
    f = open('classifier//top_sg_'+str(i+1)+'.gsp','r')
    SG_list = []
    
    for line in f:
        try:
            if '#' in line:
                G2 = nx.Graph()
                #labels2 = {}
                #node_colors2 = []
                line = f.next()
                while 'v' in line:
                    data = line.split(' ')
                    node = np.int(data[1])
                    element = np.int(data[2])
                    #labels2[node] = CHEM_EL[element]['name']
                    #node_colors2.append(CHEM_EL[element]['color'])
                    #G2.add_node(node,label=labels2[node])
                    G2.add_node(node,label=CHEM_EL[element]['name'])
                    line = f.next()
                while 'e' in line:
                    data = line.split(' ')
                    vertice1 = np.int(data[1])
                    vertice2 = np.int(data[2])
                    G2.add_edge(vertice1,vertice2,weight=np.int(data[3]))
                    line = f.next()
                SG_list.append(G2)
        except ValueError:
            print "Invalid input:", line
    f.close()
        
    # Encode the input data:
    for i_molecule in range(num_molecules_classify):
        for i_substructure in range(len(SG_list)):
            pred_X[i][i_molecule][i_substructure] = isomorph_test(M_list[i_molecule],SG_list[i_substructure])
    
# save results of encoding:   
np.save('evaluation//'+input_file+'_encoded.npy',pred_X)
#%%