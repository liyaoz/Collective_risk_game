README
# Collective-Risk Dilemma + Q-learning
This repo contains files that support the project of studying Q-learning agents
in a collective-risk dilemma.

---

To test the data of the thesis, **uncomment any code blocks in main.py/main()**
 and run:

$ python3 main.py

---

Most methods in main.py have the flexibility of passing different number of 
parameters in **kwargs. If not specified, a parameter will be defaulted with 
the value defined in __init__() methods.

To extend the tests, stackBar() and t_test() in main.py were implemented 
with the flexibility of specifying which parameter is of interest. 
Simply pass a list or tuple to the method with the interested keyword, as 
part of **kwargs.

E.g. `stackBar(0, Actions, repeat=repeat, alpha=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0
.6, 0.7, 0.8, 0.9, 1])`
compares on different alpha value, while

`stackBar(0, Actions, repeat=repeat, threshold=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 
0.7, 0.8, 0.9, 1])` compares on thresholds.


---
 
The graph.py implemented a 'small-world network' model for agent pairing. 
Current tests found no significant effect on observed results. The model is 
kept for further study, but only well-mixed graphs were deployed so far.

---

Jupyter Notebook supported. Use the .ipynb file.

To avoid installing packages and setting up virtualenv, upload .ipynb file in 
Google Colab.

---

Author: Liyao Zhu liyaoz@student.unimelb.edu.au

Last Update: 12 Jun. 2019

