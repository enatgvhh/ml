#

Systemsynthese
==============
Mit der Systemsynthese wollen wir ein System mathematisch beschreiben. Mit diesem mathematischen Modell sind dann Vorhersagen möglich. Beim Machine Learning sprechen wir von Supervised Learning. Dieses werden wir anhand unserer geclusterten Stadtteile tun. Durch die Clusteranalyse hatten wir jeden Stadtteil einem Cluster (Label) zugeordnet. Gesucht ist nun ein Modell, das diese Cluster optimal trennt.

## Diskriminanzanalyse
Dazu bedienen wir uns einer Diskriminanzanalyse, mit der Diskriminanz-Funktionen aufgestellt werden, die die Cluster trennen und eine Klassifikation nicht gelabelter Samples ermöglichen.
Der gesamte Quellcode des Kapitels ist in einem Jupyter Notebook [IPYNB-File](src/pub_4_discriminant.ipynb) gespeichert. Zunächst importieren wir die benötigten Python packages.
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
```
Lesen die gelabelten Daten aus dem CSV-File in einen pandas DataFrame ein. Diese Daten sind bereits normiert.
```
strSource = r"D:\ML\work\StadtteilprofileBerichtsjahr2018_cluster.csv"
df = pd.read_csv(strSource, sep=';', header=0, encoding='utf-8')
```
Dann splitten wir den Datensatz in einen Trainings- und einen Test-Datensatz. Wie der Name schon sagt, mit den Trainingsdaten trainieren wir das Modell und mit den Testdaten validieren wir es. Um den Vorgang reproduzieren zu können, belegen wir den random_state mit einem festen Wert.
```
X = df.loc[:,df.columns.difference(['id','Stadtteil','Cluster'])]
y = df.loc[:,['Cluster']].to_numpy().ravel()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8, stratify=y)
```
Das Modell mit den Trainingsdaten trainieren.
```
lda = LinearDiscriminantAnalysis()
lda.fit(X_train,y_train)
```
Bevor wir mit dem Modell Samples klassifizieren, wollen wir uns etwas genauer ansehen, was hier eigentlich passiert. Dazu transformieren wir mit dem trainierten Modell die trainierten Daten. Dabei werden die 55 Dimensionen (Features) zu 5 Dimensionen reduziert, d.h. es werden 5 Diskriminanzfunktionen gebildet (Anzahl Cluster - 1). Diese Funktionen haben die allgemeine Gleichung y = b0 + b1X1 + b2X2 + … + b55X55 wobei b die Gewichte sind, die erlernt werden. Diese Gleichung begegnet uns übrigens auch bei der Berechnung der linearen Kombination in künstlichen neuronalen Netzen wieder. Als Ergebnis erhalten wir für jedes Sample 5 Funktionswerte. Im Streudiagramm sind die Funktionswerte für die ersten beiden Diskriminanzfunktionen dargestellt sowie die Klassenmittel. Über das Attribut explained_variance_ratio_ erfahren wir, wieviel Prozent der Varianz durch die einzelnen Funktionen erklärt werden. Wir sehen, die ersten 4 Funktionen erklären bereits 95% der Varianz.
```
X_lda = lda.transform(X_train)

print('Original number of features:', X.shape[1])
print('Reduced number of features:', X_lda.shape[1])
print('Erklärte Varianz:', lda.explained_variance_ratio_)

plt.figure()
X_mean=lda.transform(lda.means_)
plt.scatter(X_mean[:, 0], X_mean[:, 1],  s=150, c='black',cmap=plt.cm.rainbow, edgecolor='y')

for i in range(len(lda.classes_)):
    plt.scatter(X_lda[y_train == i, 0], X_lda[y_train == i, 1], alpha=.8)

plt.title('Streudiagramm der Gruppenzugehörigkeiten (ersten 2 Funktionen)')
plt.show()
```
Nun können wir mit dem Modell die Testdaten klassifizieren und sehen, dass die Testdaten zu 80% dem richtigen Cluster zugeordnet wurden (die Trainingsdaten zu 100%).
```
y_pred = lda.predict(X_test)

print('Score train data:', lda.score(X_train,y_train))
print('Score test data:', lda.score(X_test,y_test))
print('Label Testdata true:', y_test)
print('Label Testdata pred:', y_pred)
```
Mit einer Fehlerquote von 20% können wir nicht zufrieden sein. Offensichtlich wurde das Modell zu exakt auf die Trainingsdaten zugeschnitten. Im Machine Learning spricht man vom Overfitting, das Modell ist zu komplex. Wie können wir das Modell vereinfachen und damit für die Testdaten eine höhere Trefferquote erzielen? Dazu können wir die Anzahl der in die Diskriminanzanalyse einbezogenen Features reduzieren sowie die Anzahl der Diskriminanzfunktionen selbst.
Beginnen wir mit der Reduzierung von Features. Dazu verwenden wir die sklearn recursive feature elimination (RFE) mit einem Random-Forest-Klassifikator. Im RFE-Konstruktor geben wir an, dass eine Reduktion auf 38 Features erfolgen soll. Diesen Wert muss man durch probieren herausfinden. Dann nehmen wir unseren neuen Datensatz mit nur noch 38 Features, splitten ihn wiederum in einen Trainings- und Testdatensatz und führen eine neue Diskriminanzanalyse durch. Die Trainingsdaten werden nun schon zu 90% dem richtigen Cluster zugeordnet.
```
estimator = RandomForestClassifier(max_depth=2, random_state=8)
selector = RFE(estimator, 38, step=1)
selector.fit(X, y)
X_select = selector.transform(X)

X_select_train, X_select_test, y_select_train, y_select_test = train_test_split(X_select, y, test_size=0.2, random_state=8, stratify=y)

lda_select = LinearDiscriminantAnalysis()
lda_select.fit(X_select_train,y_select_train)
y_select_pred = lda_select.predict(X_select_test)

print('Score train data:', lda_select.score(X_select_train,y_select_train))
print('Score test data:', lda_select.score(X_select_test,y_select_test))
print('Label Testdata true:', y_select_test)
print('Label Testdata pred:', y_select_pred)
```
Sehen wir uns die zweite Optimierungsmöglichkeit an. Wie wir schon festgestellt haben, erklären die ersten 4 Funktionen bereits 95% der Varianz. Die letzte Funktion trägt also nur noch relativ wenig zur Trennung der Cluster bei. Wir können unser Modell also dadurch vereinfachen, indem wir nur die ersten 4 Diskriminazfunktionen aufnehmen. In sklearn lässt sich dies im Gegensatz zu SPSS aber nicht direkt steuern. Der Parameter n_components=4 im LinearDiscriminantAnalysis Konstruktor wirkt nämlich nur in der transform Methode jedoch nicht in der predict bzw. score Methode. Uns bleibt deshalb nur die Möglichkeit, dies durch die Hintertür zu realisieren. Allerdings ist das etwas doppelt gemoppelt. Dazu reduzieren wir die Daten mit den 55 Features auf 4 Diskriminanzfunktionen und trainieren mit diesen Funktionswerten unser Modell. Als Ergebnis erhalten wir eine Trefferquote von 80%. Das Modell hat also mit 4 Funktionen die gleiche Trefferquote wie mit allen 5 Funktionen.
```
lda_reduse = LinearDiscriminantAnalysis(n_components=4)
X_reduse_train = lda_reduse.fit_transform(X_train, y_train)
X_reduse_test = lda_reduse.transform(X_test)

lda_reduce = LinearDiscriminantAnalysis()
lda_reduce.fit(X_reduse_train, y_train)
y_reduce_pred = lda_reduce.predict(X_reduse_test)

print('Score train data:', lda_reduce.score(X_reduse_train,y_train))
print('Score test data:', lda_reduce.score(X_reduse_test,y_test))
print('Label Testdata true:', y_test)
print('Label Testdata pred:', y_reduce_pred)
```

## Support Vector Machine
Zum Abschluss wollen wir unser Klassifizierungsproblem noch mit einer Support Vector Machine lösen. Dabei kommen wir auf eine Trefferquote von 100% und das sowohl mit allen 55 Features als auch nach einer Reduzierung auf 38 Features. 
```
clf = LinearSVC()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print('Score train data:', clf.score(X_train,y_train))
print('Score test data:', clf.score(X_test,y_test))
print('Label Testdata true:', y_test)
print('Label Testdata pred:', y_pred)
```
Mit nur noch 38 Features.
```
clf = make_pipeline(SelectKBest(f_classif, k=38), LinearSVC())
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print('Score train data:', clf.score(X_train,y_train))
print('Score test data:', clf.score(X_test,y_test))
print('Label Testdata true:', y_test)
print('Label Testdata pred:', y_pred)
```
