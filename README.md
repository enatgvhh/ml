#

Machine Learning / Multivariate Statistik in Python
===================================================

## Inhalt
* [Einleitung](#einleitung)
* [Systemanalyse](ml_kap2.md)
* [Systemidentifikation](ml_kap3.md)
* [Systemsynthese](ml_kap4.md)
* [Summary](#summary)


## Einleitung
Anhand der Stadtteilprofile von Hamburg sollen hier experimentell die Möglichkeiten untersucht werden, die sich mit Machine Learning Packages in Python ergeben. Es geht hier weniger um das Ergebnis als um die methodische Vorgehensweise. Wir werden uns zunächst mit Unsupervised Learning und darauf aufbauend mit Supervised Learning beschäftigen.
Die Daten stehen über das [Transparenzportal Hamburg](http://suche.transparenz.hamburg.de/dataset/stadtteil-profile-hamburg4?forceWeb=true) frei zur Verfügung. Etwas aufbereitet und als [CSV-File](data/StadtteilprofileBerichtsjahr2018_org.csv) gespeichert, können wir nun starten.


## Summary
Wir haben uns in den vorhergehenden Kapiteln etwas mit Unsupervised Learning und Supervised Learning beschäftigt. Und es wurde deutlich, dass hier vor allem der Mensch erstmal lernen muss. Lernen welche Methoden, welche Parameter zu wählen sind, wie Entscheidungen zu treffen sind und wie die Ergebnisse sinnvoll interpretiert werden. Die Ergebnisse müssen grundsätzlich kritisch hinterfragt werden. Ganz nach dem Motto "Traue nur der Statistik, die du selbst gefälscht hast". Von daher ist der Begriff multivariate Analysemethoden/Statistik möglicherweise passender.