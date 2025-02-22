# tabs/glossar.py
import streamlit as st

def display_tab():
    st.header("Glossar")
    
    glossary = {
                    "ADF-Test (Augmented Dickey-Fuller-Test)": """Ein statistischer Test, der in der Zeitreihenanalyse verwendet wird, um zu prüfen, ob eine Zeitreihe stationär ist. 
            Stationarität ist wichtig, weil viele Prognosemodelle stabile statistische Eigenschaften (Mittelwert, Varianz) voraussetzen. 
            Der ADF-Test hilft dabei, festzustellen, ob beispielsweise ein Trend oder eine stochastische Drift in den Daten vorliegt.""",

                    "Aggregation": """Das Zusammenfassen von Datenpunkten über bestimmte Zeitintervalle oder Kategorien hinweg. 
            So können Tagesdaten beispielsweise auf Wochen- oder Monatsebene aggregiert werden, um Muster besser sichtbar zu machen. 
            Aggregation vereinfacht oft die Datenstruktur, kann aber auch Informationen über kurzfristige Schwankungen verdecken.""",

                    "ARIMA": """ARIMA steht für „Autoregressive Integrated Moving Average“ und ist ein statistisches Modell zur Zeitreihenprognose. 
            Es kombiniert autoregressive Komponenten (AR), eine Integration zum Ausgleich von Nichtstationarität (I) und gleitende Durchschnitte (MA), um aus historischen Daten zukünftige Werte vorherzusagen.""",

                    "Autoregressiv": """Als „autoregressiv“ bezeichnet man Modelle, die zur Prognose zukünftiger Werte einer Zeitreihe ausschließlich auf vergangene Beobachtungen derselben Zeitreihe zurückgreifen. 
            Diese Modelle gehen davon aus, dass die eigene Historie der beste Indikator für die nahe Zukunft ist.""",

                    "Bayessche Modellierung": """Ein Ansatz in der Statistik, bei dem Unsicherheit durch Wahrscheinlichkeitsverteilungen modelliert wird. 
            Bayessche Modelle integrieren Vorwissen (Prior-Verteilung) mit beobachteten Daten (Likelihood), um posteriori Wahrscheinlichkeitsverteilungen für Parameter zu erhalten. 
            Diese Methoden sind besonders nützlich, wenn Daten knapp sind oder komplexe Zusammenhänge bestehen.""",

                    "Clustering": """Ein Verfahren des unüberwachten Lernens, bei dem ähnliche Datenpunkte in Gruppen (Cluster) eingeteilt werden, ohne zuvor Labels oder Klassen zu kennen. 
            In der Zeitreihenanalyse kann Clustering genutzt werden, um Produkte mit ähnlichen Absatzmustern zusammenzufassen oder um saisonale Mustergruppen zu identifizieren.""",

                    "Detrending": """Das Entfernen eines langfristigen Trends aus einer Zeitreihe, um andere Muster wie Saisonalität oder zyklische Effekte besser erkennen zu können. 
            Detrending sorgt dafür, dass sich das Modell auf Schwankungen um einen relativ konstanten Mittelwert konzentrieren kann.""",

                    "Exogene Variablen": """Exogene Variablen sind zusätzliche Einflussfaktoren, die nicht aus der Ziel-Zeitreihe selbst stammen. 
            Beispiele sind Wetterdaten, Konjunkturindikatoren oder Werbeausgaben. 
            Durch die Einbindung exogener Variablen in ein Prognosemodell kann dessen Genauigkeit und Erklärungsgehalt gesteigert werden.""",

                    "Feature Engineering": """Die gezielte Auswahl, Transformation und Erzeugung von Merkmalen (Features) aus Rohdaten, um ein Modell mit möglichst aussagekräftigen Eingangsvariablen zu versorgen. 
            Beim Umgang mit Zeitreihen können so zum Beispiel saisonale Indikatoren, gleitende Mittelwerte oder verzögerte Werte als zusätzliche Features genutzt werden.""",

                    "Fehlerminimierung": """Der Prozess, bei dem Prognosemodelle auf Daten trainiert werden, um den Unterschied (Fehler) zwischen den vorhergesagten und den tatsächlichen Werten zu verringern. 
            Dies geschieht meist durch Optimierungsverfahren, die Parameter so anpassen, dass Metriken wie der Mean Squared Error (MSE) oder der Mean Absolute Error (MAE) minimiert werden.""",

                    "Frequenz": """Die Frequenz einer Zeitreihe gibt an, in welchen Intervallen Beobachtungen vorliegen (z. B. täglich, wöchentlich, monatlich). 
            Die Frequenz bestimmt, welche saisonalen Muster in Frage kommen und wie die Daten vor der Modellierung gegebenenfalls umgeformt werden müssen.""",

                    "Glättende Verfahren": """Methoden, die kurzfristige Schwankungen in Zeitreihen glätten, um dauerhaftere Muster – wie Trends oder Saisonalitäten – klarer zu erkennen. 
            Beispiele sind gleitende Durchschnitte oder exponentielle Glättung, die Rauschen aus den Daten entfernen und dadurch die interpretierbaren Signale hervorheben.""",

                    "KPSS-Test (Kwiatkowski-Phillips-Schmidt-Shin-Test)": """Ein statistischer Test, um zu überprüfen, ob eine Zeitreihe stationär ist. 
            Im Gegensatz zum ADF-Test ist die Nullhypothese des KPSS-Tests, dass die Zeitreihe stationär ist. 
            Durch die Verwendung beider Tests kann man fundiertere Aussagen über die Stationarität treffen.""",

                    "Konfidenzintervalle": """Bereiche, innerhalb derer der wahre Wert mit einer bestimmten Wahrscheinlichkeit (Konfidenzniveau) liegt. 
            In der Zeitreihenprognose helfen Konfidenzintervalle dabei, die Unsicherheit in Vorhersagen zu quantifizieren. 
            So kann man abschätzen, wie zuverlässig ein prognostizierter Wert ist.""",

                    "Kointegration": """Ein Konzept in der Zeitreihenanalyse, das auftritt, wenn zwei oder mehr nichtstationäre Reihen so miteinander verknüpft sind, dass eine bestimmte Linearkombination von ihnen stationär ist. 
            Kointegration ist ein zentrales Merkmal, um langfristige Gleichgewichtsbeziehungen zwischen Variablen (z. B. Absatz und Preis) zu modellieren.""",

                    "Korrelation": """Ein statistisches Maß für den linearen Zusammenhang zwischen zwei Variablen. 
            In der Zeitreihenanalyse helfen Korrelationsmaße dabei, Beziehungen zwischen Absatz und anderen Faktoren (z. B. Wetter, Saisonalität, Werbeausgaben) zu erkennen.""",

                    "LSTM (Long Short-Term Memory)": """Eine spezielle Art von rekurrenten neuronalen Netzen, die darauf ausgelegt sind, Langzeitabhängigkeiten in Sequenzen zu erfassen. 
            Durch interne Speicher- und Gate-Mechanismen können LSTMs auch bei langen Zeitreihen relevante Muster behalten und für Prognosen nutzen.""",

                    "Maschinelles Lernen": """Ein Teilgebiet der künstlichen Intelligenz, bei dem Algorithmen aus Daten lernen und auf Basis dieser Erfahrung Vorhersagen oder Entscheidungen treffen. 
            In der Zeitreihenanalyse werden ML-Methoden eingesetzt, um komplexe Zusammenhänge zu modellieren, die klassische statistische Modelle oft überfordern.""",

                    "Metriken (MSE, RMSE, MAE)": """Gängige Gütemaße, um die Leistung von Prognosemodellen zu bewerten. 
            - MSE (Mean Squared Error): Mittlerer quadratischer Fehler, bei dem größere Abweichungen stärker gewichtet werden.  
            - RMSE (Root Mean Squared Error): Die Quadratwurzel des MSE, gut interpretierbar in Originaleinheiten.  
            - MAE (Mean Absolute Error): Mittlerer absoluter Fehler, misst durchschnittliche Abweichungen ohne sie zu quadrieren.""",

                    "Normalisierung/Standardisierung": """Verfahren, um Daten auf vergleichbare Größenordnungen zu bringen. 
            Bei der Normalisierung werden Werte häufig auf einen Bereich (z. B. 0 bis 1) skaliert, während bei der Standardisierung Daten auf einen Mittelwert von 0 und eine Standardabweichung von 1 transformiert werden. 
            Dies erleichtert das Training von Prognosemodellen, da extreme Werte weniger stark dominieren.""",

                    "Out-of-Sample": """Bezieht sich auf Daten, die nicht zur Modellentwicklung (Training) genutzt wurden. 
            Out-of-Sample-Tests überprüfen, wie gut ein Modell auf neuen, unbekannten Daten funktioniert. 
            Sie sind wichtig, um Überanpassung (Overfitting) zu vermeiden und die Generalisierbarkeit eines Modells zu beurteilen.""",

                    "Prophet": """Ein von Meta (Facebook) entwickeltes Prognosemodell, das durch eine benutzerfreundliche Parametrisierung, robuste Umgangsweisen mit saisonalen, trendbedingten und holiday-basierten Einflüssen überzeugt. 
            Prophet kann ohne tiefgehende statistische Kenntnisse angewendet werden und liefert bei vielen Datensätzen verlässliche Vorhersagen.""",

                    "SARIMA": """Seasonal ARIMA (SARIMA) ist eine Erweiterung des ARIMA-Modells, um saisonale Effekte zu berücksichtigen. 
            Es enthält zusätzliche Parameter, um periodische Muster (z. B. monatliche oder jährliche Zyklen) explizit in die Modellierung einzubinden, was zu genaueren Prognosen bei stark saisonalen Daten führt.""",

                    "Saisonalität": """Regelmäßige, sich wiederholende Muster in den Daten, z. B. ein Absatzhoch im Dezember jedes Jahres oder erhöhte Verkaufszahlen an Wochenenden. 
            Die Identifikation von saisonalen Mustern ist entscheidend, um verlässliche Prognosen erstellen zu können, insbesondere bei Produkten oder Märkten mit stark schwankender Nachfrage.""",

                    "Stationarität": """Eigenschaft einer Zeitreihe, bei der statistische Kennwerte wie Mittelwert und Varianz über die Zeit konstant bleiben. 
            Viele Prognosemodelle setzen Stationarität voraus, da sie nur dann sinnvolle Schätzungen liefern. 
            Nichtstationäre Daten müssen häufig transformiert (z. B. differenziert) werden, um stationär zu werden.""",

                    "STL-Zerlegung": """STL (Seasonal and Trend decomposition using Loess) ist ein Verfahren, um eine Zeitreihe in ihre Komponenten Trend, Saisonalität und Residuen zu zerlegen. 
            Dies erleichtert es, die strukturellen Bestandteile der Daten besser zu verstehen und dient oft als Vorverarbeitungsschritt für Prognosemodelle.""",

                    "Stichprobe": """Eine Teilmenge aus einer Grundgesamtheit, die zur statistischen Analyse herangezogen wird. 
            In der Zeitreihenanalyse bezeichnet man damit oft einen Datenausschnitt, der für Modellierung oder Testzwecke verwendet wird. 
            Wichtig ist, dass die Stichprobe repräsentativ für das Gesamtbild ist.""",

                    "Test- und Trainingsdaten": """Datensplits, um die Leistung eines Modells robust zu beurteilen. 
            Die Trainingsdaten werden zum Anpassen der Modellparameter genutzt, die Testdaten dienen im Anschluss zur unabhängigen Überprüfung der Vorhersagegüte. 
            So wird sichergestellt, dass das Modell nicht nur bekannte Muster wiedergibt, sondern auch bei neuen Daten performt.""",

                    "Theil's U": """Ein Maß zur Bewertung der Prognosequalität, bei dem die Genauigkeit eines Modells mit einer einfachen Referenzprognose verglichen wird. 
            Ein Theil’s U-Wert kleiner als 1 zeigt, dass das Modell besser als die Referenz ist, während ein Wert über 1 auf eine schlechtere Leistung hindeutet.""",

                    "Trend": """Ein längerfristiges Muster in einer Zeitreihe, bei dem der Mittelwert der Daten im Laufe der Zeit steigt oder fällt. 
            Trends sind wichtig, um zukünftige Entwicklungen abzuschätzen, beispielsweise ob die Nachfrage langfristig zunimmt oder allmählich abnimmt.""",

                    "Residual": """Der Fehlerterm, der nach der Modellierung einer Zeitreihe übrig bleibt, also die Differenz zwischen vorhergesagten und tatsächlichen Werten. 
            Die Analyse der Residuen zeigt, ob ein Modell wichtige Muster verpasst hat oder ob es noch systematische Abweichungen gibt, die verbessert werden könnten.""",

                    "XGBoost": """Ein leistungsfähiges Gradient Boosting Framework, das oft in Machine-Learning-Wettbewerben verwendet wird. 
            XGBoost kann auch für Zeitreihenprognosen eingesetzt werden, indem verzögerte Werte oder aggregierte Merkmale genutzt werden. 
            Es zeichnet sich durch hohe Geschwindigkeit und oft sehr gute Prognosegenauigkeit aus.""",

                    "Zeitreihe": """Eine Folge von Messungen oder Werten, die in zeitlicher Abfolge erhoben wurden. 
            Beispielsweise tägliche Verkaufszahlen, monatliche Besucherzahlen oder jährliche Umsätze. 
            Die Analyse von Zeitreihen zielt darauf ab, Muster zu erkennen, um zukünftige Werte möglichst zuverlässig vorherzusagen."""
    }

    # Display glossary terms in alphabetical order
    for term in sorted(glossary.keys()):
        st.subheader(term)
        st.write(glossary[term])
        st.markdown("---")