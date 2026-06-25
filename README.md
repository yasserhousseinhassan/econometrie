# Analyse des Séries Temporelles & Modèle APT

Ce projet académique porte sur l'analyse des séries temporelles appliquées aux marchés financiers, en particulier sur l'étude de l'action Renault et de l'indice CAC 40, ainsi que sur la modélisation de la rentabilité d'un actif via la théorie d'arbitrage des prix (APT - *Arbitrage Pricing Theory*).

## Contenu du Projet

Le projet est divisé en deux grandes parties :

### 1. Modélisation et Analyse des Séries Temporelles (Action Renault vs CAC 40)
* **Période d'étude** : 2002 à 2024.
* **Statistiques descriptives** : Analyse de la distribution des cours de clôture et des rentabilités (Jarque-Bera, Skewness, Kurtosis).
* **Tests de Stationnarité** : Réalisation de tests de Dickey-Fuller Augmenté (ADF) sur les prix en niveau et sur les rentabilités.
* **Modélisation Économétrique** :
  * Estimation d'un modèle **ARMA(1,1)** pour modéliser la dynamique des rentabilités de Renault.
  * Modélisation de la volatilité conditionnelle via des modèles hétéroscédastiques : **GARCH(1,1)** et **GARCH-M(1,1)**.
  * Analyse des résidus (bruit blanc, autocorrélation via Durbin-Watson).
* **Test d'efficience informationnelle** : Test d'efficience au sens faible par l'étude de l'autocorrélation des rentabilités.

### 2. Estimation du Modèle APT (Arbitrage Pricing Theory)
* **Objectif** : Expliquer et prédire la rentabilité d'un actif financier en fonction de multiples facteurs macroéconomiques.
* **Période d'étude** : 12/1998 à 09/2022 (données mensuelles).
* **Facteurs explicatifs étudiés** :
  * Consommation des ménages
  * Indice des prix à la consommation (IPC / Inflation)
  * Euribor 3M (Taux d'intérêt à court terme)
  * Taux de chômage
  * Revenu Disponible Brut (RDB)
* **Méthodologie** : Estimation par la méthode des Moindres Carrés Ordinaires (MCO/GLM), analyse de significativité des variables, détection de colinéarité, et étude d'autocorrélation des résidus.

## Fichier du Projet
* `econometrie.pdf` : Le rapport d'étude complet contenant les méthodologies détaillées, les tableaux de résultats de régression (issus d'EViews), les graphiques d'analyse des résidus et les conclusions économiques.

---
*Projet réalisé par Yasser Houssein Hassan*
