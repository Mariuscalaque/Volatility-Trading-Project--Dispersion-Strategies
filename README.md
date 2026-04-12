# Volatility Trading Project -- Stratégies de Dispersion

Projet de recherche et de backtesting autour du trading de volatilité, centré sur les stratégies de dispersion entre un indice (SPY) et l'un de ses composants (AAPL). L'ensemble du code est écrit en Python et s'appuie sur des données d'options réelles couvrant la période 2016-2023.

---

## Contexte et objectif

Le dispersion trading consiste à exploiter l'écart entre la volatilité implicite d'un indice et celle de ses composants. En pratique, on vend de la volatilité sur l'indice (ici SPY, via un straddle court) et on achète de la volatilité sur un ou plusieurs composants (ici AAPL, via un straddle long). Le profit provient du fait que la volatilité implicite de l'indice intègre une prime de corrélation : elle surestime généralement la volatilité réalisée du panier par rapport à la somme pondérée des volatilités individuelles.

Ce projet implémente trois variantes de neutralisation grecque pour dimensionner les jambes du trade :

- **Theta-neutre** : le theta net du portefeuille est ramené à zéro chaque jour de rebalancement.
- **Gamma-neutre** : le gamma net est neutralisé, ce qui isole l'exposition au vega et au passage du temps.
- **Vega-neutre** : le vega net est neutralisé, ce qui expose le portefeuille principalement au gamma et au theta.

Le backtest couvre la période janvier 2021 -- juin 2021 par défaut, mais les données permettent de tester sur une plage plus large (2020-2022 pour SPY, 2016-2023 pour AAPL).

---

## Structure du projet

```
.
├── data/                          # Données brutes (options, taux, spots)
│   ├── aapl_2016_2023.parquet         # Chaîne d'options AAPL, 2016-2023
│   ├── optiondb_2016_2023.parquet     # Base d'options générale, 2016-2023
│   ├── spy_2020_2022.parquet          # Chaîne d'options SPY, 2020-2022
│   ├── spy_2020_2022_atm.parquet      # Options SPY filtrées ATM, 2020-2022
│   ├── spy_2020_2022_dte90.parquet    # Options SPY avec DTE <= 90j, 2020-2022
│   └── par-yield-curve-rates-2020-2023.csv  # Courbe de taux US Treasury
│
├── notebooks/
│   └── Dispersion_Backtest_Project.ipynb  # Notebook principal du backtest
│
├── lectures/                      # Notebooks de cours et d'exploration
│   ├── Lecture_2.ipynb
│   ├── Lecture_3.ipynb
│   ├── Lecture_4.ipynb
│   └── Lecture_5.ipynb
│
├── src/                           # Code source principal
│   ├── __init__.py
│   ├── constants.py               # Constantes partagées (jours de trading, tenors)
│   ├── rates.py                   # Interpolation de la courbe de taux et calcul du forward
│   ├── specs.py                   # TypedDicts pour les spécifications de jambes
│   ├── util.py                    # Utilitaires (forward-fill, validations)
│   │
│   ├── data/                      # Chargement et prétraitement des données
│   │   ├── data_loader.py             # Classe abstraite DataLoader
│   │   ├── option_db.py               # Loaders concrets : OptionLoader, SPYOptionLoader, AAPLOptionLoader
│   │   └── rates_db.py                # Loader pour la courbe de taux
│   │
│   ├── pricing/                   # Modèles de pricing
│   │   ├── black_scholes.py           # Prix Black-Scholes et grecs (delta, gamma, vega, theta)
│   │   └── implied_volatility.py      # Calcul de la volatilité implicite par Newton-Raphson
│   │
│   ├── surface/                   # Modèles de surface de volatilité
│   │   ├── base.py                    # Classe abstraite VolSmoother (fit / transform)
│   │   ├── svi.py                     # Modèle SVI (Stochastic Volatility Inspired)
│   │   ├── ssvi.py                    # Modèle SSVI (Surface SVI avec noyau power-law)
│   │   └── sabr.py                    # Modèle SABR
│   │
│   ├── stochastic/                # Processus stochastiques
│   │   ├── base.py
│   │   └── geometric_brownian_motion.py  # Mouvement brownien géométrique
│   │
│   ├── metrics/                   # Métriques de performance et de volatilité
│   │   ├── performance.py             # Sharpe, Sortino, Calmar, drawdown, hit rate, etc.
│   │   ├── volatility.py             # Volatilité réalisée (ponctuelle et glissante)
│   │   ├── distance.py               # MSE pour la calibration des surfaces
│   │   └── util.py                   # Conversions niveaux <-> rendements
│   │
│   ├── dispersion/                # Logique du trade de dispersion
│   │   ├── greek_sizing.py            # Dimensionnement grec-neutre (theta, gamma, vega)
│   │   ├── dispersion_trade.py        # Orchestration du backtest de dispersion
│   │   ├── dynamic_allocation.py      # Allocation dynamique (signaux VRP, corrélation)
│   │   ├── signal_analysis.py         # Analyse conditionnelle des signaux
│   │   └── robustness.py             # Tests de sensibilité (coûts de transaction, seuils, fenêtres)
│   │
│   └── trading/                   # Infrastructure de trading et backtest
│       ├── option_trade.py            # Classe abstraite OptionTradeABC (chargement, prétraitement)
│       ├── selection.py               # Sélection d'options (maturité et strike les plus proches)
│       ├── strategies.py              # Définitions de stratégies (calendar spreads, reverse spreads)
│       └── backtest.py                # Moteur de backtest (PnL, NAV, coûts de transaction)
│
├── test_bt.py                     # Script de validation du backtest (3 variantes grecques)
├── pyproject.toml                 # Configuration du package Python
├── requirements.txt               # Dépendances
└── LICENSE                        # Licence du projet
```

---

## Fonctionnement detaillé des modules

### data/ -- Chargement des données

Le module `src/data/` fournit une classe abstraite `DataLoader` qui gère la lecture de fichiers Parquet, CSV ou Excel, le filtrage par plage de dates, et l'ajout de champs dérivés. Trois classes concrètes en héritent :

- `OptionLoader` : charge la base d'options générale (`optiondb_2016_2023.parquet`), filtre par ticker, remplace les prix à l'expiration par le payoff réel (intrinsic value), et calcule le nombre de jours avant expiration ainsi que la moneyness.
- `SPYOptionLoader` : spécialisé pour les options SPY (fichier `spy_2020_2022_dte90.parquet`, maturités jusqu'à 90 jours).
- `AAPLOptionLoader` : spécialisé pour les options AAPL (fichier `aapl_2016_2023.parquet`).

La courbe de taux est chargée via `rates_db.py` à partir du fichier CSV des taux du Trésor américain (par yield curve rates, 2020-2023).

### pricing/ -- Black-Scholes et volatilité implicite

- `black_scholes.py` contient le pricing analytique Black-Scholes (calls et puts) ainsi que le calcul des grecs : delta, gamma, vega et theta.
- `implied_volatility.py` implémente le calcul de la volatilité implicite par la méthode de Newton-Raphson, vectorisée sur des Series pandas. La convergence est contrôlée par une tolérance de 1e-7 et un maximum de 10 000 itérations.

### surface/ -- Modèles de surface de volatilité

Trois modèles de lissage de la surface de volatilité sont disponibles, tous héritant de la classe abstraite `VolSmoother` qui impose une interface `fit` / `transform` / `fit_transform` :

- **SVI** (Stochastic Volatility Inspired) : paramétrisé par (a, b, rho, m, sigma). Calibré par minimisation du MSE avec contrainte d'absence d'arbitrage (variance totale positive).
- **SSVI** (Surface SVI) : paramétrisé par (sigma, rho, eta, lambda) avec un noyau power-law. La contrainte lambda <= 0.5 garantit l'absence d'arbitrage papillon.
- **SABR** : paramétrisé par (alpha, beta, rho, nu). Utilise l'approximation de Hagan pour la volatilité implicite. Gère le cas ATM (forward = strike) séparément.

La calibration utilise l'optimiseur L-BFGS-B de SciPy avec des bornes sur chaque paramètre.

### rates.py -- Taux et prix forward

Le module calcule le taux sans risque interpolé pour chaque couple (date, expiration) en interpolant linéairement la courbe de taux du Trésor. L'extrapolation est plate aux bornes. Le prix forward est ensuite déduit par la formule classique : `F = S * exp(r * T)`.

### dispersion/ -- Logique du trade de dispersion

C'est le module central du projet.

**Dimensionnement grec-neutre (`greek_sizing.py`)** : pour chaque jour de trading, le ratio de sizing est calculé comme le rapport entre l'exposition grecque absolue de la jambe indice (SPY, short) et l'exposition grecque unitaire de la jambe composant (AAPL, long). Ce ratio est appliqué au poids de la jambe composant pour que l'exposition nette au grec choisi (theta, gamma ou vega) soit nulle. Un mécanisme de cap (par défaut le 95e percentile de la distribution du ratio) empêche les positions extrêmes lorsque le grec du dénominateur est proche de zéro, typiquement autour des expirations.

**Allocation dynamique (`dynamic_allocation.py`)** : le module permet de moduler l'exposition du trade en fonction de signaux de marché. Deux signaux sont implémentés :
- Le spread de prime de risque de variance (VRP) : écart entre la volatilité implicite et la volatilité réalisée.
- Le spread de corrélation : écart entre la corrélation implicite et la corrélation réalisée.

Ces signaux sont transformés en z-scores glissants, puis convertis en expositions via des seuils configurables.

**Analyse des signaux (`signal_analysis.py`)** : fournit des outils pour évaluer la qualité des signaux par bucketing, performance conditionnelle, et corrélation avec le PnL futur.

**Tests de robustesse (`robustness.py`)** : le module permet de tester la sensibilité de la stratégie aux coûts de transaction, aux seuils des signaux, et à la taille de la fenêtre glissante.

### trading/ -- Infrastructure de backtest

**Sélection d'options (`selection.py`)** : pour chaque jour et chaque ticker, le module sélectionne l'option la plus proche d'une maturité cible (en jours) et d'un strike cible (en moneyness, delta, ou strike absolu). La sélection se fait par minimisation de la distance absolue au sein de chaque groupe (date, ticker, expiration).

**Stratégies prédéfinies (`strategies.py`)** : le fichier définit plusieurs structures d'options sous forme de listes de `OptionLegSpec` :
- Calendar spreads (1 semaine / 1 mois, 1 mois / 6 mois) sur calls et puts ATM.
- Reverse calendar spreads (sens inversé).
- Chaque jambe spécifie la maturité cible, le type (call ou put), le poids signé, et le jour de rebalancement dans la semaine.

**Moteur de backtest (`backtest.py`)** : la classe `StrategyBacktester` prend en entrée un DataFrame de positions (avec colonnes date, poids, prix mid/bid/ask, ticker, etc.) et calcule :
- Le PnL quotidien (mark-to-market sur les variations de prix mid, pondéré par les positions).
- La NAV (valeur liquidative).
- Les positions driftées (positions après variation du prix sans rebalancement).
- Les coûts de transaction (demi-spread bid/ask appliqué aux changements de position).

Une sous-classe `BacktesterBidAskFromData` utilise directement les spreads bid/ask des données pour estimer les coûts de transaction.

### metrics/ -- Métriques de performance

Le module fournit un ensemble de métriques de performance standard :
- Rendement réalisé annualisé.
- Ratio de Sharpe, ratio de Sortino, ratio de Calmar.
- Drawdown et drawdown maximum.
- Hit rate (proportion de jours positifs).
- Information ratio et excess return par rapport à un benchmark.
- Ratio de Sharpe glissant.
- Volatilité réalisée annualisée (ponctuelle et glissante).

---

## Données

Les données d'options sont stockées au format Parquet et contiennent pour chaque contrat : date, ticker, expiration, strike, type (call/put), prix bid, ask et mid, volume, spot du sous-jacent, grecs (delta, gamma, vega, theta), et volatilité implicite.

La courbe de taux provient des données publiques du Trésor américain (par yield curve rates) et couvre les maturités de 1 mois à 30 ans.

Plages de disponibilité :
- SPY (options avec DTE <= 90 jours) : 2 janvier 2020 au 30 décembre 2022.
- AAPL : 2 janvier 2016 au 31 décembre 2023.
- Courbe de taux : 2020-2023.

---

## Installation

Python 3.10 ou supérieur est requis.

```bash
git clone https://github.com/Mariuscalaque/Volatility-Trading-Project--Dispersion-Strategies.git
cd Volatility-Trading-Project--Dispersion-Strategies
pip install -r requirements.txt
```

Les dépendances principales sont : numpy, pandas, scipy, matplotlib, seaborn, scikit-learn, statsmodels, pyarrow et tqdm.

---

## Utilisation

### Lancer le backtest de dispersion

Le script `test_bt.py` exécute le backtest sur les trois variantes grecques (theta, gamma, vega) entre le 4 janvier 2021 et le 30 juin 2021 :

```bash
python test_bt.py
```

Pour chaque variante, le script vérifie que la NAV est non vide, strictement positive, et que les positions contiennent bien les tickers SPY et AAPL. Il affiche le nombre d'observations, la NAV finale, et la plage de dates.

### Notebook principal

Le notebook `notebooks/Dispersion_Backtest_Project.ipynb` contient l'analyse complète : chargement des données, calibration des surfaces de volatilité, construction du trade de dispersion, backtest, analyse des signaux, et tests de robustesse.

---

## Licence

Ce projet est distribué sous licence (voir le fichier `LICENSE` à la racine du dépôt).
