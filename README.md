# Dispersion Trading -- Strategies theta/gamma/vega-neutres sur SPY/AAPL

Projet Python realise dans le cadre du cours de Volatility Trading du Master 272 -- Ingenierie Economique et Financiere, Universite Paris-Dauphine (PSL).

Forke depuis [BaptisteZloch/Dauphine-Lecture-Volatility](https://github.com/BaptisteZloch/Dauphine-Lecture-Volatility).

---

## Objet

Ce depot implemente et backteste une strategie de dispersion longue a deux jambes :

- **Jambe indice** : vente d'un straddle ATM 1 mois sur SPY (carry leg, short vol).
- **Jambe composant** : achat d'un straddle ATM 1 mois sur AAPL (dispersion leg, long vol).
- **Couverture delta** : independante sur chaque sous-jacent.
- **Sizing grec** : le notionnel de la jambe AAPL est ajuste pour egaliser celui de la jambe SPY en termes de theta, gamma-dollar ou vega.

Le backtest couvre la periode janvier 2020 -- decembre 2022 sur des donnees d'options quotidiennes (parquet).

## Ce que ce projet n'est pas

Ce depot n'est **pas** un moteur complet de correlation implicite indice contre panier. Il ne traite qu'un composant face a l'indice ; les sections de "correlation" doivent donc etre lues comme un proxy single-name de prime de correlation, et non comme une mesure de correlation implicite du panier.

## Contenu du notebook

Le livrable principal est :

```
notebooks/Dispersion_Backtest_ Project.ipynb
```

Il couvre les etapes suivantes :

1. Backtests statiques en theta-neutre, gamma-dollar-neutre et vega-neutre.
2. Verifications de neutralite grecque (theta, gamma, vega) et de couverture delta.
3. Diagnostics sur les periodes de stress (COVID, fin 2022).
4. Construction et analyse du proxy de correlation implied-minus-realized (single-name).
5. Allocation dynamique en walk-forward (expanding z-score, pas de look-ahead).
6. Tests de robustesse (sensibilite aux couts de transaction, au seuil de signal, a la fenetre de z-score) et test de permutation.

La conclusion empirique principale est que la surcouche de timing n'est economiquement convaincante que pour la variante theta-neutre, et seulement lorsqu'elle est evaluee en walk-forward.

## Structure du depot

```text
.
├── data/
│   ├── aapl_2016_2023.parquet
│   ├── optiondb_2016_2023.parquet/
│   ├── par-yield-curve-rates-2020-2023.csv
│   ├── spy_2020_2022.parquet/
│   ├── spy_2020_2022_atm.parquet
│   └── spy_2020_2022_dte90.parquet
├── lectures/
│   ├── Lecture_2.ipynb
│   ├── Lecture_3.ipynb
│   ├── Lecture_4.ipynb
│   └── Lecture_5.ipynb
├── notebooks/
│   └── Dispersion_Backtest_ Project.ipynb
├── src/
│   ├── __init__.py
│   ├── constants.py
│   ├── rates.py
│   ├── specs.py
│   ├── util.py
│   ├── data/
│   │   ├── data_loader.py          # Chargement generique des donnees parquet
│   │   ├── option_db.py            # Loaders SPY et AAPL
│   │   └── rates_db.py             # Loader courbe de taux US
│   ├── dispersion/
│   │   ├── dispersion_trade.py     # Orchestrateur du trade de dispersion
│   │   ├── greek_sizing.py         # Sizing theta/gamma-dollar/vega-neutre
│   │   ├── dynamic_allocation.py   # Signaux de timing et overlay dynamique
│   │   ├── robustness.py           # Sensibilite (tcost, seuil, fenetre) et permutation
│   │   └── signal_analysis.py      # Analyse par buckets, correlation forward, perf conditionnelle
│   ├── metrics/
│   │   ├── distance.py
│   │   ├── performance.py          # Sharpe, drawdown, Calmar, rendements annualises
│   │   ├── util.py
│   │   └── volatility.py           # Volatilite realisee
│   ├── pricing/
│   │   ├── black_scholes.py        # Pricing et grecs Black-Scholes
│   │   └── implied_volatility.py   # Inversion de vol implicite
│   ├── stochastic/
│   │   ├── base.py
│   │   └── geometric_brownian_motion.py
│   ├── surface/
│   │   ├── base.py
│   │   ├── sabr.py
│   │   ├── ssvi.py
│   │   └── svi.py
│   └── trading/
│       ├── backtest.py             # Moteur de backtest (NAV, PnL, positions driftees)
│       ├── option_trade.py         # Objets de trade et delta-hedging
│       ├── selection.py            # Selection d'options (strike, DTE)
│       └── strategies.py           # Definition des strategies (straddle, etc.)
├── test_bt.py                      # Smoke test CLI (3 variantes, janv.--juin 2021)
├── pyproject.toml
├── requirements.txt
└── LICENSE
```

## Installation

Prerequis : Python >= 3.10.

```bash
git clone https://github.com/Mariuscalaque/Volatility-Trading-Project--Dispersion-Strategies.git
cd Volatility-Trading-Project--Dispersion-Strategies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`pywin32` est declare comme dependance conditionnelle (`sys_platform == "win32"`), ce qui permet au meme fichier requirements de fonctionner sous Linux et dans Codespaces.

## Smoke test

Pour verifier que l'ensemble de la chaine fonctionne en dehors de Jupyter :

```bash
python test_bt.py
```

Le script lance les trois variantes (theta, gamma, vega) sur un echantillon reduit (janvier--juin 2021) et verifie que la NAV, le PnL et les positions driftees sont bien produits.

## Stack technique

| Categorie | Librairies |
|---|---|
| Calcul scientifique | numpy, scipy, pandas |
| Pricing et grecs | Black-Scholes maison (`src/pricing/`) |
| Surfaces de vol | SVI, SSVI, SABR (`src/surface/`) |
| Statistiques et ML | statsmodels, scikit-learn |
| Visualisation | matplotlib, seaborn |
| Donnees | pyarrow (parquet) |

## Licence

Voir le fichier [LICENSE](LICENSE).
