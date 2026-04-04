# Dispersion Trading — SPY/AAPL θ/Γ/ν-neutre

Projet Python consacré à une stratégie de dispersion longue construite de la façon suivante :
- vente d’un straddle ATM 1 mois sur SPY,
- achat d’un straddle ATM 1 mois sur AAPL,
- couverture delta indépendante sur chaque jambe,
- ajustement du notionnel grec pour obtenir une neutralité en θ, Γ ou ν.

## Ce qu’est ce projet

Ce dépôt étudie l’économie d’un trade de dispersion via une implémentation pratique SPY/AAPL. Le backtest est financièrement cohérent comme stratégie short vol indice contre long vol single-name, et le notebook l’étend avec une analyse de stress, des diagnostics glissants et une surcouche d’allocation dynamique en walk-forward.

## Ce que ce projet n’est pas

Ce dépôt n’est **pas** un moteur complet de corrélation implicite indice contre panier. Le repo ne traite qu’un composant face à l’indice ; les sections de « corrélation » doivent donc être lues comme un **proxy single-name SPY/AAPL de prime de risque de corrélation**, et non comme la véritable corrélation implicite moyenne du panier du S&P 500.

## Résumé de la stratégie

- **Jambe indice :** vente d’un straddle ATM SPY.
- **Jambe composant :** achat d’un straddle ATM AAPL.
- **Couverture :** chaque jambe est couverte séparément en delta.
- **Sizing :** la jambe AAPL est redimensionnée pour égaliser la jambe SPY en termes de notionnel grec.
- **Variantes de sizing :** theta-neutre, gamma-dollar-neutre et vega-neutre.

Dans le notebook nettoyé, la principale conclusion empirique est que la surcouche de timing n’est économiquement convaincante que pour la variante theta-neutre, et seulement lorsqu’elle est évaluée dans un cadre walk-forward.

## Structure du dépôt

```text
.
├── data/           # Fichiers parquet de taux et d’options
├── notebooks/      # Notebook principal et copies locales
├── lectures/       # Notebooks de support du cours
├── src/data/       # Chargeurs de données
├── src/dispersion/ # Construction du trade, sizing, robustesse, timing
├── src/metrics/    # Mesures de performance et de volatilité
├── src/pricing/    # Utilitaires Black-Scholes et volatilité implicite
└── src/trading/    # Objets de trade et moteur de backtest
```

## Mise en place de l’environnement

Créer un environnement virtuel, puis installer les dépendances du projet :

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

`pywin32` est conservé comme dépendance Windows uniquement, ce qui permet au même fichier de dépendances de s’installer sous Linux et dans Codespaces.

## Livrable principal

Le notebook principal du projet est :

```text
notebooks/Dispersion_Backtest_ Project.ipynb
```

Il contient :
- les backtests statiques en θ/Γ/ν-neutre,
- les vérifications de neutralité grecque et delta,
- les diagnostics sur les périodes de stress,
- l’analyse du proxy de corrélation single-name,
- l’allocation dynamique en walk-forward,
- les tests de robustesse et le test de permutation.

## Smoke test

Pour exécuter un smoke test CLI léger en dehors de Jupyter :

```bash
.venv/bin/python test_bt.py
```

Le script lance les trois variantes sur un échantillon plus court et vérifie que la NAV, le PnL et les positions driftées sont bien produits.
