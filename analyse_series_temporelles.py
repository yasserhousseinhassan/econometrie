# -*- coding: utf-8 -*-
"""
============================================================================
Analyse des Séries Temporelles Financières & Modèle APT
============================================================================

Étude économétrique de la dynamique des rentabilités de l'action Renault
et de l'indice CAC 40, et estimation d'un modèle multifactoriel d'évaluation
des actifs (Arbitrage Pricing Theory, Ross 1976).

Méthodes implémentées :
    1. Statistiques descriptives et test de normalité de Jarque-Bera.
    2. Test de stationnarité de Dickey-Fuller Augmenté (ADF).
    3. Modélisation de la moyenne conditionnelle : ARMA(1,1).
    4. Modélisation de la variance conditionnelle : GARCH(1,1) et GARCH-M.
    5. Test d'efficience informationnelle au sens faible (autocorrélation).
    6. Estimation du modèle APT par Moindres Carrés Ordinaires (MCO).

Auteur : Yasser Houssein Hassan
Dépendances : numpy, pandas, scipy, statsmodels, arch, matplotlib
============================================================================
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor

try:
    from arch import arch_model
    _HAS_ARCH = True
except ImportError:  # le module arch est optionnel
    _HAS_ARCH = False


# ---------------------------------------------------------------------------
# 1. Calcul des rentabilités logarithmiques
# ---------------------------------------------------------------------------
def rentabilites_log(prix: pd.Series) -> pd.Series:
    r"""Rentabilité géométrique (log-return) : r_t = ln(P_t / P_{t-1}).

    Préférée à la rentabilité arithmétique car additive dans le temps et
    approximativement normale pour des pas de temps courts.
    """
    return np.log(prix / prix.shift(1)).dropna()


# ---------------------------------------------------------------------------
# 2. Statistiques descriptives et normalité
# ---------------------------------------------------------------------------
def statistiques_descriptives(serie: pd.Series) -> dict:
    r"""Moments empiriques et test de normalité de Jarque-Bera.

    JB = (n/6) * (S^2 + (K-3)^2 / 4)  ~  chi2(2) sous H0 de normalité,
    où S est le coefficient d'asymétrie (skewness) et K l'aplatissement
    (kurtosis non centré).
    """
    x = serie.dropna().values
    jb_stat, jb_p = stats.jarque_bera(x)
    return {
        "n": len(x),
        "moyenne": np.mean(x),
        "ecart_type": np.std(x, ddof=1),
        "skewness": stats.skew(x),
        "kurtosis": stats.kurtosis(x, fisher=False),  # kurtosis non centré (3 = normale)
        "JB_stat": jb_stat,
        "JB_pvalue": jb_p,
        "normale_au_seuil_5pct": jb_p > 0.05,
    }


# ---------------------------------------------------------------------------
# 3. Test de stationnarité (Dickey-Fuller Augmenté)
# ---------------------------------------------------------------------------
def test_adf(serie: pd.Series, regression: str = "c") -> dict:
    r"""Test ADF de racine unitaire.

    Modèle :  Δy_t = α + β t + γ y_{t-1} + Σ φ_i Δy_{t-i} + ε_t
    H0 : γ = 0  (présence d'une racine unitaire => série non stationnaire).
    On rejette H0 (série stationnaire) si la p-value < 0.05.
    """
    stat, pvalue, lags, nobs, crit, _ = adfuller(serie.dropna(), regression=regression)
    return {
        "ADF_stat": stat,
        "pvalue": pvalue,
        "lags_retenus": lags,
        "valeurs_critiques": crit,
        "stationnaire_au_seuil_5pct": pvalue < 0.05,
    }


# ---------------------------------------------------------------------------
# 4. Modèle ARMA(p, q) pour la moyenne conditionnelle
# ---------------------------------------------------------------------------
def estimer_arma(serie: pd.Series, p: int = 1, q: int = 1):
    r"""Estimation d'un ARMA(p, q) par maximum de vraisemblance.

    r_t = c + Σ φ_i r_{t-i} + Σ θ_j ε_{t-j} + ε_t
    """
    modele = ARIMA(serie.dropna(), order=(p, 0, q))
    resultat = modele.fit()
    dw = durbin_watson(resultat.resid)  # ~2 => absence d'autocorrélation d'ordre 1
    return resultat, dw


# ---------------------------------------------------------------------------
# 5. Modèle GARCH pour la variance conditionnelle
# ---------------------------------------------------------------------------
def estimer_garch(serie: pd.Series, p: int = 1, q: int = 1, in_mean: bool = False):
    r"""GARCH(1,1) :  σ²_t = ω + α ε²_{t-1} + β σ²_{t-1}.

    Si in_mean=True, on estime un GARCH-M où la prime de risque entre dans
    l'équation de moyenne : r_t = μ + δ σ_t + ε_t.
    Stationnarité de la variance garantie si α + β < 1.
    """
    if not _HAS_ARCH:
        raise ImportError("Le module 'arch' est requis : pip install arch")
    r = serie.dropna() * 100.0  # mise à l'échelle pour la stabilité numérique
    modele = arch_model(r, mean="ARX" if in_mean else "Constant",
                        vol="GARCH", p=p, q=q,
                        rescale=False)
    return modele.fit(disp="off")


# ---------------------------------------------------------------------------
# 6. Test d'efficience au sens faible (autocorrélations des rentabilités)
# ---------------------------------------------------------------------------
def test_efficience_faible(serie: pd.Series, nlags: int = 10) -> pd.DataFrame:
    r"""Sous l'hypothèse d'efficience informationnelle faible, les
    rentabilités ne sont pas auto-corrélées (marche aléatoire).

    Bande de confiance approximative à 95 % : ±1.96 / sqrt(n).
    """
    r = serie.dropna()
    rho = acf(r, nlags=nlags, fft=True)[1:]
    borne = 1.96 / np.sqrt(len(r))
    return pd.DataFrame({
        "lag": np.arange(1, nlags + 1),
        "autocorrelation": rho,
        "significative_5pct": np.abs(rho) > borne,
    })


# ---------------------------------------------------------------------------
# 7. Modèle APT (Arbitrage Pricing Theory) par MCO
# ---------------------------------------------------------------------------
def estimer_apt(rentabilite: pd.Series, facteurs: pd.DataFrame):
    r"""Modèle multifactoriel :  R_t = α + Σ β_k F_{k,t} + ε_t.

    Facteurs macroéconomiques typiques : consommation des ménages, IPC
    (inflation), Euribor 3M, taux de chômage, revenu disponible brut (RDB).
    Retourne l'objet de régression et les VIF pour diagnostiquer la
    multicolinéarité (VIF > 10 => colinéarité problématique).
    """
    X = sm.add_constant(facteurs)
    modele = sm.OLS(rentabilite, X, missing="drop").fit()

    vif = pd.DataFrame({
        "variable": X.columns,
        "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
    })
    dw = durbin_watson(modele.resid)
    return modele, vif, dw


# ---------------------------------------------------------------------------
# Démonstration sur données simulées (reproductible)
# ---------------------------------------------------------------------------
def _demonstration():
    rng = np.random.default_rng(42)
    n = 500

    # --- Simulation d'un processus de prix avec volatilité variable ---
    eps = rng.standard_normal(n) * 0.02
    prix = 30.0 * np.exp(np.cumsum(eps))
    prix = pd.Series(prix, name="Renault")
    r = rentabilites_log(prix)

    print("=" * 70)
    print("1. STATISTIQUES DESCRIPTIVES DES RENTABILITÉS")
    print("=" * 70)
    for k, v in statistiques_descriptives(r).items():
        print(f"  {k:<28}: {v}")

    print("\n" + "=" * 70)
    print("2. TEST DE STATIONNARITÉ (ADF)")
    print("=" * 70)
    print("  Prix en niveau   :", test_adf(prix))
    print("  Rentabilités     :", test_adf(r))

    print("\n" + "=" * 70)
    print("3. MODÈLE ARMA(1,1)")
    print("=" * 70)
    res_arma, dw = estimer_arma(r, 1, 1)
    print(res_arma.summary().tables[1])
    print(f"  Statistique de Durbin-Watson : {dw:.4f}")

    print("\n" + "=" * 70)
    print("4. TEST D'EFFICIENCE AU SENS FAIBLE")
    print("=" * 70)
    print(test_efficience_faible(r).to_string(index=False))

    print("\n" + "=" * 70)
    print("5. MODÈLE APT (données macroéconomiques simulées)")
    print("=" * 70)
    facteurs = pd.DataFrame({
        "consommation": rng.standard_normal(len(r)),
        "inflation": rng.standard_normal(len(r)),
        "euribor_3m": rng.standard_normal(len(r)),
        "chomage": rng.standard_normal(len(r)),
        "rdb": rng.standard_normal(len(r)),
    }, index=r.index)
    modele_apt, vif, dw_apt = estimer_apt(r, facteurs)
    print(modele_apt.summary().tables[1])
    print(f"\n  R² = {modele_apt.rsquared:.4f} | Durbin-Watson = {dw_apt:.4f}")
    print("\n  Facteurs d'inflation de la variance (VIF) :")
    print(vif.to_string(index=False))


if __name__ == "__main__":
    _demonstration()
