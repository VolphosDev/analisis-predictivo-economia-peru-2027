"""Análisis predictivo de las condiciones de la economía peruana para enero-junio de 2027.

Flujo: (1) descarga series oficiales del BCRP, (2) ajusta y valida modelos SARIMA,
(3) estima probabilidades con simulación Monte Carlo, (4) mide determinantes con OLS-HAC
y aprendizaje automático, (5) construye escenarios y (6) exporta tablas y figuras.

Uso:  python prediccion_economia_peru.py             (usa data/series_bcrp.csv si existe)
      python prediccion_economia_peru.py --descargar (vuelve a consultar la API del BCRP)
      python prediccion_economia_peru.py --rapido    (reutiliza órdenes y validación ya calculados)

La corrida completa tarda unos 25 minutos por la búsqueda en malla y la validación de origen móvil.
"""
import argparse
import itertools
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent
DATA, RES, FIG = BASE / "data", BASE / "resultados", BASE / "figuras"
for carpeta in (DATA, RES, FIG):
    carpeta.mkdir(exist_ok=True)
SEED = 2027
INICIO_H1, FIN_H1 = pd.Timestamp("2027-01-01"), pd.Timestamp("2027-06-01")

SERIES = {
    "pbi": "PN01770AM", "ipc": "PN38705PM", "infl_12m": "PN01273PM", "infl_sae": "PN01277PM",
    "tc": "PN01207PM", "cobre": "PN01652XM", "oro": "PN01654XM", "ti": "PN38923BM",
    "export": "PN38714BM", "exp_econ_12m": "PD37981AM", "exp_pbi_12m": "PD38048AM",
    "exp_infl_12m": "PD12912AM", "tasa_ref": "PD04722MM", "cemento": "PD37967GM",
}
MESES = dict(Ene=1, Feb=2, Mar=3, Abr=4, May=5, Jun=6, Jul=7, Ago=8, Set=9, Sep=9, Oct=10, Nov=11, Dic=12)


def descargar():
    """Consulta la API pública de series del BCRP y guarda una copia reproducible."""
    salida = {}
    for nombre, codigo in SERIES.items():
        url = f"https://estadisticas.bcrp.gob.pe/estadisticas/series/api/{codigo}/json/2004-1/2026-12/esp"
        js = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"}).json()
        valores = {}
        for periodo in js["periods"]:
            mes, anio = periodo["name"].split(".")
            try:
                valores[pd.Timestamp(int(anio), MESES[mes], 1)] = float(periodo["values"][0])
            except ValueError:
                pass  # "n.d."
        salida[nombre] = pd.Series(valores)
        print(nombre, codigo, len(valores), max(valores).date())
    df = pd.DataFrame(salida).sort_index()
    df.index.name = "fecha"
    df.to_csv(DATA / "series_bcrp.csv")
    return df


def cargar(forzar=False):
    archivo = DATA / "series_bcrp.csv"
    if forzar or not archivo.exists():
        return descargar()
    return pd.read_csv(archivo, index_col=0, parse_dates=True)


# ---------- Modelos SARIMA ----------
def pulsos(indice):
    """Variables de intervención para los meses de la pandemia (mar-2020 a mar-2021)."""
    meses = pd.date_range("2020-03-01", "2021-03-01", freq="MS")
    return pd.DataFrame({m.strftime("c%Y%m"): (indice == m).astype(float) for m in meses}, index=indice)


def ajustar(y, orden, estacional):
    return SARIMAX(y, exog=pulsos(y.index), order=orden, seasonal_order=estacional, trend="n",
                   enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=200)


def exog_futuro(y, h):
    futuro = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    return pulsos(futuro)


def seleccionar(y, estacional=True):
    """Búsqueda en malla por AIC."""
    ordenes = [(p, 1, q) for p, q in itertools.product(range(3), range(3))]
    est = [(P, 1, Q, 12) for P, Q in [(0, 1), (1, 0), (1, 1)]] if estacional else [(0, 0, 0, 0)]
    mejor = (np.inf, None, None)
    for o, s in itertools.product(ordenes, est):
        try:
            r = ajustar(y, o, s)
            if r.aic < mejor[0]:
                mejor = (r.aic, o, s)
        except Exception:
            pass
    return mejor


def backtest(y, orden, est, hs=(3, 6, 10), n_origenes=24):
    """Origen móvil: compara el modelo con el pronóstico ingenuo, en variación a 12 meses."""
    filas = []
    for k in range(n_origenes, 0, -1):
        corte = len(y) - k - max(hs) + 1
        tr = y.iloc[:corte]
        try:
            f = ajustar(tr, orden, est).get_forecast(max(hs), exog=exog_futuro(tr, max(hs))).predicted_mean
        except Exception:
            continue
        ult = tr.index[-1]
        ingenuo = 100 * (np.exp(tr.loc[ult] - y.loc[ult - pd.DateOffset(years=1)]) - 1)
        for h in hs:
            fecha = f.index[h - 1]
            base = y.loc[fecha - pd.DateOffset(years=1)]
            filas.append(dict(h=h, real=100 * (np.exp(y.loc[fecha] - base) - 1),
                              modelo=100 * (np.exp(f.iloc[h - 1] - base) - 1), ingenuo=ingenuo))
    d = pd.DataFrame(filas)
    return pd.DataFrame([dict(h=h, n=len(g), MAE_modelo=(g.real - g.modelo).abs().mean(),
                              MAE_ingenuo=(g.real - g.ingenuo).abs().mean()) for h, g in d.groupby("h")])


def pronostico(serie, nombre, estacional=True, nsim=5000, previo=None):
    y = np.log(serie.dropna()).asfreq("MS").loc["2010-01-01":]
    if previo:
        o, s = tuple(previo["orden"]), tuple(previo["estacional"])
        r = ajustar(y, o, s)
        aic = r.aic
    else:
        aic, o, s = seleccionar(y, estacional)
        r = ajustar(y, o, s)
    h = (FIN_H1.year - y.index[-1].year) * 12 + FIN_H1.month - y.index[-1].month
    x = exog_futuro(y, h)
    g = r.get_forecast(h, exog=x)
    ic80, ic95 = g.conf_int(alpha=.20), g.conf_int(alpha=.05)
    np.random.seed(SEED)
    sim = r.simulate(h, repetitions=nsim, anchor="end", exog=x)
    sim = pd.DataFrame(np.asarray(sim).reshape(h, nsim), index=g.predicted_mean.index)
    lb = sm.stats.acorr_ljungbox(r.resid.iloc[24:], lags=[12], return_df=True).lb_pvalue.iloc[0]
    tabla = pd.DataFrame(dict(central=np.exp(g.predicted_mean), li80=np.exp(ic80.iloc[:, 0]), ls80=np.exp(ic80.iloc[:, 1]),
                              li95=np.exp(ic95.iloc[:, 0]), ls95=np.exp(ic95.iloc[:, 1])))
    bt = previo["backtest"] if previo else backtest(y, o, s)
    return dict(nombre=nombre, orden=o, est=s, aic=aic, tabla=tabla, sim=sim, backtest=bt, ljungbox=lb)


def var12_sim(sim_log, historia):
    """Variación a 12 meses de cada trayectoria simulada."""
    niv = np.exp(sim_log)
    salida = pd.DataFrame(index=niv.index, columns=niv.columns, dtype=float)
    for f in niv.index:
        b = f - pd.DateOffset(years=1)
        base = historia.loc[b] if b in historia.index else niv.loc[b]
        salida.loc[f] = 100 * (niv.loc[f] / base - 1)
    return salida


def var12_central(central, historia):
    s = pd.concat([historia, central])
    return (100 * (s / s.shift(12) - 1)).loc[central.index]


# ---------- Determinantes ----------
def preparar_variables(df):
    g = 100 * (df.pbi / df.pbi.shift(12) - 1)
    exp3 = df.export.rolling(3).mean()
    X = pd.DataFrame({
        "Cobre (var. % 12m)": 100 * (df.cobre / df.cobre.shift(12) - 1),
        "Oro (var. % 12m)": 100 * (df.oro / df.oro.shift(12) - 1),
        "Términos de intercambio (var. % 12m)": 100 * (df.ti / df.ti.shift(12) - 1),
        "Exportaciones (var. % 12m, prom. 3m)": 100 * (exp3 / exp3.shift(12) - 1),
        "Expectativas económicas a 12 meses": df.exp_econ_12m,
        "Tasa real ex ante": df.tasa_ref - df.exp_infl_12m,
        "Inflación a 12 meses": df.infl_12m,
        "Tipo de cambio (var. % 12m)": 100 * (df.tc / df.tc.shift(12) - 1),
        "PBI (var. % 12m, prom. 6m)": g.rolling(6).mean(),
    })
    # Objetivo directo: crecimiento promedio de t+5 a t+10 (ene-jun de 2027 si t = ago-2026)
    Y = sum(g.shift(-k) for k in range(5, 11)) / 6
    return X, Y.rename("objetivo")


def determinantes(df):
    X, Y = preparar_variables(df)
    cols = list(X.columns)
    ult = X.dropna().index[-1]
    tr = pd.concat([X, Y], axis=1).loc["2005-01-01":].dropna()
    # Excluye observaciones cuyo objetivo o variación a 12 meses toca la pandemia
    tr = tr[~((tr.index >= "2019-03-01") & (tr.index <= "2022-12-01"))]
    ols = sm.OLS(tr.objetivo, sm.add_constant(tr[cols])).fit(cov_type="HAC", cov_kwds={"maxlags": 12})
    coef = pd.DataFrame(dict(beta=ols.params, ee_hac=ols.bse, p=ols.pvalues)).drop("const")
    coef["efecto_1de"] = coef.beta * tr[cols].std()
    modelos = {
        "Ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30))),
        "Bosque aleatorio": RandomForestRegressor(n_estimators=300, max_depth=4, min_samples_leaf=5, random_state=SEED),
        "Gradient boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=.03, max_depth=2, subsample=.8, random_state=SEED),
    }
    # Origen móvil trimestral: cada predicción usa solo objetivos ya conocidos en t (t-10 o antes)
    fechas = tr.index[tr.index >= "2014-01-01"][::3]
    pred = {m: [] for m in modelos}
    real, persist = [], []
    for t in fechas:
        ent = tr[tr.index <= t - pd.DateOffset(months=10)]
        for m, est in modelos.items():
            pred[m].append(est.fit(ent[cols], ent.objetivo).predict(tr.loc[[t], cols])[0])
        real.append(tr.objetivo[t])
        persist.append(tr.loc[t, "PBI (var. % 12m, prom. 6m)"])
    mae = {m: mean_absolute_error(real, p) for m, p in pred.items()}
    mae["Persistencia (referencia)"] = mean_absolute_error(real, persist)
    pred_h1 = {m: float(est.fit(tr[cols], tr.objetivo).predict(X.loc[[ult], cols])[0]) for m, est in modelos.items()}
    mejor = min(modelos, key=lambda m: mae[m])
    pi = permutation_importance(modelos[mejor].fit(tr[cols], tr.objetivo), tr[cols], tr.objetivo, n_repeats=50,
                                random_state=SEED, scoring="neg_mean_absolute_error")
    imp = pd.Series(pi.importances_mean, index=cols).sort_values(ascending=False)
    return dict(coef=coef, r2=ols.rsquared, n=int(ols.nobs), mae=mae, pred_h1=pred_h1, importancia=imp, mejor=mejor,
                x_ult=X.loc[ult], fecha_x=ult, n_val=len(fechas))


# ---------- Figuras ----------
TINTA, GRIS, AZUL, BANDA = "#222222", "#8a8a8a", "#1f5f99", "#cfe0ef"
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 12, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": GRIS, "axes.grid": True, "grid.color": "#e6e6e6",
                     "grid.linewidth": .7, "axes.axisbelow": True, "figure.dpi": 200})


def coma(ax, eje="y"):
    """Coma decimal, como publica el BCRP en español."""
    fmt = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
    (ax.yaxis if eje == "y" else ax.xaxis).set_major_formatter(fmt)


def abanico(ax, hist, t):
    ax.axvspan(INICIO_H1, FIN_H1, color="#f1f1f1", zorder=0)
    ax.plot(hist.index, hist.values, color=TINTA, lw=1.6, label="Observado")
    ax.fill_between(t.index, t.li80, t.ls80, color=BANDA, lw=0, label="Intervalo de predicción al 80 %")
    ax.plot(t.index, t.central, color=AZUL, lw=2, ls="--", label="Pronóstico central")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--descargar", action="store_true")
    ap.add_argument("--rapido", action="store_true")
    args = ap.parse_args()
    df = cargar(args.descargar)

    def previo(clave):
        if not args.rapido:
            return None
        m = json.loads((RES / "resumen.json").read_text(encoding="utf-8"))["modelos"][clave]
        b = pd.read_csv(RES / "backtest.csv", index_col=0)
        return dict(m, backtest=b[b.index == clave][["h", "n", "MAE_modelo", "MAE_ingenuo"]].reset_index(drop=True))

    pbi, ipc = pronostico(df.pbi, "PBI", previo=previo("PBI")), pronostico(df.ipc, "IPC", previo=previo("IPC"))
    otros = {k: pronostico(df[k], k, estacional=False, previo=previo(k)) for k in ["cobre", "tc", "ti", "exp_econ_12m"]}

    h_pbi, h_ipc = df.pbi.dropna(), df.ipc.dropna()
    sim_g, sim_pi = var12_sim(pbi["sim"], h_pbi), var12_sim(ipc["sim"], h_ipc)
    t_g = pd.DataFrame(dict(central=var12_central(pbi["tabla"].central, h_pbi),
                            li80=sim_g.quantile(.1, axis=1), ls80=sim_g.quantile(.9, axis=1)))
    t_pi = pd.DataFrame(dict(central=var12_central(ipc["tabla"].central, h_ipc),
                             li80=sim_pi.quantile(.1, axis=1), ls80=sim_pi.quantile(.9, axis=1)))
    h1 = sim_g.loc[INICIO_H1:FIN_H1].mean(axis=0)
    jun = sim_pi.loc[FIN_H1]
    prob = dict(h1_central=float(t_g.central.loc[INICIO_H1:FIN_H1].mean()), h1_media=float(h1.mean()),
                h1_p10=float(h1.quantile(.1)), h1_p90=float(h1.quantile(.9)), p_h1_mayor_2=float((h1 > 2).mean()),
                p_h1_mayor_3=float((h1 > 3).mean()), p_h1_mayor_34=float((h1 > 3.4).mean()),
                p_infl_jun_rango=float(((jun >= 1) & (jun <= 3)).mean()), p_infl_jun_menor_4=float((jun < 4).mean()),
                p_infl_ene_rango=float(((sim_pi.loc[INICIO_H1] >= 1) & (sim_pi.loc[INICIO_H1] <= 3)).mean()))
    # Inflación mensual promedio de nov-2026 a jun-2027 compatible con 3 % a 12 meses en junio de 2027
    ipc_ult, ipc_base = h_ipc.iloc[-1], h_ipc.loc["2026-06-01"]
    n_meses = 8
    prob["mensual_para_3"] = float(100 * ((1.03 * ipc_base / ipc_ult) ** (1 / n_meses) - 1))
    prob["mensual_modelo"] = float(100 * ((ipc["tabla"].central.loc[FIN_H1] / ipc_ult) ** (1 / n_meses) - 1))
    det = determinantes(df)

    # Escenarios: desplazan el central con los coeficientes OLS (asociación, no causalidad)
    c = det["coef"].beta
    base = prob["h1_central"]

    def esc(ti, tr):
        return base + c["Términos de intercambio (var. % 12m)"] * ti + c["Tasa real ex ante"] * tr
    # Solo se mueven las dos variables con coeficiente significativo al 5 %
    escenarios = pd.DataFrame([
        dict(escenario="Favorable", ti=5, tasa_real=-.25, pbi_h1=esc(5, -.25)),
        dict(escenario="Base", ti=0, tasa_real=0, pbi_h1=base),
        dict(escenario="Adverso", ti=-5, tasa_real=.5, pbi_h1=esc(-5, .5))])

    # Exportación
    t_g.round(3).to_csv(RES / "pronostico_pbi_var12m.csv")
    t_pi.round(3).to_csv(RES / "pronostico_inflacion_12m.csv")
    pd.concat({k: v["tabla"] for k, v in otros.items()}).round(3).to_csv(RES / "pronostico_condiciones.csv")
    det["coef"].round(4).to_csv(RES / "determinantes_ols_hac.csv")
    det["importancia"].round(4).to_csv(RES / "importancia_permutacion.csv")
    escenarios.round(3).to_csv(RES / "escenarios.csv", index=False)
    todos = {"PBI": pbi, "IPC": ipc, **otros}
    bts = pd.concat({k: v["backtest"] for k, v in todos.items()}).round(3).droplevel(1)
    bts.to_csv(RES / "backtest.csv")
    resumen = dict(
        ultimo_dato={k: df[k].dropna().index[-1].strftime("%Y-%m") for k in df},
        ultimo_valor={k: round(float(df[k].dropna().iloc[-1]), 3) for k in df},
        modelos={k: dict(orden=v["orden"], estacional=v["est"], aic=round(float(v["aic"]), 1), ljungbox_p=round(float(v["ljungbox"]), 3)) for k, v in todos.items()},
        probabilidades={k: round(v, 3) for k, v in prob.items()},
        ml_mae={k: round(v, 3) for k, v in det["mae"].items()}, ml_pred_h1={k: round(v, 3) for k, v in det["pred_h1"].items()},
        ml_mejor=det["mejor"], ml_n_validacion=det["n_val"], ols_r2=round(det["r2"], 3), ols_n=det["n"],
        condiciones={k: dict(prom_h1=round(float(v["tabla"].central.loc[INICIO_H1:FIN_H1].mean()), 3), jun=round(float(v["tabla"].central.loc[FIN_H1]), 3),
                             li80_jun=round(float(v["tabla"].li80.loc[FIN_H1]), 3), ls80_jun=round(float(v["tabla"].ls80.loc[FIN_H1]), 3)) for k, v in otros.items()})
    (RES / "resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")

    # Figuras: sin título interno, el título APA va en el documento
    d0 = "2022-01-01"
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    abanico(ax, (100 * (h_pbi / h_pbi.shift(12) - 1)).loc[d0:], t_g)
    ax.axhline(0, color=GRIS, lw=.8)
    ax.set_ylabel("Variación a 12 meses (%)")
    coma(ax)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(.5, 1.0), ncol=3, fontsize=10, columnspacing=1.2, handlelength=1.6)
    fig.tight_layout(); fig.savefig(FIG / "figura1_pbi.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.axhspan(1, 3, color="#e9e9e9", zorder=0)
    abanico(ax, df.infl_12m.dropna().loc[d0:], t_pi)
    ax.text(pd.Timestamp("2022-02-01"), 1.25, "Rango meta del BCRP (1 % a 3 %)", fontsize=10, color=TINTA)
    ax.set_ylabel("Inflación a 12 meses (%)")
    coma(ax)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(.5, 1.0), ncol=3, fontsize=10, columnspacing=1.2, handlelength=1.6)
    fig.tight_layout(); fig.savefig(FIG / "figura2_inflacion.png"); plt.close(fig)

    etiquetas = {"cobre": "Cobre (¢US$/lb)", "tc": "Tipo de cambio (S/ por US$)", "ti": "Términos de intercambio (2007 = 100)",
                 "exp_econ_12m": "Expectativas económicas a 12 meses (índice)"}
    fig, axs = plt.subplots(2, 2, figsize=(6.5, 4.6))
    for ax, (k, v) in zip(axs.ravel(), otros.items()):
        abanico(ax, df[k].dropna().loc["2023-01-01":], v["tabla"])
        ax.set_title(etiquetas[k], fontsize=10.5, loc="left", color=TINTA)
        ax.tick_params(labelsize=9)
        coma(ax)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout(); fig.savefig(FIG / "figura3_condiciones.png"); plt.close(fig)

    imp = det["importancia"].sort_values()
    fig, ax = plt.subplots(figsize=(6.5, 3.3))
    ax.barh(imp.index, imp.values, color=AZUL, height=.55)
    ax.set_xlabel("Aumento del error absoluto medio al permutar la variable (p. p.)", fontsize=10)
    ax.grid(axis="y", visible=False)
    ax.tick_params(labelsize=10)
    coma(ax, "x")
    fig.tight_layout(); fig.savefig(FIG / "figura4_importancia.png"); plt.close(fig)

    print(json.dumps(resumen, indent=1, ensure_ascii=False))
    for titulo, obj in [("PBI", t_g), ("INFL", t_pi), ("BT", bts), ("OLS", det["coef"]), ("IMP", det["importancia"]), ("ESC", escenarios), ("X", det["x_ult"])]:
        print("---", titulo); print(obj.round(3).to_string())
    print(df.iloc[-14:].round(2).to_string())


if __name__ == "__main__":
    main()
