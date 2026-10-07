from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="Wine Quality · Análise Estatística",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
h1, h2, h3 {letter-spacing: -0.02em;}
[data-testid="stMetricLabel"] p {color: #cbd5e1 !important; font-weight: 600;}
[data-testid="stMetricValue"] {color: #ffffff !important;}
.hero {padding: 22px 26px; border-radius: 18px; margin-bottom: 18px; background: linear-gradient(110deg, #172033 0%, #334155 55%, #4b5563 100%); color: white; box-shadow: 0 8px 26px rgba(15,23,42,.18);}
.hero h1 {margin: 0 0 6px 0; color: white;}
.hero p {margin: 0; color: #dbe4ef; font-size: 1.02rem;}
.insight-card {padding: 14px 16px; border-radius: 14px; background: #f8fafc; border: 1px solid #dbe3ec; min-height: 105px;}
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #18212f 0%, #26364a 100%);
    border: 1px solid #40536b;
    border-radius: 12px;
    padding: 14px 16px;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def carregar_dados():
    caminho = Path(__file__).resolve().parent / "WineQT.csv"
    df = pd.read_csv(caminho)
    n_original = len(df)

    df = df.drop("Id", axis=1).drop_duplicates().copy()
    n_pos_duplicatas = len(df)

    df["chlorides_log"] = np.log(df["chlorides"])
    df["residual_sugar_log"] = np.log(df["residual sugar"])

    def mascara_iqr(serie, multiplicador):
        q1 = serie.quantile(0.25)
        q3 = serie.quantile(0.75)
        iqr = q3 - q1
        return (serie < q1 - multiplicador * iqr) | (serie > q3 + multiplicador * iqr)

    mascara = pd.Series(False, index=df.index)

    for coluna in ["fixed acidity", "volatile acidity", "citric acid"]:
        mascara |= mascara_iqr(df[coluna], 1.5)

    for coluna in ["free sulfur dioxide", "total sulfur dioxide", "pH", "sulphates", "alcohol"]:
        mascara |= mascara_iqr(df[coluna], 3.0)

    for coluna in ["chlorides_log", "residual_sugar_log"]:
        mascara |= mascara_iqr(df[coluna], 3.0)

    df = df.loc[~mascara].copy()
    df["faixa_qualidade"] = pd.cut(
        df["quality"],
        bins=[2, 5, 6, 8],
        labels=["baixa", "media", "alta"],
        ordered=True,
    )
    df["razao_so2_livre"] = df["free sulfur dioxide"] / df["total sulfur dioxide"]

    return df, {
        "n_original": n_original,
        "n_pos_duplicatas": n_pos_duplicatas,
        "n_final": len(df),
        "duplicatas_removidas": n_original - n_pos_duplicatas,
        "outliers_removidos": n_pos_duplicatas - len(df),
    }

@st.cache_resource
def ajustar_modelo(df):
    variaveis = [
        "alcohol",
        "volatile acidity",
        "sulphates",
        "citric acid",
        "density",
        "total sulfur dioxide",
    ]
    X = sm.add_constant(df[variaveis])
    y = df["quality"]
    modelo = sm.OLS(y, X).fit()
    robusto = modelo.get_robustcov_results(cov_type="HC3")

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        df[variaveis], y, test_size=0.20, random_state=42
    )
    modelo_treino = sm.OLS(
        y_treino, sm.add_constant(X_treino, has_constant="add")
    ).fit()
    previsoes = modelo_treino.predict(
        sm.add_constant(X_teste, has_constant="add")
    )

    vif = pd.DataFrame({
        "Variável": variaveis,
        "VIF": [
            variance_inflation_factor(df[variaveis].values, i)
            for i in range(len(variaveis))
        ],
    })

    bp = het_breuschpagan(modelo.resid, modelo.model.exog)

    coef = pd.DataFrame({
        "Variável": ["Intercepto"] + variaveis,
        "Coeficiente": robusto.params,
        "p-valor": robusto.pvalues,
        "IC 95% inferior": robusto.conf_int()[:, 0],
        "IC 95% superior": robusto.conf_int()[:, 1],
    })

    metricas = {
        "R²": modelo.rsquared,
        "R² ajustado": modelo.rsquared_adj,
        "F robusto": float(robusto.fvalue),
        "p-valor F": float(robusto.f_pvalue),
        "MAE": mean_absolute_error(y_teste, previsoes),
        "RMSE": mean_squared_error(y_teste, previsoes) ** 0.5,
        "R² teste": r2_score(y_teste, previsoes),
        "BP p-valor": float(bp[1]),
        "BP F p-valor": float(bp[3]),
    }

    pred_df = pd.DataFrame({
        "real": y_teste.to_numpy(),
        "previsto": np.asarray(previsoes),
    }).sort_values("real")

    return modelo, robusto, coef, vif, metricas, pred_df

@st.cache_data
def testes(df):
    alpha = 0.05
    faixas = ["baixa", "media", "alta"]
    resultados = []

    for coluna, nome in [
        ("alcohol", "Álcool"),
        ("volatile acidity", "Acidez volátil"),
        ("sulphates", "Sulfatos"),
    ]:
        grupos = [
            df.loc[df["faixa_qualidade"] == faixa, coluna].dropna()
            for faixa in faixas
        ]
        kw = stats.kruskal(*grupos)
        rho, p_spear = stats.spearmanr(df[coluna], df["quality"])

        resultados.append({
            "Variável": nome,
            "Kruskal-Wallis H": kw.statistic,
            "p-valor KW": kw.pvalue,
            "Spearman rho": rho,
            "p-valor Spearman": p_spear,
            "Decisão 5%": "Rejeitar H₀" if kw.pvalue < alpha else "Não rejeitar H₀",
        })

    return pd.DataFrame(resultados)

df, info = carregar_dados()
modelo, robusto, coef, vif, metricas, pred_df = ajustar_modelo(df)
tabela_testes = testes(df)

st.markdown("""<div class="hero"><h1>🍷 Wine Quality · Análise Estatística</h1><p>Exploração dos dados, evidências estatísticas e modelo de regressão para entender os fatores associados à qualidade.</p></div>""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Filtros")
    faixa = st.multiselect(
        "Faixa de qualidade",
        options=["baixa", "media", "alta"],
        default=["baixa", "media", "alta"],
        format_func=lambda x: {"baixa": "Baixa (3–5)", "media": "Média (6)", "alta": "Alta (7–8)"}[x],
    )
    qualidade = st.slider("Nota de qualidade", 3, 8, (3, 8))

df_filtro = df[
    df["faixa_qualidade"].isin(faixa)
    & df["quality"].between(qualidade[0], qualidade[1])
].copy()

aba1, aba2, aba3, aba4, aba5 = st.tabs([
    "Visão geral",
    "Relações e testes",
    "Regressão",
    "Diagnósticos",
    "Previsão de qualidade",
])

with aba1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Vinhos analisados", f"{info['n_final']:,}".replace(",", "."))
    c2.metric("Nota média", f"{df_filtro['quality'].mean():.2f}".replace(".", ","))
    c3.metric("Álcool médio", f"{df_filtro['alcohol'].mean():.2f}%".replace(".", ","))
    c4.metric("Variáveis químicas", "11")

    col1, col2 = st.columns(2)

    with col1:
        cont = (
            df_filtro["quality"]
            .value_counts()
            .sort_index()
            .rename_axis("quality")
            .reset_index(name="quantidade")
        )
        fig = px.bar(
            cont, x="quality", y="quantidade",
            labels={"quality": "Nota de qualidade", "quantidade": "Quantidade"},
            title="Distribuição da qualidade",
        )
        fig.update_layout(template="plotly_white", height=390)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(
            df_filtro,
            x="faixa_qualidade",
            y="alcohol",
            category_orders={"faixa_qualidade": ["baixa", "media", "alta"]},
            labels={"faixa_qualidade": "Faixa", "alcohol": "Álcool"},
            title="Álcool por faixa de qualidade",
        )
        fig.update_layout(template="plotly_white", height=390)
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        fig = px.box(
            df_filtro,
            x="faixa_qualidade",
            y="volatile acidity",
            category_orders={"faixa_qualidade": ["baixa", "media", "alta"]},
            labels={"faixa_qualidade": "Faixa", "volatile acidity": "Acidez volátil"},
            title="Acidez volátil por faixa",
        )
        fig.update_layout(template="plotly_white", height=390)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(
            df_filtro,
            x="faixa_qualidade",
            y="sulphates",
            category_orders={"faixa_qualidade": ["baixa", "media", "alta"]},
            labels={"faixa_qualidade": "Faixa", "sulphates": "Sulfatos"},
            title="Sulfatos por faixa",
        )
        fig.update_layout(template="plotly_white", height=390)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Evolução dos principais indicadores por faixa")
    col1, col2 = st.columns(2)
    with col1:
        medias = df_filtro.groupby("faixa_qualidade", observed=True)["alcohol"].mean().reindex(["baixa", "media", "alta"]).reset_index()
        fig = px.bar(medias, x="faixa_qualidade", y="alcohol", text_auto=".2f", category_orders={"faixa_qualidade": ["baixa", "media", "alta"]}, labels={"faixa_qualidade":"Faixa de qualidade", "alcohol":"Álcool médio"}, title="Álcool médio cresce com a qualidade")
        fig.update_layout(template="plotly_white", height=350)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        medias = df_filtro.groupby("faixa_qualidade", observed=True)["volatile acidity"].mean().reindex(["baixa", "media", "alta"]).reset_index()
        fig = px.bar(medias, x="faixa_qualidade", y="volatile acidity", text_auto=".3f", category_orders={"faixa_qualidade": ["baixa", "media", "alta"]}, labels={"faixa_qualidade":"Faixa de qualidade", "volatile acidity":"Acidez volátil média"}, title="Acidez volátil diminui com a qualidade")
        fig.update_layout(template="plotly_white", height=350)
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(df_filtro, x="volatile acidity", y="quality", trendline="ols", opacity=0.4, labels={"volatile acidity":"Acidez volátil", "quality":"Qualidade"}, title="Acidez volátil × qualidade")
        fig.update_layout(template="plotly_white", height=400)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.scatter(df_filtro, x="sulphates", y="quality", trendline="ols", opacity=0.4, labels={"sulphates":"Sulfatos", "quality":"Qualidade"}, title="Sulfatos × qualidade")
        fig.update_layout(template="plotly_white", height=400)
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(df_filtro, x="citric acid", y="quality", trendline="ols", opacity=0.4, labels={"citric acid":"Ácido cítrico", "quality":"Qualidade"}, title="Ácido cítrico × qualidade")
        fig.update_layout(template="plotly_white", height=400)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.scatter(df_filtro, x="total sulfur dioxide", y="quality", trendline="ols", opacity=0.4, labels={"total sulfur dioxide":"SO₂ total", "quality":"Qualidade"}, title="SO₂ total × qualidade")
        fig.update_layout(template="plotly_white", height=400)
        st.plotly_chart(fig, use_container_width=True)

    faixa_cont = df_filtro["faixa_qualidade"].value_counts().reindex(["baixa", "media", "alta"]).fillna(0).reset_index()
    faixa_cont.columns = ["faixa_qualidade", "quantidade"]
    fig = px.pie(faixa_cont, names="faixa_qualidade", values="quantidade", hole=0.45, title="Participação de cada faixa na amostra")
    fig.update_layout(template="plotly_white", height=420)
    st.plotly_chart(fig, use_container_width=True)

with aba2:
    st.subheader("Associação entre variáveis")

    fig = px.scatter(
        df_filtro,
        x="alcohol",
        y="quality",
        trendline="ols",
        labels={"alcohol": "Álcool", "quality": "Qualidade"},
        title="Álcool × qualidade",
        opacity=0.45,
    )
    fig.update_layout(template="plotly_white", height=430)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.dataframe(
            tabela_testes.style.format({
                "Kruskal-Wallis H": "{:.3f}",
                "p-valor KW": "{:.3e}",
                "Spearman rho": "{:.3f}",
                "p-valor Spearman": "{:.3e}",
            }),
            use_container_width=True,
            hide_index=True,
        )

    with col2:
        resumo = (
            df_filtro.groupby("faixa_qualidade", observed=True)
            .agg(
                quantidade=("quality", "size"),
                qualidade_media=("quality", "mean"),
                alcool_medio=("alcohol", "mean"),
                acidez_volatil_media=("volatile acidity", "mean"),
                sulfatos_medio=("sulphates", "mean"),
            )
            .reset_index()
        )
        st.dataframe(
            resumo.style.format({
                "qualidade_media": "{:.2f}",
                "alcool_medio": "{:.2f}",
                "acidez_volatil_media": "{:.3f}",
                "sulfatos_medio": "{:.3f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Matriz de correlação")
    corr_cols = [
        "fixed acidity", "volatile acidity", "citric acid",
        "residual sugar", "chlorides", "free sulfur dioxide",
        "total sulfur dioxide", "density", "pH", "sulphates",
        "alcohol", "quality"
    ]
    corr = df_filtro[corr_cols].corr()
    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Correlação de Pearson",
    )
    fig.update_layout(template="plotly_white", height=650)
    st.plotly_chart(fig, use_container_width=True)

with aba3:
    st.subheader("Modelo de regressão linear múltipla")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R²", f"{metricas['R²']:.3f}".replace(".", ","))
    c2.metric("R² ajustado", f"{metricas['R² ajustado']:.3f}".replace(".", ","))
    c3.metric("MAE · teste", f"{metricas['MAE']:.3f}".replace(".", ","))
    c4.metric("RMSE · teste", f"{metricas['RMSE']:.3f}".replace(".", ","))

    st.dataframe(
        coef.style.format({
            "Coeficiente": "{:.4f}",
            "p-valor": "{:.3e}",
            "IC 95% inferior": "{:.4f}",
            "IC 95% superior": "{:.4f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    fig = px.scatter(
        pred_df,
        x="real",
        y="previsto",
        labels={"real": "Qualidade real", "previsto": "Qualidade prevista"},
        title="Valores reais × previstos · conjunto de teste",
        opacity=0.6,
    )
    minimo = min(pred_df["real"].min(), pred_df["previsto"].min())
    maximo = max(pred_df["real"].max(), pred_df["previsto"].max())
    fig.add_trace(go.Scatter(
        x=[minimo, maximo],
        y=[minimo, maximo],
        mode="lines",
        name="Linha ideal",
    ))
    fig.update_layout(template="plotly_white", height=430)
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "A inferência dos coeficientes usa erros-padrão robustos HC3, "
        "pois o diagnóstico indicou heterocedasticidade."
    )

with aba4:
    st.subheader("Diagnóstico do modelo")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Multicolinearidade — VIF**")
        st.dataframe(
            vif.style.format({"VIF": "{:.2f}"}),
            use_container_width=True,
            hide_index=True,
        )

    with col2:
        st.write("**Homoscedasticidade — Breusch-Pagan**")
        st.metric("p-valor", f"{metricas['BP p-valor']:.3e}")
        st.write(
            "H₀: variância dos resíduos constante. "
            "O p-valor inferior a 0,05 fornece evidência de heterocedasticidade."
        )

    residuos = modelo.resid
    ajustados = modelo.fittedvalues
    fig = px.scatter(
        x=ajustados,
        y=residuos,
        labels={"x": "Valores ajustados", "y": "Resíduos"},
        title="Resíduos × valores ajustados",
        opacity=0.55,
    )
    fig.add_hline(y=0, line_dash="dash")
    fig.update_layout(template="plotly_white", height=430)
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Os diagnósticos devem ser considerados junto às métricas de desempenho. "
        "O modelo apresenta capacidade explicativa moderada e deve ser interpretado "
        "como modelo de associação/predição, não como evidência de causalidade."
    )

with aba5:
    st.subheader("Previsão da qualidade de um vinho")
    st.write(
        "Informe as características químicas do vinho para obter uma estimativa "
        "de qualidade pelo modelo de regressão linear múltipla."
    )

    variaveis_predicao = [
        "alcohol",
        "volatile acidity",
        "sulphates",
        "citric acid",
        "density",
        "total sulfur dioxide",
    ]
    nomes_predicao = {
        "alcohol": "Álcool (%)",
        "volatile acidity": "Acidez volátil",
        "sulphates": "Sulfatos",
        "citric acid": "Ácido cítrico",
        "density": "Densidade",
        "total sulfur dioxide": "SO₂ total",
    }

    st.info(
        "Os valores iniciais são as médias da base tratada. Você pode substituí-los "
        "pelas características do vinho que deseja avaliar."
    )

    esquerda, direita = st.columns([1, 1.15])

    with esquerda:
        st.markdown("### Características do vinho")
        alcool = st.number_input(
            "Álcool (%)",
            min_value=float(df["alcohol"].min()),
            max_value=float(df["alcohol"].max()),
            value=float(df["alcohol"].mean()),
            step=0.01,
            format="%.2f",
        )
        acidez_volatil = st.number_input(
            "Acidez volátil",
            min_value=float(df["volatile acidity"].min()),
            max_value=float(df["volatile acidity"].max()),
            value=float(df["volatile acidity"].mean()),
            step=0.001,
            format="%.3f",
        )
        sulfatos = st.number_input(
            "Sulfatos",
            min_value=float(df["sulphates"].min()),
            max_value=float(df["sulphates"].max()),
            value=float(df["sulphates"].mean()),
            step=0.001,
            format="%.3f",
        )
        acido_citrico = st.number_input(
            "Ácido cítrico",
            min_value=float(df["citric acid"].min()),
            max_value=float(df["citric acid"].max()),
            value=float(df["citric acid"].mean()),
            step=0.001,
            format="%.3f",
        )
        densidade = st.number_input(
            "Densidade",
            min_value=float(df["density"].min()),
            max_value=float(df["density"].max()),
            value=float(df["density"].mean()),
            step=0.0001,
            format="%.4f",
        )
        so2_total = st.number_input(
            "SO₂ total",
            min_value=float(df["total sulfur dioxide"].min()),
            max_value=float(df["total sulfur dioxide"].max()),
            value=float(df["total sulfur dioxide"].mean()),
            step=1.0,
            format="%.1f",
        )

    entrada = pd.DataFrame([{
        "alcohol": alcool,
        "volatile acidity": acidez_volatil,
        "sulphates": sulfatos,
        "citric acid": acido_citrico,
        "density": densidade,
        "total sulfur dioxide": so2_total,
    }], columns=variaveis_predicao)

    previsao = float(np.asarray(modelo.predict(sm.add_constant(entrada, has_constant="add")))[0])
    nota_arredondada = int(np.clip(np.rint(previsao), 3, 8))
    faixa_prevista = (
        "baixa" if nota_arredondada <= 5
        else "media" if nota_arredondada == 6
        else "alta"
    )

    with direita:
        st.markdown("### Resultado da previsão")
        st.metric("Qualidade prevista", f"{previsao:.2f}".replace(".", ","))
        st.metric("Nota inteira mais próxima", str(nota_arredondada))
        st.metric("Faixa estimada", {"baixa": "Baixa (3–5)", "media": "Média (6)", "alta": "Alta (7–8)"}[faixa_prevista])

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=previsao,
            number={"valueformat": ".2f"},
            title={"text": "Escore previsto"},
            gauge={
                "axis": {"range": [3, 8], "dtick": 1},
                "bar": {"thickness": 0.25},
                "steps": [
                    {"range": [3, 5.5], "name": "Baixa"},
                    {"range": [5.5, 6.5], "name": "Média"},
                    {"range": [6.5, 8], "name": "Alta"},
                ],
                "threshold": {
                    "line": {"width": 4},
                    "thickness": 0.8,
                    "value": previsao,
                },
            },
        ))
        fig.update_layout(template="plotly_white", height=330, margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Características informadas")
    tabela_entrada = pd.DataFrame({
        "Variável": [nomes_predicao[v] for v in variaveis_predicao],
        "Valor informado": [entrada.iloc[0][v] for v in variaveis_predicao],
        "Média da base": [df[v].mean() for v in variaveis_predicao],
    })
    st.dataframe(
        tabela_entrada.style.format({
            "Valor informado": "{:.4f}",
            "Média da base": "{:.4f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "A previsão é um escore estimado pelo modelo ajustado na base tratada completa. "
        "Ela não representa uma garantia de avaliação sensorial e não deve ser interpretada como causalidade."
    )

st.divider()
st.caption(
    f"Base final: {info['n_final']} observações · "
    f"{info['duplicatas_removidas']} duplicatas removidas após exclusão do identificador · "
    f"{info['outliers_removidos']} observações removidas pelo filtro híbrido de outliers."
)
