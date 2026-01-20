# Optimización de Portafolios de Inversión

Este documento presenta las principales metodologías para optimización de portafolios de inversión, con implementaciones prácticas en Python.

## Tabla de Contenidos

1. [Teoría Moderna de Portafolios (Markowitz)](#1-teoría-moderna-de-portafolios-markowitz)
2. [Modelo Black-Litterman](#2-modelo-black-litterman)
3. [Risk Parity](#3-risk-parity)
4. [Maximum Sharpe Ratio](#5-maximum-sharpe-ratio)
5. [Hierarchical Risk Parity (HRP)](#6-hierarchical-risk-parity-hrp)
6. [Estrategias Activas contra Benchmark](#7-estrategias-activas-contra-benchmark)
8. [Backtesting y Evaluación](#8-backtesting-y-evaluación)

---

## Librerías Necesarias

```python
import numpy as np # arrays y operaciones matemáticas
import pandas as pd # manejo de datos
import yfinance as yf # descarga de datos financieros
from scipy.optimize import minimize # optimización
from scipy.cluster.hierarchy import linkage, dendrogram # clustering jerárquico
from sklearn.covariance import LedoitWolf # estimación robusta de covarianza
import matplotlib.pyplot as plt # visualización
import seaborn as sns # visualización avanzada
```


## 1. Teoría Moderna del Portafolio (Markowitz)

### Metodología

La teoría moderna del portafolio, desarrollada por Harry Markowitz (1952), es el marco conceptual que explica cómo combinar activos para obtener la mejor relación posible entre retorno y riesgo, entendiendo el riesgo no de forma aislada, sino a nivel de portafolio.

La idea central del modelo es que no importa solo cuánto rinde un activo ni cuán riesgoso es por sí solo, sino cómo se comporta en conjunto con los demás activos del portafolio. Dos activos riesgosos pueden, al combinarse, reducir el riesgo total si no se mueven exactamente igual. A esta propiedad se le llama **diversificación**.


**Conceptos Fundamentales:**

1. **Retorno Esperado del Portafolio:**

    $$\mu_p = \sum_{i=1}^{n} w_i \mu_i$$
    
    Donde:
    - $\mu_p$ = retorno esperado del portafolio completo
    - $w_i$ = peso o proporción del activo $i$ en el portafolio (por ejemplo, 0.3 = 30%)
    - $\mu_i$ = retorno esperado del activo $i$
    - $n$ = número total de activos
    
    **Ejemplo simple:** Si tienes 40% en acción A (retorno 10%) y 60% en acción B (retorno 8%), entonces:
    $\mu_p = 0.4 \times 0.10 + 0.6 \times 0.08 = 0.088 = 8.8\%$

2. **Riesgo del Portafolio (Varianza):**
    $$\sigma_p^2 = w^T \Sigma w$$
    
    Donde:
    - $\sigma_p^2$ = varianza del portafolio (medida de riesgo)
    - $w$ = vector de pesos
    - $w^T$ = vector de pesos transpuesto
    - $\Sigma$ = matriz de covarianza (tabla de todas las covarianzas entre activos)
    - Cuando $i=j$, $\sigma_{ii}$ es la varianza del activo $i$
    
    **Volatilidad del portafolio:** $\sigma_p = \sqrt{\sigma_p^2}$ (desviación estándar)
    
    **Nota importante:** La varianza del portafolio NO es simplemente el promedio ponderado de las varianzas individuales. Las correlaciones entre activos reducen el riesgo total (beneficio de la diversificación).

3. **Diversificación:**
    El riesgo total se descompone en:
    - **Riesgo sistemático (de mercado)**: Afecta a todos los activos (recesiones, inflación). No se puede eliminar diversificando.
    - **Riesgo idiosincrático (específico)**: Afecta solo a un activo o sector. Se puede reducir diversificando.
4. **Frontera Eficiente:**

    La frontera eficiente es el conjunto de todos los portafolios óptimos que ofrecen el máximo retorno esperado para cada nivel de riesgo, o el mínimo riesgo para cada nivel de retorno deseado.

    <p align="center">
        <img src="images/esquema_frontera_eficiente.png" alt="Frontera Eficiente" width="600">
    </p>
    
    **¿Qué significa?**
    - El eje X es el riesgo (volatilidad) y el eje Y es el retorno esperado
    - Cada punto representa un portafolio posible con diferentes combinaciones de pesos
    - La frontera eficiente es la curva que une los mejores portafolios posibles
    - Cualquier portafolio por debajo de esta curva es subóptimo (existe otro con más retorno para el mismo riesgo, o menos riesgo para el mismo retorno)
    
    
    **Portafolios clave en la frontera:**
    - **Portafolio de Mínima Varianza Global (GMV)**: El punto más a la izquierda, con el menor riesgo posible
    - **Portafolio Tangente**: El punto con el mejor ratio de Sharpe (retorno por unidad de riesgo)
    
    **Teorema de los Dos Fondos:** Cualquier portafolio en la frontera eficiente puede construirse como una combinación lineal de dos portafolios cualesquiera sobre la frontera.

### Implementación en Python

#### <u>Portafolio de Mínima Varianza </u>

Para encontrar el portafolio de mínima varianza, hay que encontrar la combinación de activos que resuelvan el siguiente problema de optimización:

$$\text{Minimizar varianza del portfolio }(\sigma_p^2): f(\mathbf{w}) = w^T \Sigma w$$

Sujeto a:
- $\sum_{i=1}^{n} w_i = 1$ (los pesos suman 100%)

- $w_i \geq 0$ (no se permiten posiciones cortas)

**Consideración:**

Para resolver el problema de optiminzación, hay que derivar $\sigma_p^2$ respecto a > los pesos $w$:

$$\frac{\partial}{\partial w_k}  \left( \mathbf{w}^\top \Sigma \mathbf{w} \right) = 2 \sum_{j=1}^{n} \Sigma_{kj} w_j$$

Como se aprecia en la ecuación anterior, al momento de derivar aparece la cosntante 2. Si bien es cierto que esta constante no afecta el resultado de la minimización, muchas veces se busca evitar que esta aparezca al multiplicar la fuciión por $\frac{1}{2}$. Debido a lo anterior es que muchas veces la función objtetivo aparece escrita de la siguiente forma:

$$\text{Minimizar: } f(\mathbf{w}) = \frac{1}{2} \mathbf{w}^\top \Sigma \mathbf{w}$$

A continuación se muestra un ejemplo de como estimar el portafolio de mínima varianza en Python:

```python
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize

# tickers de acciones
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
prices = yf.download(tickers, start='2023-01-01', end='2025-12-31')['Close']

# Calcular retornos diarios
returns = np.log(prices / prices.shift(1)).dropna()

# Matriz de covarianza anualizada
cov_matrix = returns.cov() * 252

# Función objetivo: minimizar la varianza del portafolio
def portfolio_variance(weights, cov_matrix):
    contribution_vector = np.dot(weights, cov_matrix)
    variance = np.dot(contribution_vector, weights)
    return variance

# Restricciones: suma de pesos = 1
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

# Límites: pesos entre 0 y 1 (no hay ventas en corto)
bounds = tuple((0, 1) for _ in range(len(tickers)))

# Peso inicial igual para todos los activos
initial_weights = np.array(len(tickers) * [1. / len(tickers)])

# Optimización
result = minimize(portfolio_variance, initial_weights, args=(cov_matrix,),
                  method='SLSQP', bounds=bounds, constraints=constraints)   
min_variance_weights = result.x

df_min_variance = pd.DataFrame({'Ticker': tickers, 'Optimal Weight': min_variance_weights})   
df_min_variance.sort_values(by='Optimal Weight', ascending=False, inplace=True)
df_min_variance['Optimal Weight'] = df_min_variance['Optimal Weight'].apply(lambda x: f"{x:.2%}")
df_min_variance.reset_index(drop=True, inplace=True)

# Retorno esperado y volatilidad del portafolio de mínima varianza
expected_returns = returns.mean() * 252
min_variance_return = np.dot(min_variance_weights, expected_returns)
min_variance_volatility = np.sqrt(portfolio_variance(min_variance_weights, cov_matrix))


```


#### <u>Portafolio Tangente</u>

Para encontrar el portafolio tangente hay que encontrar la combinacion de activos que resuelvan el siguiente problema de optimización:

$$\text{Maximizar el Ratio de Sharpe }(S): S(\mathbf{w}) = \frac{\mu_p - r_f}{\sigma_p}$$

Sujeto a:
- $\sum_{i=1}^{n} w_i = 1$ (los pesos suman 100%)
- $w_i \geq 0$ (no se permiten posiciones cortas)

A continuación, se muestra un ejemplo de cómo implementar esto en Python:

```python
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize

# tickers de acciones
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
prices = yf.download(tickers, start='2023-01-01', end='2025-12-31')['Close']

# Calcular retornos diarios
returns = np.log(prices / prices.shift(1)).dropna()

# Matriz de covarianza anualizada
cov_matrix = returns.cov() 

# Función objetivo: máximizar el ratio de Sharpe (rendimiento/volatilidad)
def negative_sharpe_ratio(weights, returns, cov_matrix):
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    sharpe_ratio = portfolio_return / portfolio_volatility
    return -sharpe_ratio  # Negativo para maximizar

# Restricciones: suma de pesos = 1
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

# Límites: pesos entre 0 y 1 (no hay ventas en corto)
bounds = tuple((0, 1) for _ in range(len(tickers)))

# Peso inicial igual para todos los activos
initial_weights = np.array(len(tickers) * [1. / len(tickers)])

# Optimización
result = minimize(negative_sharpe_ratio, initial_weights, args=(returns, cov_matrix),
                  method='SLSQP', bounds=bounds, constraints=constraints)   
max_sharpe_weights = result.x
df_max_sharpe = pd.DataFrame({'Ticker': tickers, 'Optimal Weight': max_sharpe_weights})   
df_max_sharpe.sort_values(by='Optimal Weight', ascending=False, inplace=True)
df_max_sharpe['Optimal Weight'] = df_max_sharpe['Optimal Weight'].apply(lambda x: f"{x:.2%}")
df_max_sharpe.reset_index(drop=True, inplace=True)

# retorno esperado y volatilidad del portafolio de máxima Sharpe
max_sharpe_returm = np.sum(returns.mean() * max_sharpe_weights) * 252
max_sharpe_volatility = np.sqrt(np.dot(max_sharpe_weights.T, np.dot(cov_matrix, max_sharpe_weights))) * np.sqrt(252)
```


#### Visualización de la Frontera Eficiente

Para poder visualizar la frontera eficiente, vamos a simular 1.000.000 portafolios aleatorios y graficar sus retornos esperados (eje y) vs volatilidad (eje x), junto con el portafolio tangente y el portafolio de mínima varianza.


```python
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ======================================
# Parámetros
# ======================================
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
start_date = "2023-01-01"
end_date = "2025-12-31"
n_portfolios = 1_000_000
rf = 0.03  # tasa libre de riesgo anual (3%)

np.random.seed(42)

# ======================================
# Descarga de precios
# ======================================
prices = yf.download(tickers, start=start_date, end=end_date)["Close"]

# Retornos logarítmicos diarios
returns = np.log(prices / prices.shift(1)).dropna()

# Media y covarianza
mean_returns = returns.mean().values * 252      # retorno esperado anual
cov_matrix = returns.cov().values * 252         # covarianza anualizada

n_assets = len(tickers)

# ======================================
# Simulación Monte Carlo
# ======================================
weights = np.random.random((n_portfolios, n_assets))
weights /= np.sum(weights, axis=1)[:, None]

# Retorno esperado del portafolio
portfolio_returns = weights @ mean_returns

# Volatilidad del portafolio
portfolio_vols = np.sqrt(
    np.einsum("ij,jk,ik->i", weights, cov_matrix, weights)
)

# Sharpe Ratio
sharpe_ratios = (portfolio_returns - rf) / portfolio_vols

# ======================================
# Identificar la Frontera Eficiente
# ======================================
# Dividir en bins de volatilidad y encontrar el máximo retorno en cada bin
n_bins = 100
vol_bins = np.linspace(portfolio_vols.min(), portfolio_vols.max(), n_bins)
efficient_frontier_vols = []
efficient_frontier_returns = []

for i in range(len(vol_bins) - 1):
    mask = (portfolio_vols >= vol_bins[i]) & (portfolio_vols < vol_bins[i + 1])
    if np.any(mask):
        max_return_idx = np.argmax(portfolio_returns[mask])
        efficient_frontier_vols.append(portfolio_vols[mask][max_return_idx])
        efficient_frontier_returns.append(portfolio_returns[mask][max_return_idx])

# ======================================
# Portafolio tangente (Máx Sharpe)
# ======================================
idx_tangent = np.argmax(sharpe_ratios)

ret_tangent = portfolio_returns[idx_tangent]
vol_tangent = portfolio_vols[idx_tangent]
weights_tangent = weights[idx_tangent]

# ======================================
# Capital Market Line (CML)
# ======================================
sigma_cml = np.linspace(0, portfolio_vols.max(), 200)
sharpe_max = sharpe_ratios[idx_tangent]
cml_returns = rf + sharpe_max * sigma_cml

# ======================================
# Gráfico
# ======================================
plt.figure(figsize=(11, 7))

# Todos los portafolios simulados
plt.scatter(
    portfolio_vols,
    portfolio_returns,
    c=sharpe_ratios,
    cmap="viridis",
    s=1,
    alpha=0.3,
    label="Portafolios simulados"
)

# Frontera eficiente resaltada
plt.plot(
    efficient_frontier_vols,
    efficient_frontier_returns,
    color='red',
    linewidth=3,
    label='Frontera Eficiente',
    zorder=5
)

# Portafolio tangente
plt.scatter(max_sharpe_volatility, max_sharpe_returm, color='orange', marker='*', s=300, 
            edgecolors='black', linewidths=1.5, label='Máxima Sharpe', zorder=6)

# Portafolio de mínima varianza
plt.scatter(min_variance_volatility, min_variance_return, color='blue', marker='*', s=300,
            edgecolors='black', linewidths=1.5, label='Mínima Varianza', zorder=6)

# Capital Market Line
plt.plot(
    sigma_cml,
    cml_returns,
    color="black",
    linewidth=2,
    linestyle='--',
    label="Capital Market Line (CML)",
    zorder=4
)

plt.xlabel("Volatilidad anual", fontsize=12)
plt.ylabel("Retorno esperado anual", fontsize=12)
plt.title("Frontera Eficiente, Portafolio Tangente y Portafolio Mínima Varianza", fontsize=14, fontweight='bold')
plt.colorbar(label="Sharpe Ratio")
plt.legend(loc='upper left', fontsize=10)
plt.grid(alpha=0.3)

# Añadir información del portafolio tangente en el gráfico
info_text = f"Portafolio Tangente:\n"
info_text += f"Retorno: {ret_tangent:.2%}\n"
info_text += f"Volatilidad: {vol_tangent:.2%}\n"
info_text += f"Sharpe: {sharpe_max:.2f}\n\n"
info_text += "Pesos:\n"
for t, w in zip(tickers, weights_tangent):
    info_text += f"{t}: {w:.1%}\n"

plt.gcf().text(0.65, 0.65, info_text, fontsize=9, 
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
               verticalalignment='top')

# Añadir información del portafolio de mínima varianza
info_text_minvar = f"Portafolio Mínima Varianza:\n"
info_text_minvar += f"Retorno: {min_variance_return:.2%}\n"
info_text_minvar += f"Volatilidad: {min_variance_volatility:.2%}\n\n"
info_text_minvar += "Pesos:\n"
for t, w in zip(tickers, min_variance_weights):
    info_text_minvar += f"{t}: {w:.1%}\n"

plt.gcf().text(0.30, 0.35, info_text_minvar, fontsize=9,
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
               verticalalignment='top')
plt.savefig(r'images/frontera_eficiente.png', dpi=300, bbox_inches='tight')

plt.show()
```
<p align="center">
    <img src="images/frontera_eficiente.png" alt="Frontera Eficiente" width="600">
</p>

**Resumen de Resultados:**

- La nube de puntos representa todos los portafolios posibles con diferentes combinaciones de activos.
- La línea roja es la frontera eficiente, que muestra los portafolios que ofrecen para nivel de volatilidad el mayor retorno esperado.
- El portafolio de mínima varianza (estrella azul) es el portafolio con el menor riesgo posible.
- El portafolio tangente (estrella naranja) es el portafolio que maximiza el ratio de Sharpe, ofreciendo el mejor retorno por unidad de riesgo.
- La línea discontinua negra es la Capital Market Line (CML), que representa la combinación de la tasa libre de riesgo y el portafolio tangente.  

---

## 2. Modelo Black-Litterman

## 3. Risk Parity

## 4. Minimum Variance Portfolio

## 5. Maximum Sharpe Ratio

## 6. Hierarchical Risk Parity (HRP)

## 7. Estrategias Activas contra Benchmark

## 8. Backtesting y Evaluación

