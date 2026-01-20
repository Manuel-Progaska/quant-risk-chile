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


## 1. Teoría Moderna de Portafolios (Markowitz)

### Metodología

La teoría moderna de portafolios, desarrollada por Harry Markowitz (1952), es el marco conceptual que explica cómo combinar activos para obtener la mejor relación posible entre retorno y riesgo, entendiendo el riesgo no de forma aislada, sino a nivel de portafolio.

La idea central del modelo es que no importa solo cuánto rinde un activo ni cuán riesgoso es por sí solo, sino cómo se comporta en conjunto con los demás activos del portafolio. Dos activos riesgosos pueden, al combinarse, reducir el riesgo total si no se mueven exactamente igual. A esta propiedad se le llama diversificación.


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

    ![Frontera Eficiente](images/frontera_eficiente.png)
    
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

Para encontrar el portafolio de mínima varianza, se resuelve el siguiente problema de optimización:

$$\text{Minimizar varianza del portfolio }(\sigma_p^2): f(\mathbf{w}) = w^T \Sigma w$$

Sujeto a:
- $\sum_{i=1}^{n} w_i = 1$ (los pesos suman 100%)

- $w_i \geq 0$ (no se permiten posiciones cortas)

**Consideración:**

> Para resolver el problema de optiminzación, hay que derivar $\sigma_p^2$ respecto a > los pesos $w$:
>
> $$\frac{\partial}{\partial w_k}
> \left( \mathbf{w}^\top \Sigma \mathbf{w} \right)
> =
> 2 \sum_{j=1}^{n} \Sigma_{kj} w_j$$
>
> Como se aprecia en la ecuación anterior, al momento de derivar aparece un escalar 2 que no afecta el resultado final de la optimización. Para evitar que aparezca este escalar, se suele multipocar la fuciión por $\frac{1}{2}$, quedando la función objetivo como:
>
> $$\text{Minimizar: } f(\mathbf{w}) = \frac{1}{2} \mathbf{w}^\top \Sigma \mathbf{w}$$

A continuación, se muestra un ejemplo de cómo implementar esto en Python:

```python
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize

# tickers de acciones chilenas
tickers = [
    "IAM.SN",         # Inversiones Aguas Metropolitanas S.A.
    "CONCHATORO.SN",  # Viña Concha y Toro S.A.
    "LTM.SN",         # LATAM Airlines Group S.A.
    "SONDA.SN",       # Sonda S.A.
    "BSANTANDER.SN",  # Banco Santander Chile
    "SALFACORP.SN",   # SalfaCorp S.A.
    "AGUAS-A.SN",     # Aguas Andinas S.A.
    "RIPLEY.SN",      # Ripley Corp S.A.
    "ENELAM.SN",      # Enel Américas S.A.
    "CMPC.SN",        # Empresas CMPC S.A.
    "BCI.SN",         # Banco de Crédito e Inversiones
    "CHILE.SN",       # Banco de Chile
    "COLBUN.SN",      # Colbún S.A.
    "ENELCHILE.SN",   # Enel Chile S.A.
    "ENTEL.SN",       # Empresa Nacional de Telecomunicaciones
    "FALABELLA.SN",    # Falabella S.A.     
    "SQM-B.SN"       # Sociedad Química y Minera de Chile S.A. 
]
df_prices = yf.download(tickers, start='2023-01-01', end='2025-12-31')['Close']

# Calcular retornos diarios
returns = df_prices.pct_change().dropna()

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
optimal_weights = result.x

df_min_variance = pd.DataFrame({'Ticker': tickers, 'Optimal Weight': optimal_weights})   
df_min_variance.sort_values(by='Optimal Weight', ascending=False, inplace=True)
df_min_variance['Optimal Weight'] = df_min_variance['Optimal Weight'].apply(lambda x: f"{x:.2%}")
df_min_variance.reset_index(drop=True, inplace=True)
```

El resultado `df_min_variance` mostrará los pesos óptimos para cada activo en el portafolio de mínima varianza:


| Ticker | Optimal Weight |
|--------|----------------|
| SONDA.SN | 13.89% |
| CHILE.SN | 13.11% |
| AGUAS-A.SN | 13.08% |
| FALABELLA.SN | 12.60% |
| RIPLEY.SN | 9.80% |
| IAM.SN | 9.09% |
| SALFACORP.SN | 7.62% |
| CMPC.SN | 7.44% |
| ENTEL.SN | 6.46% |
| COLBUN.SN | 4.59% |
| LTM.SN | 2.18% |
| CONCHATORO.SN | 0.14% |
| ENELAM.SN | 0.00% |
| BSANTANDER.SN | 0.00% |
| ENELCHILE.SN | 0.00% |
| BCI.SN | 0.00% |
| SQM-B.SN | 0.00% |


---

## 2. Modelo Black-Litterman

## 3. Risk Parity

## 4. Minimum Variance Portfolio

## 5. Maximum Sharpe Ratio

## 6. Hierarchical Risk Parity (HRP)

## 7. Estrategias Activas contra Benchmark

## 8. Backtesting y Evaluación

