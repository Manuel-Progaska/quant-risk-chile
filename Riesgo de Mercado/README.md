<div align="center">

# 📊 Gestión de Riesgo de Mercado
## *Implementación Cuantitativa con Python*



![Python](https://img.shields.io/badge/Python-Financial%20Risk-blue?style=for-the-badge&logo=python)
![NumPy](https://img.shields.io/badge/NumPy-Mathematical%20Computing-013243?style=for-the-badge&logo=numpy)
![SciPy](https://img.shields.io/badge/SciPy-Statistical%20Analysis-8CAAE6?style=for-the-badge&logo=scipy)

</div>

<br>

## 🎯 Definición del Riesgo de Mercado

El **riesgo de mercado** se refiere a la posibilidad de que el valor de una inversión disminuya debido a cambios en las condiciones del mercado financiero. Este tipo de riesgo es **sistemático** y afecta a todos los activos financieros, incluyendo:

- **🏛️ Acciones** - Riesgo de precio y volatilidad
- **📋 Bonos** - Riesgo de tasa de interés y crédito  
- **💱 Divisas** - Riesgo cambiario
- **📈 Derivados** - Riesgo de subyacente y volatilidad

<br>

### Factores Determinantes

| Factor | Descripción | Impacto |
|--------|-------------|----------|
| **📈 Tasas de Interés** | Fluctuaciones en política monetaria | Alto |
| **💹 Precios de Activos** | Movimientos del mercado | Directo |
| **⚡ Volatilidad** | Incertidumbre e inestabilidad | Variable |
| **🌍 Eventos Macro** | Políticos, económicos, geopolíticos | Sistémico |

<br>

## 📊 Métricas Fundamentales de Riesgo de Mercado

Para **cuantificar el riesgo de mercado**, se utilizan diversas métricas estadísticas y financieras. La siguiente taxonomía presenta las principales medidas utilizadas en la industria financiera:

### 🔄 **Métricas de Volatilidad y Dispersión**

#### 📈 **Volatilidad**

**Definición**: Mide la variabilidad de los rendimientos de un activo financiero mediante la desviación estándar de los rendimientos históricos.

**Interpretación**: Una mayor volatilidad indica mayor riesgo, ya que los precios pueden fluctuar significativamente en períodos cortos.

**Fórmula**: $\sigma = \sqrt{\frac{1}{n-1} \sum_{i=1}^{n} (r_i - \bar{r})^2}$

#### ⚠️ **Value at Risk (VaR)**

**Definición**: Medida estadística que estima la pérdida máxima potencial de una cartera durante un período específico con un nivel de confianza determinado.

**Ejemplo**: VaR del 5% a un día = Probabilidad del 5% de que la cartera pierda más de una cantidad específica en un día.

**Métodos de Cálculo**:
- Paramétrico (distribución normal)
- Simulación histórica
- Simulación Monte Carlo

#### 📏 **Tracking Error**

**Definición**: Desviación estándar de las diferencias entre los rendimientos de una cartera y su índice de referencia.

**Utilidad**: Evaluar el desempeño relativo vs benchmark. Menor Tracking Error indica mayor similitud comportamental.

**Tipos**:
- **Ex-post**: Basado en datos históricos
- **Ex-ante**: Estimación prospectiva

<br>
### 📊 **Métricas de Sensibilidad al Mercado**

#### 🎯 **Beta de Mercado**

**Definición**: Sensibilidad de los rendimientos de un activo en relación con los rendimientos del mercado.

**Interpretación**:
- β > 1: Activo más volátil que el mercado
- β < 1: Activo menos volátil que el mercado
- β = 1: Misma volatilidad que el mercado

**Aplicación**: Evaluación del riesgo sistemático en el marco del modelo CAPM.

### 📈 **Ratios de Rendimiento Ajustado por Riesgo**

| Ratio | Fórmula | Enfoque | Utilidad |
|-------|---------|---------|----------|
| **Sharpe** | $\frac{R_p - R_f}{\sigma_p}$ | Volatilidad total | Rendimiento por unidad de riesgo total |
| **Sortino** | $\frac{R_p - R_f}{\sigma_{downside}}$ | Volatilidad negativa | Rendimiento por unidad de riesgo a la baja |
| **Treynor** | $\frac{R_p - R_f}{\beta_p}$ | Riesgo sistemático | Rendimiento por unidad de riesgo de mercado |

Donde:
- $R_p$: Rendimiento de la cartera
- $R_f$: Tasa libre de riesgo  
- $\sigma_p$: Desviación estándar de la cartera
- $\sigma_{downside}$: Desviación estándar de rendimientos negativos
- $\beta_p$: Beta de la cartera

<br>

## 💻 Implementación Cuantitativa en Python

Esta sección presenta **implementaciones prácticas** de las métricas de riesgo utilizando Python y sus principales librerías para computación científica y análisis financiero.

### 🎯 **Prerrequisitos Técnicos**

```python
# Librerías requeridas
import numpy as np           # Computación numérica
import pandas as pd          # Manipulación de datos
from scipy import stats      # Estadística avanzada
import matplotlib.pyplot as plt  # Visualización
```

---

### 📊 **1. Cálculo de Volatilidad**

La **volatilidad** se implementa como la desviación estándar de los rendimientos históricos, anualizada mediante el factor $\sqrt{252}$ para datos diarios.

#### **1.1 Volatilidad de Activo Individual**

> **Método**: Desviación estándar de rendimientos históricos anualizada

```python
import numpy as np
import pandas as pd

# Configuración de simulación
np.random.seed(42)
days = 252 * 3  # 3 años de datos diarios (252 días de trading por año)
returns = np.random.normal(0, 0.01, days)  # Rendimientos ~ N(0%, 1%)
returns_series = pd.Series(returns, name='Daily_Returns')

# Cálculo de volatilidad anualizada
volatility = returns_series.std() * np.sqrt(252)
print(f'📊 Volatilidad anualizada: {volatility:.2%}')
```

**Output esperado**: `📊 Volatilidad anualizada: 15.87%`

#### **1.2 Volatilidad de Cartera Multi-Activo**

> **Método**: Matriz de varianza-covarianza con pesos de cartera  
> **Fórmula**: $\sigma_p = \sqrt{w^T \Sigma w}$

```python
import numpy as np
import pandas as pd

# Simulación de cartera multi-activo
num_assets = 4
returns_matrix = np.random.normal(0, 0.01, (days, num_assets))
returns_df = pd.DataFrame(returns_matrix, 
                         columns=[f'Asset_{i+1}' for i in range(num_assets)])

# Configuración de cartera equiponderada
weights = np.array([0.25, 0.25, 0.25, 0.25])  # Pesos iguales

# Matriz de varianza-covarianza
cov_matrix = returns_df.cov() 

# Cálculo de volatilidad de cartera: σ_p = √(w^T Σ w)
portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix.values, weights)))
portfolio_volatility_annualized = portfolio_volatility * np.sqrt(252)

print(f'📈 Volatilidad anual de la cartera: {portfolio_volatility_annualized:.2%}')
```

**Output esperado**: `📈 Volatilidad anual de la cartera: 7.94%`

---

#### **1.3 Método EWMA (Exponentially Weighted Moving Average)**

> **Ventaja**: Asigna mayor peso a observaciones recientes, mejorando la reactividad del modelo  
> **Parámetro clave**: λ (lambda) = factor de decaimiento, típicamente 0.94 para datos diarios

##### **EWMA para Activo Individual**

```python
def ewma_volatility(returns: pd.Series, lambda_: float = 0.94) -> pd.Series:
    """
    Calcula volatilidad EWMA para un activo individual
    
    Parameters:
    -----------
    returns : pd.Series
        Serie de rendimientos diarios
    lambda_ : float
        Factor de decaimiento (default: 0.94, RiskMetrics standard)
    
    Returns:
    --------
    pd.Series : Volatilidad EWMA en el tiempo
    """
    ewma_var = returns.ewm(alpha=1 - lambda_).var()
    ewma_vol = np.sqrt(ewma_var)
    return ewma_vol

# Implementación
np.random.seed(42)
days = 252 * 3  
returns = np.random.normal(0, 0.01, days)
returns_series = pd.Series(returns, name='Daily_Returns')

# Volatilidad EWMA
ewma_vol = ewma_volatility(returns_series)
ewma_vol_annualized = ewma_vol * np.sqrt(252)
print(f'⚡ Volatilidad EWMA (último valor): {ewma_vol_annualized.iloc[-1]:.2%}')
```

**Output esperado**: `⚡ Volatilidad EWMA (último valor): 15.23%`

##### **EWMA para Cartera Multi-Activo**

```python
def ewma_covariance(returns: pd.DataFrame, lambda_: float = 0.94) -> pd.DataFrame:
    """
    Calcula matriz de covarianza EWMA para múltiples activos
    
    Parameters:
    -----------
    returns : pd.DataFrame
        DataFrame con rendimientos de múltiples activos
    lambda_ : float
        Factor de decaimiento EWMA
        
    Returns:
    --------
    pd.DataFrame : Matriz de covarianza EWMA (último período)
    """
    ewma_cov = returns.ewm(alpha=1 - lambda_).cov()
    # Extraer la última matriz de covarianza
    last_date = ewma_cov.index.get_level_values(0)[-1]
    return ewma_cov.xs(last_date, level=0)

# Implementación para cartera
ewma_cov_matrix = ewma_covariance(returns_df)
portfolio_ewma_volatility = np.sqrt(np.dot(weights.T, np.dot(ewma_cov_matrix.values, weights)))
portfolio_ewma_vol_annual = portfolio_ewma_volatility * np.sqrt(252)

print(f'⚡ Volatilidad EWMA de cartera: {portfolio_ewma_vol_annual:.2%}')
```

---

### 📉 **2. Value at Risk (VaR)**

El **VaR** cuantifica la pérdida máxima esperada con un nivel de confianza específico. Se implementa mediante tres metodologías principales:

#### **2.1 Método Paramétrico**

> **Supuesto**: Rendimientos siguen distribución normal  
> **Ventaja**: Cálculo rápido y eficiente  
> **Limitación**: Subestima riesgo de colas pesadas

##### **VaR Paramétrico - Activo Individual**

> **Fórmula**: $\text{VaR} = -(\mu + z_\alpha \cdot \sigma)$  
> **Donde**: $z_\alpha$ es el quantil de la distribución normal estándar

```python
from scipy.stats import norm

def calculate_parametric_var(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """
    Calcula VaR paramétrico asumiendo distribución normal
    
    Parameters:
    -----------
    returns : pd.Series
        Serie de rendimientos históricos
    confidence_level : float
        Nivel de confianza (default: 0.95 = 95%)
        
    Returns:
    --------
    float : VaR como proporción positiva
    """
    mean_return = returns.mean()
    std_dev = returns.std()
    z_score = norm.ppf(1 - confidence_level)  # Quantil crítico
    
    var_parametric = -(mean_return + z_score * std_dev)
    return var_parametric

# Implementación
np.random.seed(42)
days = 252 * 3
returns = np.random.normal(0, 0.01, days)
returns_series = pd.Series(returns, name='Daily_Returns')

# Cálculo de VaR a diferentes niveles de confianza
var_95 = calculate_parametric_var(returns_series, confidence_level=0.95)
var_99 = calculate_parametric_var(returns_series, confidence_level=0.99)

print(f'📊 VaR Paramétrico (95% confianza): {var_95:.2%}')
print(f'📊 VaR Paramétrico (99% confianza): {var_99:.2%}')
```

**Output esperado**:
```
📊 VaR Paramétrico (95% confianza): 1.64%
📊 VaR Paramétrico (99% confianza): 2.33%
```

##### **VaR Paramétrico - Cartera Multi-Activo**

> **Método**: Utiliza volatilidad de cartera calculada mediante matriz de covarianza

```python
def calculate_portfolio_parametric_var(returns_df: pd.DataFrame, 
                                     weights: np.ndarray, 
                                     confidence_level: float = 0.95) -> float:
    """
    Calcula VaR paramétrico para cartera de múltiples activos
    
    Parameters:
    -----------
    returns_df : pd.DataFrame
        Rendimientos históricos de los activos
    weights : np.ndarray
        Pesos de la cartera
    confidence_level : float
        Nivel de confianza
        
    Returns:
    --------
    float : VaR de la cartera
    """
    # Rendimientos ponderados de la cartera
    portfolio_returns = returns_df.dot(weights)
    
    # Estadísticas de la cartera
    mean_return = portfolio_returns.mean()
    
    # Volatilidad usando matriz de covarianza
    cov_matrix = returns_df.cov()
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix.values, weights)))
    
    # VaR paramétrico
    z_score = norm.ppf(1 - confidence_level)
    var_portfolio = -(mean_return + z_score * portfolio_volatility)
    
    return var_portfolio

# Implementación para cartera
portfolio_var_95 = calculate_portfolio_parametric_var(returns_df, weights, 0.95)
print(f'📈 VaR Cartera (95% confianza): {portfolio_var_95:.2%}')
```

**Output esperado**: `📈 VaR Cartera (95% confianza): 0.82%`


En ambos ejercicios anteriores, se puede ajustar el cálculo ultilizando EWMA, para esto, se debe utilizar la volatilidad calculada con EWMA en lugar de la volatilidad estándar.

#### Método Histórico
El VaR histórico se basa en datos históricos de rendimientos para estimar la pérdida máxima potencial. A continuación, se muestra un ejemplo de cómo calcular el VaR histórico para un activo financiero utilizando Python:
```python
import numpy as np
import pandas as pd

# Simular rendimientos de un activo financiero
np.random.seed(42)
days = 252 * 3  # 3 años de datos diarios
returns = np.random.normal(0, 0.01, days)  # Rendimientos
returns_series = pd.Series(returns)

# Parámetros del VaR
confidence_level = 0.95

# Calcular el VaR histórico
var_historical = -returns_series.quantile(1 - confidence_level)
print(f'📊 VaR histórico (95%): {var_historical:.2%}')
```

**Output esperado**: `📊 VaR histórico (95%): 1.68%`

Para el caso de una cartera de varios activos, se puede calcular el VaR histórico utilizando los rendimientos ponderados de la cartera:

```python
import numpy as np
import pandas as pd
# Simular rendimientos de 4 activos
num_assets = 4
returns_matrix = np.random.normal(0, 0.01, (days, num_assets))
returns_df = pd.DataFrame(returns_matrix, columns=[f'Asset_{i+1}' for i in range(num_assets)])  
# Pesos de la cartera
weights = np.array([0.25, 0.25, 0.25, 0.25])
# Calcular los rendimientos de la cartera
portfolio_returns = returns_df.dot(weights)
# Parámetros del VaR
confidence_level = 0.95
# Calcular el VaR histórico de la cartera
var_historical_portfolio = -portfolio_returns.quantile(1 - confidence_level)
print(f'📈 VaR histórico de la cartera (95%): {var_historical_portfolio:.2%}')
```

**Output esperado**: `📈 VaR histórico de la cartera (95%): 0.84%`


#### Método Simulación de Monte Carlo
El VaR mediante simulación de Monte Carlo implica generar múltiples escenarios de rendimientos futuros. Para lo anterior, simulamos muchos futuros posibles usando un modelo matemático de cómo se mueven los precios: el Browniano Geométrico (GBM).

##### **Movimiento Browniano Geométrico**
El GBM es un modelo simple y popular en finanzas. Supone tres cosas clave:

1. Los retornos logarítmicos son normales
Esto significa que el retorno de un día sigue algo parecido a una campana (normal), lo cual es una aproximación simplificada, pero útil.

2. La volatilidad es constante en el tiempo
El riesgo de cada activo no cambia durante el horizonte simulado.

3. Los precios nunca caen bajo cero
Porque el modelo trabaja con exponenciales (lo cual es razonable).

La fórmula del GBM es:

$$ S_1 = S_0 \cdot e^{(\mu - \frac{1}{2}\sigma^2)\,\Delta t + \sigma \sqrt{\Delta t}\, Z} $$

Donde:
- $( S_1 )$ es el precio simulado al final del período.
- $( S_0 )$ es el precio inicial.
- $(\mu )$ es el retorno esperado (drift).
- $(\sigma )$ es la volatilidad del activo.
- $(\Delta t)$ es el tamaño del paso de tiempo (por ejemplo, 1 día = 1/252 años).
- $(Z)$ es una variable aleatoria normal estándar (media 0, desviación estándar 1).

Intuitivamente, el término $(\mu - \frac{1}{2}\sigma^2)\,\Delta t$ representa el crecimiento esperado ajustado por la volatilidad, mientras que el término $\sigma \sqrt{\Delta t}\, Z$ introduce la aleatoriedad en los precios.

Para simular múltiples trayectorias de precios futuros de un solo activo financiero, hay que seguir los siguientes pasos:

1. Calcular los parámetros necesarios: retorno esperado $(\mu)$ y volatilidad $(\sigma)$ a partir de los datos históricos.

2. Generar un número Z que provenga de una distribución normal estándar.

3. Aplicar la fórmula del GBM para obtener el precio simulado al final del período.

4. Repetir los pasos 2 y 3 para generar múltiples simulaciones.

A continuación, se muestra un ejemplo de cómo implementar la simulación de Monte Carlo para calcular el VaR de un activo financiero utilizando Python:

```python
import numpy as np
import pandas as pd

# Simular rendimientos de un activo financiero
np.random.seed(42)
days = 252 * 3  # 3 años de datos diarios
returns = np.random.normal(0, 0.01, days)  # Rendimientos
returns_series = pd.Series(returns) 

# Parámetros del activo
S0 = 100  # Precio inicial
# Retorno anualizado
mu = returns_series.mean() * 252  
# Volatilidad anualizada   
sigma = returns_series.std() * np.sqrt(252)  

# Simulación de Monte Carlo
num_simulations = 10000
time_horizon = 1/252  # 1 día
simulated_prices = []   
for _ in range(num_simulations):
    Z = np.random.normal()
    S1 = S0 * np.exp((mu - 0.5 * sigma**2) * time_horizon + sigma * np.sqrt(time_horizon) * Z)
    simulated_prices.append(S1)
simulated_prices = np.array(simulated_prices)

# Calcular los rendimientos simulados
simulated_returns = (simulated_prices - S0) / S0

# Parámetros del VaR
confidence_level = 0.95

# Calcular el VaR mediante simulación de Monte Carlo
var_monte_carlo = -np.percentile(simulated_returns, (1 - confidence_level) * 100)
print(f'🎲 VaR Monte Carlo (95%): {var_monte_carlo:.2%}')
```

**Output esperado**: `🎲 VaR Monte Carlo (95%): 1.64%`


Al considerar una cartera de varios activos, hay que tener presente que los activos no se mueven de forma independiente, algunos suben juntos, otros se mueven en sentido contrario.

Para que las simulaciones sean realistas necesitamos que los shocks aleatorios $(\mu)$ de los activos estén correlacionados. Para lo anterior se utiliza la descomposición de Cholesky de la matriz de correlación de los activos, a partir de esta, se obtiene la matriz triangular inferior, la cual es la que se utliza para correlacionar los shocks aleatorios generados.

##### **Descomposición de Cholesky**
Imaginemos un portafolio de tres activos, del cual se puede calcular una matriz de correlación como la siguiente:

$$
\begin{bmatrix}
1 & 0.8 & 0.1 \\
0.8 & 1 & 0.2 \\
0.1 & 0.2 & 1
\end{bmatrix}
$$   

La descomposición de Cholesky consiste en descomponer esta matriz en el producto de una matriz triangular inferior y su transpuesta. 

$$
\begin{bmatrix}
1 & 0.8 & 0.1 \\
0.8 & 1 & 0.2 \\
0.1 & 0.2 & 1
\end{bmatrix} =
\begin{bmatrix}
1 & 0 & 0 \\
0.8 & 0.6 & 0 \\
0.1 & 0.18 & 0.98
\end{bmatrix}
\cdot
\begin{bmatrix}
1 & 0.8 & 0.1 \\
0 & 0.6 & 0.18 \\
0 & 0 & 0.98
\end{bmatrix}
$$

Lo anterior se resume como:

$$\Sigma  = L \cdot L^T $$

Donde $(L)$ es la matriz triangular inferior y $(\Sigma)$ es la matriz de correlación original.

Una vez obtenida la matriz $(L)$, se genera un vector de variables aleatorias normales estándar independientes $(z_1, z_2, ...,z_n)$. Donde el producto punto de la matriz $(L)$ y el vector de variables aleatorias independientes da como resultado un nuevo vector de variables aleatorias correlacionadas 


$$
Z =
\begin{bmatrix}
1 & 0 & 0 \\
0.8 & 0.6 & 0 \\
0.1 & 0.18 & 0.98
\end{bmatrix}
\cdot
\begin{bmatrix}
z_1 \\
z_2 \\
z_3
\end{bmatrix}
$$

Cada elemento del vector $(Z)$ representa un shock aleatorio correlacionado para cada activo en la cartera. Estos shocks se utilizan luego en la fórmula del Movimiento Browniano Geométrico para simular los precios futuros de cada activo, teniendo en cuenta la correlación entre ellos.

Así, la formula del GBM para cada activo $(i)$ en la cartera se ajusta de la siguiente manera:  

$$ S_{1,i} = S_{0,i} \cdot e^{(\mu_i - \frac{1}{2}\sigma_i^2)\,\Delta t + \sigma_i \sqrt{\Delta t}\, Z_i} $$

Donde:
- $( S_{1,i} )$ es el precio simulado del activo $(i)$ al final del período.
- $( S_{0,i} )$ es el precio inicial del activo $(i)$.
- $(\mu_i )$ es el retorno esperado del activo $(i)$.
- $(\sigma_i )$ es la volatilidad del activo $(i)$.
- $(Z_i)$ es el shock aleatorio correlacionado para el activo $(i)$.    

A continuación, se muestra un ejemplo de cómo implementar la simulación de Monte Carlo con descomposición de Cholesky para calcular el VaR de una cartera de varios activos utilizando Python:
```python
import numpy as np
import pandas as pd

# Simular rendimientos de 4 activos
num_assets = 4
returns_matrix = np.random.normal(0, 0.01, (days, num_assets))
returns_df = pd.DataFrame(returns_matrix, columns=[f'Asset_{i+1}' for i in range(num_assets)])

# Pesos de la cartera
weights = np.array([0.25, 0.25, 0.25, 0.25]) 

# Parámetros de los activos
S0 = np.array([100, 150, 200, 250])  
mu = returns_df.mean() * 252  
sigma = returns_df.std() * np.sqrt(252)

# Matriz de correlación y descomposición de Cholesky
correlation_matrix = returns_df.corr()
L = np.linalg.cholesky(correlation_matrix)

# Simulación de Monte Carlo con Cholesky
num_simulations = 10000
time_horizon = 1/252  # 1 día
simulated_portfolio_returns = []   
for _ in range(num_simulations):
    Z_independent = np.random.normal(size=num_assets)
    # Correlacionar los shocks
    Z_correlated = L @ Z_independent  
    S1 = S0 * np.exp((mu - 0.5 * sigma**2) * time_horizon + sigma * np.sqrt(time_horizon) * Z_correlated)   
    portfolio_return = np.dot(weights, (S1 - S0) / S0)
    simulated_portfolio_returns.append(portfolio_return)
simulated_portfolio_returns = np.array(simulated_portfolio_returns)     

# Parámetros del VaR
confidence_level = 0.95 

# Calcular el VaR mediante simulación de Monte Carlo para la cartera
var_monte_carlo_portfolio = -np.percentile(simulated_portfolio_returns, (1 - confidence_level) * 100)
print(f'🃈 VaR Monte Carlo de la cartera (95%): {var_monte_carlo_portfolio:.2%}')
```

**Output esperado**: `🃈 VaR Monte Carlo de la cartera (95%): 0.81%`

### Cálculo Tracking Error

#### Tracking Error Expost
El Tracking Error Expost mide la desviación estándar de las diferencias entre los rendimientos de una cartera y su índice de referencia durante un período pasado. A continuación, se muestra un ejemplo de cómo calcular el Tracking Error Expost utilizando Python:   

```python
import numpy as np  
import pandas as pd

# Simular rendimientos de 4 activos y su benchmark
num_assets = 4  
returns_matrix = np.random.normal(0, 0.01, (days, num_assets))
returns_df = pd.DataFrame(returns_matrix, columns=[f'Asset_{i+1}' for i in range(num_assets)])  
benchmark_returns = np.random.normal(0, 0.008, days)  # Rendimientos del benchmark
benchmark_returns_series = pd.Series(benchmark_returns)

# Pesos de la cartera
weights = np.array([0.25, 0.25, 0.25, 0.25])

# Calcular los rendimientos de la cartera
portfolio_returns = returns_df.dot(weights) 

# Calcular las diferencias de rendimiento
return_differences = portfolio_returns - benchmark_returns_series   

# Calcular el Tracking Error Expost de la cartera
tracking_error_expost_portfolio = return_differences.std() * np.sqrt(252)
print(f'📏 Tracking Error Ex-post (anualizado): {tracking_error_expost_portfolio:.2%}')
```

**Output esperado**: `📏 Tracking Error Ex-post (anualizado): 3.15%`

#### Tracking Error Exante
El Tracking Error Exante estima la desviación estándar de las diferencias entre los rendimientos esperados de una cartera y su índice de referencia utilizando la matriz de covarianza de los activos en la cartera. A continuación, se muestra un ejemplo de cómo calcular el Tracking Error Exante utilizando Python:

```python
import numpy as np
import pandas as pd

# Simular rendimientos de 4 activos
num_assets = 4
returns_matrix = np.random.normal(0, 0.01, (days, num_assets))
returns_df = pd.DataFrame(returns_matrix, columns=[f'Asset_{i+1}' for i in range(num_assets)])

# Pesos de la cartera
weights = np.array([0.25, 0.25, 0.25, 0.25])

# Calcular la matriz de covarianza
cov_matrix = returns_df.cov()

# Calcular el Tracking Error Exante de la cartera

tracking_error_exante_portfolio = np.sqrt(np.dot(weights.T, np.dot(cov_matrix.values, weights))) * np.sqrt(252)
print(f'📏 Tracking Error Ex-ante (anualizado): {tracking_error_exante_portfolio:.2%}')
```

**Output esperado**: `📏 Tracking Error Ex-ante (anualizado): 7.94%`


### Cálculo Beta de Mercado
El beta de mercado se calcula mediante la regresión lineal de los rendimientos de un activo o cartera contra los rendimientos del mercado, por métdo de los mínimos cuadrados ordinarios (OLS). A continuación, se muestra un ejemplo de cómo calcular el beta de mercado utilizando Python:

La siguiente fórmula representa la relación lineal entre los rendimientos del activo $(R_i)$ y los rendimientos del mercado $(R_m)$:    

$$ R_i = \alpha + \beta R_m + \epsilon $$

Donde:
- $(R_i)$ es el rendimiento del activo o cartera.
- $(R_m)$ es el rendimiento del mercado.
- $(\alpha)$ es la intersección de la regresión.
- $(\beta)$ es el coeficiente que mide la sensibilidad del activo o cartera a los
rendimientos del mercado.
- $(\epsilon)$ es el término de error.

Es esta relación lineal de la que inspira el modelo CAPM (Capital Asset Pricing Model), el cual establece que el rendimiento esperado de un activo o cartera está relacionado con su beta de la siguiente manera:

$$ E(R_i) = \alpha + \beta (E(R_m) - R_f) +  \epsilon  $$

Donde:
- $(E(R_i))$ es el rendimiento esperado del activo.
- $(E(R_m))$ es el rendimiento esperado del mercado.
- $(R_f)$ es la tasa libre de riesgo.
- $(\beta)$ es el riesgo sistemático del activo.
- $(\alpha)$ es el riesgo no sistemático del activo.
- $(\epsilon)$ retorno no explicado.

Cuando se genenera una cartera de activos financieros, a medida que esta se diversifica, el riesgo no sistemático $(\alpha)$ y el retorno no explicaco $(\epsilon)$ tiende a reducirse, dejando al beta $(\beta)$ como la principal medida del riesgo asociado a la cartera en relación con el mercado.

La formula principql para calcular el beta de mercado es:

$$ \beta = \frac{Cov(R_i, R_m)}{Var(R_m)} $$

Donde:
- $(Cov(R_i, R_m))$ es la covarianza entre los rendimientos
del activo o cartera y los rendimientos del mercado.
- $(Var(R_m))$ es la varianza de los rendimientos del mercado.


Otra forma de estimar el beta es:

$$ \beta = \rho_{i,m} \cdot \frac{\sigma_i}{\sigma_m} $$

Donde:
- $(\rho_{i,m})$ es el coeficiente de correlación entre los rendimientos del activo o cartera y los rendimientos del mercado.
- $(\sigma_i)$ es la desviación estándar de los rendimientos del activo o cartera
- $(\sigma_m)$ es la desviación estándar de los rendimientos del mercado.


Para calcular el beta de mercado de un activo financiero utilizando Python, se puede seguir el siguiente ejemplo:

```python
import numpy as np
import pandas as pd
from scipy.stats import linregress

# Simular rendimientos de un activo financiero y del mercado
np.random.seed(42)
days = 252 * 3  # 3 años de datos diarios
asset_returns = np.random.normal(0, 0.01, days)  # Rendimientos del activo
market_returns = np.random.normal(0, 0.008, days)  # Rendimientos del mercado
asset_returns_series = pd.Series(asset_returns)
market_returns_series = pd.Series(market_returns)

# Calcular el beta mediante regresión lineal
slope, intercept, r_value, p_value, std_err = linregress(market_returns_series, asset_returns_series)
beta = slope
print(f'🎯 Beta del activo: {beta:.4f}')
```

**Output esperado**: `🎯 Beta del activo: 0.0344`


Para el caso de una cartera de varios activos, se puede calcular el beta de mercado utilizando los rendimientos ponderados de la cartera:

```python
import numpy as np
import pandas as pd

# Simular rendimientos de 4 activos y del mercado
num_assets = 4
returns_matrix = np.random.normal(0, 0.01, (days, num_assets))
returns_df = pd.DataFrame(returns_matrix, columns=[f'Asset_{i+1}' for i in range(num_assets)])
market_returns = np.random.normal(0, 0.008, days)  # Rendimientos del mercado
market_returns_series = pd.Series(market_returns)

# Pesos de la cartera
weights = np.array([0.25, 0.25, 0.25, 0.25])

# Calcular los rendimientos de la cartera
portfolio_returns = returns_df.dot(weights)

# Calcular el beta de la cartera mediante regresión lineal
slope, intercept, r_value, p_value, std_err = linregress(market_returns_series, portfolio_returns)
beta_portfolio = slope
print(f'🃈 Beta de la cartera: {beta_portfolio:.4f}')
```

**Output esperado**: `🃈 Beta de la cartera: 0.0267`


#### Cálculo Ratio de Sharpe
El Ratio de Sharpe mide el rendimiento ajustado al riesgo de una inversión. A continuación, se muestra un ejemplo de cómo calcular el Ratio de Sharpe utilizando Python:

```python
import numpy as np
import pandas as pd

# Simular rendimientos de un activo financiero
np.random.seed(42)
days = 252 * 3  # 3 años de datos diarios
returns = np.random.normal(0, 0.01, days)  # Rendimientos
returns_series = pd.Series(returns)

# Parámetros del Ratio de Sharpe
risk_free_rate = 0.01  # Tasa libre de riesgo anualizada    

# Calcular el exceso de retorno anualizado
excess_return = returns_series.mean() * 252 - risk_free_rate

# Calcular la volatilidad anualizada
volatility = returns_series.std() * np.sqrt(252)

# Calcular el Ratio de Sharpe
sharpe_ratio = excess_return / volatility
print(f'📈 Ratio de Sharpe: {sharpe_ratio:.4f}')
```

**Output esperado**: `📈 Ratio de Sharpe: -0.0630`


Para el caso de una cartera de varios activos, se puede calcular el Ratio de Sharpe utilizando los rendimientos ponderados de la cartera:

```python
import numpy as np
import pandas as pd

# Simular rendimientos de 4 activos
num_assets = 4
returns_matrix = np.random.normal(0, 0.01, (days, num_assets))
returns_df = pd.DataFrame(returns_matrix, columns=[f'Asset_{i+1}' for i in range(num_assets)])  

# Pesos de la cartera
weights = np.array([0.25, 0.25, 0.25, 0.25])

# Calcular los rendimientos de la cartera
portfolio_returns = returns_df.dot(weights)

# Parámetros del Ratio de Sharpe
risk_free_rate = 0.01  # Tasa libre de riesgo anualizada    

# Calcular el exceso de retorno anualizado de la cartera
excess_return_portfolio = portfolio_returns.mean() * 252 - risk_free_rate

# Calcular la volatilidad anualizada de la cartera
volatility_portfolio = portfolio_returns.std() * np.sqrt(252)

# Calcular el Ratio de Sharpe de la cartera
sharpe_ratio_portfolio = excess_return_portfolio / volatility_portfolio
print(f'📈 Ratio de Sharpe de la cartera: {sharpe_ratio_portfolio:.4f}')
```

**Output esperado**: `📈 Ratio de Sharpe de la cartera: -0.1260`


#### Cálculo Ratio de Sortino
El Ratio de Sortino mide el rendimiento ajustado al riesgo, considerando solo la volatilidad negativa. A continuación, se muestra un ejemplo de cómo calcular el Ratio de Sortino utilizando Python:

```python
import numpy as np
import pandas as pd

# Simular rendimientos de un activo financiero
np.random.seed(42)
days = 252 * 3  # 3 años de datos diarios
returns = np.random.normal(0, 0.01, days)  # Rendimientos
returns_series = pd.Series(returns)

# Parámetros del Ratio de Sortino
risk_free_rate = 0.01  # Tasa libre de riesgo anualizada    

# Calcular el exceso de retorno anualizado
excess_return = returns_series.mean() * 252 - risk_free_rate

# Calcular la desviación estándar de los rendimientos negativos
downside_returns = returns_series[returns_series < 0]
downside_deviation = downside_returns.std() * np.sqrt(252)

# Calcular el Ratio de Sortino
sortino_ratio = excess_return / downside_deviation
print(f'📈 Ratio de Sortino: {sortino_ratio:.4f}')
```

**Output esperado**: `📈 Ratio de Sortino: -0.0891`


Para el caso de una cartera de varios activos, se puede calcular el Ratio de Sortino utilizando los rendimientos ponderados de la cartera:

```python
import numpy as np
import pandas as pd

# Simular rendimientos de 4 activos
num_assets = 4
returns_matrix = np.random.normal(0, 0.01, (days, num_assets))
returns_df = pd.DataFrame(returns_matrix, columns=[f'Asset_{i+1}' for i in range(num_assets)])  

# Pesos de la cartera
weights = np.array([0.25, 0.25, 0.25, 0.25])

# Calcular los rendimientos de la cartera
portfolio_returns = returns_df.dot(weights)

# Parámetros del Ratio de Sortino
risk_free_rate = 0.01  # Tasa libre de riesgo anualizada    

# Calcular el exceso de retorno anualizado de la cartera
excess_return_portfolio = portfolio_returns.mean() * 252 - risk_free_rate

# Calcular la desviación estándar de los rendimientos negativos de la cartera

downside_returns_portfolio = portfolio_returns[portfolio_returns < 0]
downside_deviation_portfolio = downside_returns_portfolio.std() * np.sqrt(252)  

# Calcular el Ratio de Sortino de la cartera
sortino_ratio_portfolio = excess_return_portfolio / downside_deviation_portfolio
print(f'📈 Ratio de Sortino de la cartera: {sortino_ratio_portfolio:.4f}')
```

**Output esperado**: `📈 Ratio de Sortino de la cartera: -0.1783`


#### Cálculo Ratio de Treynor
El Ratio de Treynor mide el rendimiento ajustado al riesgo sistemático. A continuación, se muestra un ejemplo de cómo calcular el Ratio de Treynor utilizando Python:

```python
import numpy as np
import pandas as pd
from scipy.stats import linregress

# Simular rendimientos de un activo financiero y del mercado
np.random.seed(42)
days = 252 * 3  # 3 años de datos diarios
asset_returns = np.random.normal(0, 0.01, days)  # Rendimientos del activo
market_returns = np.random.normal(0, 0.008, days)  # Rendimientos del mercado
asset_returns_series = pd.Series(asset_returns)
market_returns_series = pd.Series(market_returns)

# Calcular el beta mediante regresión lineal
slope, intercept, r_value, p_value, std_err = linregress(market_returns_series, asset_returns_series)
beta = slope    

# Parámetros del Ratio de Treynor
risk_free_rate = 0.01  # Tasa libre de riesgo anualizada    

# Calcular el exceso de retorno anualizado
excess_return = asset_returns_series.mean() * 252 - risk_free_rate

# Calcular el Ratio de Treynor
treynor_ratio = excess_return / beta
print(f'📈 Ratio de Treynor: {treynor_ratio:.4f}')
```

**Output esperado**: `📈 Ratio de Treynor: -0.2907`

Para el caso de una cartera de varios activos, se puede calcular el Ratio de Treynor utilizando los rendimientos ponderados de la cartera:

```python
import numpy as np
import pandas as pd
from scipy.stats import linregress

# Simular rendimientos de 4 activos y del mercado
num_assets = 4
returns_matrix = np.random.normal(0, 0.01, (days, num_assets))
returns_df = pd.DataFrame(returns_matrix, columns=[f'Asset_{i+1}' for i in range(num_assets)])
market_returns = np.random.normal(0, 0.008, days)  # Rendimientos del mercado
market_returns_series = pd.Series(market_returns)

# Pesos de la cartera
weights = np.array([0.25, 0.25, 0.25, 0.25])

# Calcular los rendimientos de la cartera
portfolio_returns = returns_df.dot(weights)

# Calcular el beta de la cartera mediante regresión lineal
slope, intercept, r_value, p_value, std_err = linregress(market_returns_series, portfolio_returns)
beta_portfolio = slope  

# Parámetros del Ratio de Treynor
risk_free_rate = 0.01  # Tasa libre de riesgo anualizada

# Calcular el exceso de retorno anualizado de la cartera
excess_return_portfolio = portfolio_returns.mean() * 252 - risk_free_rate   

# Calcular el Ratio de Treynor de la cartera
treynor_ratio_portfolio = excess_return_portfolio / beta_portfolio
print(f'📈 Ratio de Treynor de la cartera: {treynor_ratio_portfolio:.4f}')
```

**Output esperado**: `📈 Ratio de Treynor de la cartera: -0.3745`

---

<br>

## 📊 **Resumen de Métricas Implementadas**

### 🎯 **Tabla Comparativa de Métricas de Riesgo**

| Métrica | Tipo | Fórmula | Interpretación | Ventajas | Limitaciones |
|---------|------|---------|----------------|----------|-------------|
| **Volatilidad** | Dispersión | $\sigma = \sqrt{\text{Var}(R)}$ | Mayor valor = mayor riesgo | Simple, intuitiva | No captura direccionalidad |
| **VaR** | Pérdida máxima | $P(L > \text{VaR}) = \alpha$ | Pérdida con probabilidad α | Regulatorio, comparable | No informa sobre colas |
| **Beta** | Sensibilidad | $\beta = \frac{\text{Cov}(R_i, R_m)}{\text{Var}(R_m)}$ | Riesgo sistemático | Relación con mercado | Solo riesgo sistemático |
| **Sharpe** | Eficiencia | $\frac{R_p - R_f}{\sigma_p}$ | Rendimiento/riesgo | Ajuste por riesgo total | Asume normalidad |
| **Sortino** | Eficiencia | $\frac{R_p - R_f}{\sigma_{\text{downside}}}$ | Penaliza solo volatilidad negativa | Enfoque en pérdidas | Requiere más datos |
| **Treynor** | Eficiencia | $\frac{R_p - R_f}{\beta_p}$ | Rendimiento/riesgo sistemático | Ajuste por riesgo de mercado | Ignora riesgo específico |

