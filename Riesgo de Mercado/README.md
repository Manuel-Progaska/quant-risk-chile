# Riesgo de Mercado con Python

## ¿Qué es el Riesgo de Mercado?
El riesgo de mercado se refiere a la posibilidad de que el valor de una inversión disminuya debido a cambios en las condiciones del mercado financiero. Este tipo de riesgo afecta a todos los activos financieros, incluyendo acciones, bonos, divisas y derivados. Los factores que contribuyen al riesgo de mercado incluyen fluctuaciones en las tasas de interés, cambios en los precios de los activos, volatilidad del mercado y eventos económicos o políticos.

## Métricas de Riesgo de Mercado
Para cuantificar el riesgo de mercado, se utilizan diversas métricas. A continuación, se describen algunas de las más comunes:
### Volatilidad
La volatilidad mide la variabilidad de los rendimientos de un activo financiero. Se calcula comúnmente como la desviación estándar de los rendimientos históricos. Una mayor volatilidad indica un mayor riesgo, ya que los precios del activo pueden fluctuar significativamente en un corto período de tiempo.    
### Value at Risk (VaR)
El Value at Risk (VaR) es una medida estadística que estima la pérdida máxima potencial de una cartera de inversión durante un período específico, con un nivel de confianza determinado. Por ejemplo, un VaR del 5% a un día indica que hay un 5% de probabilidad de que la cartera pierda más de una cantidad específica en un solo día.    
### Tracking Error
El Tracking Error mide la desviación estándar de las diferencias entre los rendimientos de una cartera y su índice de referencia. Esta métrica es útil para evaluar el desempeño de una cartera en comparación con un benchmark, y un menor Tracking Error indica una mayor similitud en el comportamiento entre la cartera y el índice de referencia.  

## Implementación en Python
A continuación, se presentan ejemplos de cómo calcular estas métricas utilizando Python.

### Cálculo de Volatilidad
Para estimar la volatilidad de un activo financiero, se utiliza como inidcador la desviación estandar de los rendimientos histórico.

A continuación, se muestra un ejemplo de cómo calcular la volatilidad simulando 3 años de rendimientos diarios de un activo financiero:

```python
import numpy as np
import pandas as pd

# Generar datos de ejemplo
np.random.seed(42)
days = 252 * 3  # 3 años de datos diarios
returns = np.random.normal(0, 0.01, days)  # Rendimientos
returns_series = pd.Series(returns)

# Calcular volatilidad anualizada
volatility = returns_series.std() * np.sqrt(252)
print(f'Volatilidad anualizada: {volatility:.2%}')
```

Si en vez de tener un activo financiero, tenemos una cartera de varios activos, podemos calcular la volatilidad de la cartera utilizando la matriz de covarianza de los rendimientos de los activos y los pesos de la cartera:

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

# Calcular la volatilidad de la cartera
portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot
portfolio_volatility_annualized = portfolio_volatility * np.sqrt(252)
print(f'Volatilidad anual de la cartera: {portfolio_volatility_annualized:.2%}')
```

#### Ajuste con EWMA
Una forma de mejorar la estimación de la volatilidad es utilizando el método EWMA (Exponentially Weighted Moving Average), que asigna más peso a los datos recientes. A continuación, se muestra cómo implementar este método en Python para un solo activo financiero:  

```python
import numpy as np
import pandas as pd

# Función para calcular la volatilidad con EWMA
def ewma_volatility(returns:pd.DataFrame, lambda_:float=0.94) -> pd.Series:
    ewma_var = returns.ewm(alpha=1 - lambda_).var()
    ewma_vol = np.sqrt(ewma_var)
    return ewma_vol 

# Calcular volatilidad con EWMA
ewma_vol = ewma_volatility(returns_series)
ewma_vol_annualized = ewma_vol * np.sqrt(252)
print(f'Volatilidad anualizada con EWMA: {ewma_vol_annualized[-1]:.2%}')
```




