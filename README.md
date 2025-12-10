<div align="center">

# 📊 Finanzas Cuantitativas Chile 
### *Herramientas y metodologías para la gestión de riesgos financieros*

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Scipy](https://img.shields.io/badge/Scipy-013243?style=for-the-badge&logo=scipy&logoColor=white)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Contributors Welcome](https://img.shields.io/badge/Contributors-Welcome-brightgreen.svg)](CONTRIBUTING.md)

</div>

---

## 🎯 Descripción

Este repositorio contiene una **colección completa de documentos y recursos** relacionados con las finanzas cuantitativas utilizando Python. Aquí encontrarás guías detalladas sobre diversas metodologías y técnicas empleadas en la gestión de riesgos financieros, análisis de carteras y modelado estadístico.

> 💡 **Objetivo**: Proporcionar herramientas prácticas y teóricas para profesionales del área financiera que buscan implementar soluciones cuantitativas robustas.

---

## 📚 Contenido

### **Riesgo de Mercado**

<details>
<summary><b>📈 Fundamentos y Metodologías Principales</b></summary>

#### 🎯 Conceptos Fundamentales
- **[¿Qué es el riesgo de mercado?](Riesgo%20de%20Mercado/README.md#¿qué-es-el-riesgo-de-mercado)** - Introducción conceptual


#### 📊 Métricas para medir el riesgo de mercado
| Métrica | Descripción | Implementación |
|---------|-------------|----------------|
| **🌊 Volatilidad** | Variabilidad de rendimientos históricos | [Ver guía](Riesgo%20de%20Mercado/README.md#volatilidad) |
| **⚠️ Value at Risk (VaR)** | Pérdida máxima esperada con confianza estadística | [Ver guía](Riesgo%20de%20Mercado/README.md#value-at-risk) |
| **📏 Tracking Error** | Desviación estándar vs benchmark | [Ver guía](Riesgo%20de%20Mercado/README.md#tracking-error) |
| **📈 Beta de Mercado** | Sensibilidad al mercado (riesgo sistemático) | [Ver guía](Riesgo%20de%20Mercado/README.md#beta-de-mercado) |

</details>


<details>
<summary><b>🐍 Herramientas y Códigos Prácticos</b></summary>

#### 🌊 **Cálculo de Volatilidad**
- 📊 **[Volatilidad de Activo Individual](Riesgo%20de%20Mercado/README.md#cálculo-de-volatilidad)** - Desviación estándar histórica
- 📈 **[Volatilidad de Cartera](Riesgo%20de%20Mercado/README.md#cálculo-de-volatilidad)** - Usando matriz de covarianza
- ⚡ **[EWMA (Exponentially Weighted Moving Average)](Riesgo%20de%20Mercado/README.md#ajuste-con-ewma)** - Volatilidad adaptativa
  - 🔄 Implementación para activos individuales
  - 📊 Matriz de covarianza EWMA para carteras

#### ⚠️ **Value at Risk (VaR)**
- 📐 **[Método Paramétrico](Riesgo%20de%20Mercado/README.md#método-paramétrico)**
  - 📈 VaR para activos individuales
  - 📊 VaR para carteras de múltiples activos
  - ⚡ Integración con EWMA
- 📋 **[Método Histórico](Riesgo%20de%20Mercado/README.md#método-histórico)**
  - 📈 Simulación histórica simple
  - 📊 VaR histórico para carteras ponderadas
- 🎲 **[Simulación Monte Carlo](Riesgo%20de%20Mercado/README.md#método-simulación-de-monte-carlo)**
  - 🌱 **[Movimiento Browniano Geométrico](Riesgo%20de%20Mercado/README.md#movimiento-browniano-geométrico)** - Modelo matemático completo
  - 🔗 **[Descomposición de Cholesky](Riesgo%20de%20Mercado/README.md#descomposición-de-cholesky)** - Correlación entre activos
  - 📊 Implementación para carteras correlacionadas

#### 📏 **Tracking Error**
- 📈 **[Tracking Error Ex-post](Riesgo%20de%20Mercado/README.md#tracking-error-expost)** - Análisis histórico vs benchmark
- 🔮 **[Tracking Error Ex-ante](Riesgo%20de%20Mercado/README.md#tracking-error-exante)** - Estimación prospectiva

#### 📈 **Beta de Mercado**
- 🔍 **[Fundamentos del Beta](Riesgo%20de%20Mercado/README.md#cálculo-beta-de-mercado)** - Teoría CAPM
- 📊 **[Cálculo para Activos Individuales](Riesgo%20de%20Mercado/README.md#cálculo-beta-de-mercado)** - Regresión lineal
- 📈 **[Beta de Cartera](Riesgo%20de%20Mercado/README.md#cálculo-beta-de-mercado)** - Rendimientos ponderados
- 🔢 **Fórmulas alternativas**: Covarianza y correlación

</details>

---

### 🔥 **Algoritmos Implementados**

- 🔄 **EWMA**: Volatilidad adaptativa con decaimiento exponencial
- 🎯 **VaR Paramétrico**: Distribución normal y matriz de covarianza
- 📊 **VaR Histórico**: Percentiles de distribución empírica
- 🎲 **Monte Carlo**: GBM con descomposición de Cholesky
- 📏 **Tracking Error**: Ex-post y Ex-ante para benchmarks
- 📈 **Beta de Mercado**: CAPM y análisis de sensibilidad

## 🤝 Contribuir

¡Las contribuciones son bienvenidas! Por favor:

1. 🍴 **Fork** el proyecto
2. 🌱 Crea una **rama** para tu feature (`git checkout -b feature/AmazingFeature`)
3. 💾 **Commit** tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. 📤 **Push** a la rama (`git push origin feature/AmazingFeature`)
5. 📋 Abre un **Pull Request**

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para detalles.

## 📞 Contacto

**Manuel Progaska** - [Linkedin](https://www.linkedin.com/in/manuel-progaska-concha-98b304135/)


🔗 **Link del Proyecto**: [https://github.com/Manuel-Progaska/quant-risk-chile](https://github.com/Manuel-Progaska/quant-risk-chile)

---

<div align="center">

### 💫 ¡Gracias por visitar este repositorio!

Si te resulta útil, ¡no olvides darle una ⭐!

</div>
