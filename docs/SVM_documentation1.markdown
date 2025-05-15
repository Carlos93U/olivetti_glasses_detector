# Documentación del Algoritmo Support Vector Machine (SVM) - Clasificador SVC

## Introducción

El **Support Vector Machine (SVM)** es un algoritmo de aprendizaje supervisado utilizado para tareas de clasificación y regresión, aunque es más comúnmente empleado en problemas de clasificación. En este documento, nos enfocaremos en el **Support Vector Classification (SVC)**, que es la implementación de SVM para clasificación en la biblioteca scikit-learn.

El SVM busca encontrar el hiperplano óptimo que separa las clases en un espacio de características, maximizando el margen entre las clases. Este enfoque lo hace robusto frente al sobreajuste, especialmente en datasets de alta dimensionalidad. Además, SVM puede manejar problemas no linealmente separables mediante el uso del **truco del kernel**.

## Fundamento Matemático

### 1. Concepto Básico: Hiperplano y Margen

En un problema de clasificación binaria, el SVM busca un hiperplano que separe las dos clases. En un espacio de \( n \)-dimensiones, un hiperplano se define como:

\[
\mathbf{w}^T \mathbf{x} + b = 0
\]

Donde:
- \(\mathbf{w}\) es el vector normal al hiperplano.
- \(\mathbf{x}\) es un punto en el espacio de características.
- \(b\) es el término de sesgo (bias).

El objetivo del SVM es encontrar el hiperplano que maximice el **margen** entre las clases. El margen se define como la distancia entre el hiperplano y los puntos más cercanos de cada clase, conocidos como **vectores de soporte**.

<div align="center">
    <img src="https://upload.wikimedia.org/wikipedia/commons/7/72/SVM_margin.png" alt="SVM Margin" style="width:60%;">
    <p>Figura 1: Ilustración del hiperplano que maximiza el margen entre dos clases. Los vectores de soporte son los puntos más cercanos al hiperplano (Fuente: Wikimedia Commons).</p>
</div>

La distancia de un punto \(\mathbf{x}_i\) al hiperplano se calcula como:

\[
\text{Distancia} = \frac{|\mathbf{w}^T \mathbf{x}_i + b|}{||\mathbf{w}||}
\]

Para maximizar el margen, se busca maximizar \(\frac{2}{||\mathbf{w}||}\), lo que equivale a minimizar \(\frac{1}{2}||\mathbf{w}||^2\), sujeto a la restricción de que los puntos estén correctamente clasificados:

\[
y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1, \quad \forall i
\]

Donde \(y_i \in \{-1, 1\}\) es la etiqueta de la clase para el punto \(\mathbf{x}_i\).

### 2. Soft Margin y Parámetro \(C\)

En casos reales, los datos no siempre son perfectamente separables. Para manejar esto, SVM introduce el concepto de **soft margin**, que permite cierta tolerancia a errores de clasificación. Esto se logra introduciendo variables de holgura (\(\xi_i\)) y un parámetro de regularización \(C\):

\[
\text{Minimizar: } \frac{1}{2}||\mathbf{w}||^2 + C \sum_{i=1}^m \xi_i
\]

Sujeto a:

\[
y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0, \quad \forall i
\]

El parámetro \(C\) controla el balance entre maximizar el margen y minimizar los errores de clasificación:
- Un \(C\) grande penaliza más los errores, resultando en un margen más estrecho.
- Un \(C\) pequeño permite más errores, favoreciendo un margen más amplio.

<div align="center">
    <img src="https://miro.medium.com/max/1400/1*0vOVPBmYCkw-sUtA31QVw.png" alt="Soft Margin SVM" style="width:60%;">
    <p>Figura 2: Comparación entre un margen duro y un margen blando (soft margin). El parámetro \(C\) ajusta la tolerancia a errores (Fuente: Medium).</p>
</div>

### 3. Truco del Kernel

Cuando los datos no son linealmente separables en el espacio original, SVM utiliza el **truco del kernel** para mapear los datos a un espacio de mayor dimensionalidad donde sí puedan ser separados linealmente. Esto se hace sin calcular explícitamente las coordenadas en el nuevo espacio, gracias a la función kernel.

La función de decisión se convierte en:

\[
f(\mathbf{x}) = \sum_{i=1}^m \alpha_i y_i K(\mathbf{x}_i, \mathbf{x}) + b
\]

Donde:
- \(K(\mathbf{x}_i, \mathbf{x})\) es la función kernel.
- \(\alpha_i\) son los multiplicadores de Lagrange obtenidos al resolver el problema dual.

Algunos kernels comunes incluyen:
- **Lineal**: \(K(\mathbf{x}_i, \mathbf{x}_j) = \mathbf{x}_i^T \mathbf{x}_j\)
- **Polinómico**: \(K(\mathbf{x}_i, \mathbf{x}_j) = (\gamma \mathbf{x}_i^T \mathbf{x}_j + r)^d\)
- **RBF (Radial Basis Function)**: \(K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma ||\mathbf{x}_i - \mathbf{x}_j||^2)\)

El kernel RBF es particularmente popular porque puede manejar separaciones no lineales complejas.

<div align="center">
    <img src="https://www.researchgate.net/publication/341107838/figure/fig2/AS:886847599087616@1590330711680/The-effect-of-the-kernel-trick-in-SVM-The-left-image-shows-the-data-in-the-original.png" alt="Kernel Trick" style="width:60%;">
    <p>Figura 3: Efecto del truco del kernel. Los datos no separables linealmente en el espacio original (izquierda) se mapean a un espacio donde sí lo son (derecha) (Fuente: ResearchGate).</p>
</div>

### 4. Optimización del Problema Dual

El SVM se resuelve típicamente en su forma dual, ya que es más eficiente computacionalmente y permite el uso de kernels. El problema dual se formula como:

\[
\text{Maximizar: } \sum_{i=1}^m \alpha_i - \frac{1}{2} \sum_{i=1}^m \sum_{j=1}^m \alpha_i \alpha_j y_i y_j K(\mathbf{x}_i, \mathbf{x}_j)
\]

Sujeto a:

\[
\sum_{i=1}^m \alpha_i y_i = 0, \quad 0 \leq \alpha_i \leq C, \quad \forall i
\]

Los \(\alpha_i\) determinan los vectores de soporte (\(\alpha_i > 0\)) y se utilizan para calcular \(\mathbf{w}\) y \(b\).

## Implementación en Scikit-learn: Ejemplo de Código

Scikit-learn proporciona una implementación eficiente de SVC a través de la clase `sklearn.svm.SVC`. A continuación, se muestra un ejemplo básico tomado de la [documentación oficial de scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html):

### Ejemplo 1: Clasificación con SVC y Kernel Lineal

```python
from sklearn import svm
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Generar datos sintéticos
X, y = make_blobs(n_samples=100, centers=2, random_state=42)

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear y entrenar el modelo SVC con kernel lineal
svc = svm.SVC(kernel='linear', C=1.0)
svc.fit(X_train, y_train)

# Predecir y evaluar
y_pred = svc.predict(X_test)
print("Reporte de clasificación:")
print(classification_report(y_test, y_pred))

# Visualizar el hiperplano y los vectores de soporte
plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap='autumn')
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# Crear una malla para graficar el hiperplano
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = svc.decision_function(xy).reshape(XX.shape)

# Graficar el hiperplano y los márgenes
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
ax.scatter(svc.support_vectors_[:, 0], svc.support_vectors_[:, 1], s=100, linewidth=1, facecolors='none', edgecolors='k')
plt.title("Hiperplano y Vectores de Soporte con Kernel Lineal")
plt.show()
```

Este código genera datos sintéticos, entrena un SVC con kernel lineal y visualiza el hiperplano de separación junto con los vectores de soporte.

### Ejemplo 2: Clasificación con Kernel RBF

```python
from sklearn import svm
from sklearn.datasets import make_moons

# Generar datos no lineales
X, y = make_moons(n_samples=100, noise=0.1, random_state=42)

# Dividir los datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear y entrenar el modelo SVC con kernel RBF
svc_rbf = svm.SVC(kernel='rbf', C=1.0, gamma='scale')
svc_rbf.fit(X_train, y_train)

# Predecir y evaluar
y_pred = svc_rbf.predict(X_test)
print("Reporte de clasificación con kernel RBF:")
print(classification_report(y_test, y_pred))
```

Este ejemplo utiliza el kernel RBF para clasificar datos no lineales generados con `make_moons`.

## Ventajas y Desventajas del SVC

### Ventajas
- Efectivo en espacios de alta dimensionalidad, como imágenes o texto.
- Robusto frente al sobreajuste gracias a la maximización del margen.
- Flexible gracias al uso de kernels para problemas no lineales.

### Desventajas
- Computacionalmente costoso para datasets muy grandes.
- Sensible a la elección de hiperparámetros (\(C\), \(\gamma\), tipo de kernel).
- No proporciona probabilidades de clasificación de forma directa (a menos que se use `probability=True`).

## Referencias Bibliográficas

1. Cortes, C., & Vapnik, V. (1995). "Support-Vector Networks." *Machine Learning*, 20(3), 273–297. [DOI: 10.1007/BF00994018](https://doi.org/10.1007/BF00994018)
2. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer. (Capítulo 7: Sparse Kernel Machines).
3. Scikit-learn Documentation. "1.4. Support Vector Machines." [https://scikit-learn.org/stable/modules/svm.html](https://scikit-learn.org/stable/modules/svm.html).
4. Garreta, R., & Moncecchi, G. (2013). *Learning scikit-learn: Machine Learning in Python*. Packt Publishing.