#---- Archivo: regresionlogistica.R ----

# Instalación del paquete necesario
#install.packages("stats")

# Carga del paquete
library(stats)

# Fijar la semilla para reproducibilidad
set.seed(123)

# Número de observaciones
n <- 100

# Generar las variables independientes X1, X2, ..., X15
# Creamos una matriz de tamaño n x 15 con valores generados aleatoriamente de una distribución normal
X <- as.data.frame(matrix(rnorm(n * 15), nrow = n, ncol = 15))
colnames(X) <- paste0("X", 1:15)  # Nombramos las columnas como X1, X2, ..., X15

# Coeficientes verdaderos para las variables independientes
# Generamos un vector de 16 coeficientes (incluyendo el intercepto) aleatorios entre -1 y 1
beta <- runif(16, -1, 1)  # 15 coeficientes más el intercepto

# Generar el término lineal
# Calculamos el término lineal utilizando los coeficientes y las variables independientes
linear_term <- beta[1] + as.matrix(X) %*% beta[-1]

# Generar la probabilidad utilizando la función logística
# Calculamos las probabilidades utilizando la función logística
p <- 1 / (1 + exp(-linear_term))

# Generar la variable dependiente binaria Y
# Generamos valores binarios (0 o 1) utilizando las probabilidades calculadas
Y <- rbinom(n, 1, p)

# Combinar las variables independientes y la variable dependiente en un data frame
data <- cbind(Y, X)

# Dividir el conjunto de datos en entrenamiento y prueba
set.seed(123)  # Fijar la semilla para reproducibilidad
train_indices <- sample(1:n, size = 0.7 * n)  # 70% de los datos para entrenamiento
train_set <- data[train_indices, ]  # Conjunto de entrenamiento
test_set <- data[-train_indices, ]  # Conjunto de prueba

# Ajuste del modelo de regresión logística en el conjunto de entrenamiento
# Ajustamos un modelo de regresión logística utilizando las variables independientes para predecir Y
model <- glm(Y ~ ., data = train_set, family = binomial)

# Resumen del modelo
# Mostramos un resumen del modelo ajustado
summary(model)

# Guardar el modelo y los resultados en un archivo
# Guardamos el modelo ajustado en un archivo .RData
save(model, file = "regresion_logistica_modelo.RData")

# Guardar los datos simulados en archivos CSV
# Guardamos los conjuntos de datos de entrenamiento y prueba en archivos CSV
write.csv(train_set, "datos_entrenamiento_regresion_logistica.csv", row.names = FALSE)
write.csv(test_set, "datos_prueba_regresion_logistica.csv", row.names = FALSE)

# Hacer predicciones en el conjunto de prueba
# Utilizamos el modelo ajustado para hacer predicciones en el conjunto de prueba
test_set$prob_pred <- predict(model, newdata = test_set, type = "response")
test_set$Y_pred <- ifelse(test_set$prob_pred > 0.5, 1, 0)  # Convertimos probabilidades a clases binarias

# Calcular la precisión de las predicciones
# Calculamos la precisión de las predicciones comparando con los valores reales de Y
accuracy <- mean(test_set$Y_pred == test_set$Y)
cat("La precisión del modelo en el conjunto de prueba es:", accuracy, "\n")

# Guardar las predicciones en un archivo CSV
# Guardamos las predicciones y las probabilidades predichas en un archivo CSV
write.csv(test_set, "predicciones_regresion_logistica.csv", row.names = FALSE)

# Graficar los coeficientes estimados
# Graficamos los coeficientes estimados del modelo ajustado
plot(coef(model), main = "Coeficientes Estimados del Modelo de Regresión Logística", 
     xlab = "Variables", ylab = "Coeficientes", type = "h", col = "blue")
abline(h = 0, col = "red", lwd = 2)

# Mostrar un mensaje indicando que el proceso ha finalizado
cat("El modelo de regresión logística se ha ajustado, se han hecho predicciones y los resultados se han guardado en 'regresion_logistica_modelo.RData'.\n")

#---- Archivo: regresionlogistica_titanic.R ----

# Instalación del paquete necesario
#install.packages("titanic")
#install.packages("dplyr")

# Carga de los paquetes
library(titanic)
library(dplyr)

# Cargar el conjunto de datos Titanic
data("titanic_train")

# Ver las primeras filas del conjunto de datos
head(titanic_train)

# Preprocesamiento de los datos
# Seleccionar variables relevantes y eliminar filas con valores faltantes
titanic_clean <- titanic_train %>%
  select(Survived, Pclass, Sex, Age, SibSp, Parch, Fare) %>%
  na.omit()

# Convertir la variable 'Sex' a factor
titanic_clean$Sex <- as.factor(titanic_clean$Sex)

# Dividir el conjunto de datos en entrenamiento (70%) y prueba (30%)
set.seed(123)
train_indices <- sample(1:nrow(titanic_clean), size = 0.7 * nrow(titanic_clean))
train_set <- titanic_clean[train_indices, ]
test_set <- titanic_clean[-train_indices, ]

# Ajuste del modelo de regresión logística en el conjunto de entrenamiento
model <- glm(Survived ~ ., data = train_set, family = binomial)

# Resumen del modelo
summary(model)

# Guardar el modelo y los resultados en un archivo
save(model, file = "regresion_logistica_titanic_modelo.RData")

# Guardar los datos simulados en archivos CSV
write.csv(train_set, "datos_entrenamiento_titanic.csv", row.names = FALSE)
write.csv(test_set, "datos_prueba_titanic.csv", row.names = FALSE)

# Hacer predicciones en el conjunto de prueba
test_set$prob_pred <- predict(model, newdata = test_set, type = "response")
test_set$Survived_pred <- ifelse(test_set$prob_pred > 0.5, 1, 0)

# Calcular la precisión de las predicciones
accuracy <- mean(test_set$Survived_pred == test_set$Survived)
cat("La precisión del modelo en el conjunto de prueba es:", accuracy, "\n")

# Guardar las predicciones en un archivo CSV
write.csv(test_set, "predicciones_titanic.csv", row.names = FALSE)

# Graficar los coeficientes estimados
plot(coef(model), main = "Coeficientes Estimados del Modelo de Regresión Logística", 
     xlab = "Variables", ylab = "Coeficientes", type = "h", col = "blue")
abline(h = 0, col = "red", lwd = 2)

# Mostrar un mensaje indicando que el proceso ha finalizado
cat("El modelo de regresión logística se ha ajustado, se han hecho predicciones y los resultados se han guardado en 'regresion_logistica_titanic_modelo.RData'.\n")

# Calcular la matriz de confusión
table(test_set$Survived, test_set$Survived_pred)

# Calcular sensibilidad y especificidad
sensitivity <- sum(test_set$Survived == 1 & test_set$Survived_pred == 1) / sum(test_set$Survived == 1)
specificity <- sum(test_set$Survived == 0 & test_set$Survived_pred == 0) / sum(test_set$Survived == 0)

# Calcular AUC-ROC
library(pROC)
roc_curve <- roc(test_set$Survived, test_set$prob_pred)
auc(roc_curve)

# Graficar la curva ROC
plot(roc_curve, main = "Curva ROC para el Modelo de Regresión Logística")

#---- Archivo: cancerLogReg.R ----

# Instalación de paquetes necesarios
if (!requireNamespace("mlbench", quietly = TRUE)) {
  install.packages("mlbench")
}
if (!requireNamespace("dplyr", quietly = TRUE)) {
  install.packages("dplyr")
}

# Carga de los paquetes
library(mlbench)
library(dplyr)

# Cargar el conjunto de datos BreastCancer
data("BreastCancer")

# Ver las primeras filas del conjunto de datos
head(BreastCancer)

# Preprocesamiento de los datos
# Eliminar la columna de identificación y filas con valores faltantes
breast_cancer_clean <- BreastCancer %>%
  select(-Id) %>%
  na.omit()

# Convertir la variable 'Class' a factor binario
breast_cancer_clean$Class <- ifelse(breast_cancer_clean$Class == "malignant", 1, 0)
breast_cancer_clean$Class <- as.factor(breast_cancer_clean$Class)

# Convertir las demás columnas a numéricas
breast_cancer_clean[, 1:9] <- lapply(breast_cancer_clean[, 1:9], as.numeric)

# Dividir el conjunto de datos en entrenamiento (70%) y prueba (30%)
set.seed(123)
train_indices <- sample(1:nrow(breast_cancer_clean), size = 0.7 * nrow(breast_cancer_clean))
train_set <- breast_cancer_clean[train_indices, ]
test_set <- breast_cancer_clean[-train_indices, ]

# Ajuste del modelo de regresión logística en el conjunto de entrenamiento
model <- glm(Class ~ ., data = train_set, family = binomial)

# Resumen del modelo
summary(model)

# Guardar el modelo y los resultados en un archivo
save(model, file = "regresion_logistica_cancer_modelo.RData")

# Guardar los datos simulados en archivos CSV
write.csv(train_set, "datos_entrenamiento_cancer.csv", row.names = FALSE)
write.csv(test_set, "datos_prueba_cancer.csv", row.names = FALSE)

# Hacer predicciones en el conjunto de prueba
test_set$prob_pred <- predict(model, newdata = test_set, type = "response")
test_set$Class_pred <- ifelse(test_set$prob_pred > 0.5, 1, 0)

# Calcular la precisión de las predicciones
accuracy <- mean(test_set$Class_pred == test_set$Class)
cat("La precisión del modelo en el conjunto de prueba es:", accuracy, "\n")

# Guardar las predicciones en un archivo CSV
write.csv(test_set, "predicciones_cancer.csv", row.names = FALSE)

# Graficar los coeficientes estimados
plot(coef(model), main = "Coeficientes Estimados del Modelo de Regresión Logística", 
     xlab = "Variables", ylab = "Coeficientes", type = "h", col = "blue")
abline(h = 0, col = "red", lwd = 2)

# Mostrar un mensaje indicando que el proceso ha finalizado
cat("El modelo de regresión logística se ha ajustado, se han hecho predicciones y los resultados se han guardado en 'regresion_logistica_cancer_modelo.RData'.\n")

#---- Archivo: cancerLogRegSimulado.R ----

# Instalación del paquete necesario
if (!requireNamespace("dplyr", quietly = TRUE)) {
  install.packages("dplyr")
}

# Carga del paquete
library(dplyr)

# Fijar la semilla para reproducibilidad
set.seed(123)

# Número de observaciones
n <- 150

# Generar las variables independientes X1, X2, ..., X15
# Creamos una matriz de tamaño n x 15 con valores generados aleatoriamente de una distribución normal
X <- as.data.frame(matrix(rnorm(n * 15), nrow = n, ncol = 15))
colnames(X) <- paste0("X", 1:15)  # Nombramos las columnas como X1, X2, ..., X15

# Coeficientes verdaderos para las variables independientes
# Generamos un vector de 16 coeficientes (incluyendo el intercepto) aleatorios entre -1 y 1
beta <- runif(16, -1, 1)  # 15 coeficientes más el intercepto

# Generar el término lineal
# Calculamos el término lineal utilizando los coeficientes y las variables independientes
linear_term <- beta[1] + as.matrix(X) %*% beta[-1]

# Generar la probabilidad utilizando la función logística
# Calculamos las probabilidades utilizando la función logística
p <- 1 / (1 + exp(-linear_term))

# Generar la variable dependiente binaria Y
# Generamos valores binarios (0 o 1) utilizando las probabilidades calculadas
Y <- rbinom(n, 1, p)

# Combinar las variables independientes y la variable dependiente en un data frame
data <- cbind(Y, X)

# Dividir el conjunto de datos en entrenamiento y prueba
set.seed(123)  # Fijar la semilla para reproducibilidad
train_indices <- sample(1:n, size = 0.7 * n)  # 70% de los datos para entrenamiento
train_set <- data[train_indices, ]  # Conjunto de entrenamiento
test_set <- data[-train_indices, ]  # Conjunto de prueba

# Ajuste del modelo de regresión logística en el conjunto de entrenamiento
# Ajustamos un modelo de regresión logística utilizando las variables independientes para predecir Y
model <- glm(Y ~ ., data = train_set, family = binomial)

# Resumen del modelo
# Mostramos un resumen del modelo ajustado
summary(model)

# Guardar el modelo y los resultados en un archivo
# Guardamos el modelo ajustado en un archivo .RData
save(model, file = "regresion_logistica_cancer_modelo_simulado.RData")

# Guardar los datos simulados en archivos CSV
# Guardamos los conjuntos de datos de entrenamiento y prueba en archivos CSV
write.csv(train_set, "datos_entrenamiento_cancer_simulado.csv", row.names = FALSE)
write.csv(test_set, "datos_prueba_cancer_simulado.csv", row.names = FALSE)

# Hacer predicciones en el conjunto de prueba
# Utilizamos el modelo ajustado para hacer predicciones en el conjunto de prueba
test_set$prob_pred <- predict(model, newdata = test_set, type = "response")
test_set$Y_pred <- ifelse(test_set$prob_pred > 0.5, 1, 0)  # Convertimos probabilidades a clases binarias

# Calcular la precisión de las predicciones
# Calculamos la precisión de las predicciones comparando con los valores reales de Y
accuracy <- mean(test_set$Y_pred == test_set$Y)
cat("La precisión del modelo en el conjunto de prueba es:", accuracy, "\n")

# Guardar las predicciones en un archivo CSV
# Guardamos las predicciones y las probabilidades predichas en un archivo CSV
write.csv(test_set, "predicciones_cancer_simulado.csv", row.names = FALSE)

# Graficar los coeficientes estimados
# Graficamos los coeficientes estimados del modelo ajustado
plot(coef(model), main = "Coeficientes Estimados del Modelo de Regresión Logística", 
     xlab = "Variables", ylab = "Coeficientes", type = "h", col = "blue")
abline(h = 0, col = "red", lwd = 2)

# Mostrar un mensaje indicando que el proceso ha finalizado
cat("El modelo de regresión logística se ha ajustado, se han hecho predicciones y los resultados se han guardado en 'regresion_logistica_cancer_modelo_simulado.RData'.\n")

#---- Archivo: simulcorrectedCancer.R ----

# Instalación del paquete necesario
if (!requireNamespace("dplyr", quietly = TRUE)) {
  install.packages("dplyr")
}

# Carga del paquete
library(dplyr)

# Fijar la semilla para reproducibilidad
set.seed(123)

# Número de observaciones
n <- 1500

# Simulación de las variables independientes
# Tamaño del Tumor (en cm)
Tumor_Size <- rnorm(n, mean = 3, sd = 1.5)

# Estado de los Ganglios Linfáticos (número de ganglios afectados)
Lymph_Nodes <- rpois(n, lambda = 3)

# Grado del Tumor (1 a 3)
Tumor_Grade <- sample(1:3, n, replace = TRUE)

# Receptores Hormonales (0: negativo, 1: positivo)
Estrogen_Receptor <- rbinom(n, 1, 0.7)
Progesterone_Receptor <- rbinom(n, 1, 0.7)

# Estado HER2 (0: negativo, 1: positivo)
HER2_Status <- rbinom(n, 1, 0.3)

# Ki-67 (% de células proliferativas)
Ki_67 <- rnorm(n, mean = 20, sd = 10)

# Edad del Paciente (años)
Age <- rnorm(n, mean = 50, sd = 10)

# Histopatología (1: ductal, 2: lobular, 3: otros)
Histopathology <- sample(1:3, n, replace = TRUE)

# Márgenes Quirúrgicos (0: positivo, 1: negativo)
Surgical_Margins <- rbinom(n, 1, 0.8)

# Invasión Linfovascular (0: no, 1: sí)
Lymphovascular_Invasion <- rbinom(n, 1, 0.4)

# Tratamientos Previos (0: no, 1: sí)
Prior_Treatments <- rbinom(n, 1, 0.5)

# Tipo de Cirugía (0: mastectomía, 1: lumpectomía)
Surgery_Type <- rbinom(n, 1, 0.5)

# Metástasis (0: no, 1: sí)
Metastasis <- rbinom(n, 1, 0.2)

# Índice de Masa Corporal (IMC)
BMI <- rnorm(n, mean = 25, sd = 5)

# Marcadores Genéticos (0: negativo, 1: positivo)
Genetic_Markers <- rbinom(n, 1, 0.1)

# Generar la variable dependiente binaria Y (sobrevivencia 0: no, 1: sí)
# Utilizaremos una combinación arbitraria de las variables para generar Y
linear_term <- -1 + 0.5 * Tumor_Size - 0.3 * Lymph_Nodes + 0.2 * Tumor_Grade + 
  0.4 * Estrogen_Receptor + 0.3 * Progesterone_Receptor - 0.2 * HER2_Status + 
  0.1 * Ki_67 - 0.05 * Age + 0.3 * Surgical_Margins - 0.4 * Lymphovascular_Invasion +
  0.2 * Prior_Treatments + 0.1 * Surgery_Type - 0.5 * Metastasis + 0.01 * BMI + 
  0.2 * Genetic_Markers
p <- 1 / (1 + exp(-linear_term))
Y <- rbinom(n, 1, p)

# Combinar las variables independientes y la variable dependiente en un data frame
data <- data.frame(Y, Tumor_Size, Lymph_Nodes, Tumor_Grade, Estrogen_Receptor, 
                   Progesterone_Receptor, HER2_Status, Ki_67, Age, Histopathology,
                   Surgical_Margins, Lymphovascular_Invasion, Prior_Treatments,
                   Surgery_Type, Metastasis, BMI, Genetic_Markers)

# Dividir el conjunto de datos en entrenamiento y prueba
set.seed(123)  # Fijar la semilla para reproducibilidad
train_indices <- sample(1:n, size = 0.7 * n)  # 70% de los datos para entrenamiento
train_set <- data[train_indices, ]  # Conjunto de entrenamiento
test_set <- data[-train_indices, ]  # Conjunto de prueba

# Ajuste del modelo de regresión logística en el conjunto de entrenamiento
# Ajustamos un modelo de regresión logística utilizando las variables independientes para predecir Y
model <- glm(Y ~ ., data = train_set, family = binomial)

# Resumen del modelo
# Mostramos un resumen del modelo ajustado
summary(model)

# Guardar el modelo y los resultados en un archivo
# Guardamos el modelo ajustado en un archivo .RData
save(model, file = "regresion_logistica_cancer_modelo_simulado.RData")

# Guardar los datos simulados en archivos CSV
# Guardamos los conjuntos de datos de entrenamiento y prueba en archivos CSV
write.csv(train_set, "datos_entrenamiento_cancer_simulado.csv", row.names = FALSE)
write.csv(test_set, "datos_prueba_cancer_simulado.csv", row.names = FALSE)

# Hacer predicciones en el conjunto de prueba
# Utilizamos el modelo ajustado para hacer predicciones en el conjunto de prueba
test_set$prob_pred <- predict(model, newdata = test_set, type = "response")
test_set$Y_pred <- ifelse(test_set$prob_pred > 0.5, 1, 0)  # Convertimos probabilidades a clases binarias

# Calcular la precisión de las predicciones
# Calculamos la precisión de las predicciones comparando con los valores reales de Y
accuracy <- mean(test_set$Y_pred == test_set$Y)
cat("La precisión del modelo en el conjunto de prueba es:", accuracy, "\n")

# Guardar las predicciones en un archivo CSV
# Guardamos las predicciones y las probabilidades predichas en un archivo CSV
write.csv(test_set, "predicciones_cancer_simulado.csv", row.names = FALSE)

# Graficar los coeficientes estimados
# Graficamos los coeficientes estimados del modelo ajustado
plot(coef(model), main = "Coeficientes Estimados del Modelo de Regresión Logística", 
     xlab = "Variables", ylab = "Coeficientes", type = "h", col = "blue")
abline(h = 0, col = "red", lwd = 2)

# Mostrar un mensaje indicando que el proceso ha finalizado
cat("El modelo de regresión logística se ha ajustado, se han hecho predicciones y los resultados se han guardado en 'regresion_logistica_cancer_modelo_simulado.RData'.\n")

