75.06/95.58 Organización de Datos
Primer Cuatrimestre de 2020
Trabajo Práctico 2

## Introducción

Este trabajo va a consistir en el procedimiento necesario para poder conseguir buenos resultados a la hora de probar distintos algoritmos de machine learning. 

Para dar un poco de contexto, esta investigación se desarrolló bajo la materia Organización de Datos de la facultad de ingeniería de la Universidad de Buenos Aires. Al mismo tiempo se basa en una competencia de Kaggle llamada “ Real or Not? NLP with Disaster Tweets”.

Esta competencia provee dos sets de datos (train y test) los cuales tienen un formato de 7316 x 6 (filas x columnas) más una columna de target. Cada fila representa un tweet que habla sobre una catástrofe y el target declara si efectivamente sucedió o no.

El procedimiento que se va a ver en las siguientes páginas consta de una serie de pasos los cuales fueron cuidadosamente trabajados. 

Previo a esta presentación hicimos una investigación de las cualidades de este set, la cual la pueden visitar en el siguiente link: https://github.com/maxilevi/tp1-datos. En base a estos datos vamos a tener una base sobre que datos interesantes se pueden generar como features. Por ende, luego se comienza un amplio proceso de feature engineering. 

En esta sección primero se establecieron algunos tipos de procesamiento de texto base (TF-IDF, Bag of Words, Embeddings de Word2vec). Posteriormente se dispuso de la opción de aplicarle embeddings spaCy, hacer una limpieza de texto y aplicar features que nosotros creamos manualmente. 

Una vez que ya se procesó el set de entrenamiento y se creó el data frame deseado, se pasó a probar los distintos algoritmos de Machine Learning que aparecen en el índice, en la sección de Modelos. Para poder obtener lo mejor de estos algoritmos se usaron tres algoritmos para hacer una optimización de hiperparametros.

Ya habiendo probado todos los algoritmos se procedió a ensamblarlos, algunos en conjunto y otros solos. 

Dentro de este informe y del repositorio provisto en la primer carilla van a poder observar cómo fue que hicimos todo lo anterior mencionado. 

Esperamos que les resulte interesante y puedan llegar a entender todos los pasos que realizamos.

## Colabs

[RoBERTa](https://colab.research.google.com/drive/1frFhlSWtsrwT14CAAsYCY5tTJ-iqz-P4?usp=sharing)
[XLNet](https://colab.research.google.com/drive/1i23hQAcW5-1cUJg8YCSuSXDcAldV3KZE?usp=sharing)
