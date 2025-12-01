## **Activity 1:** Exploring Word Embeddings with GloVe and Numpy

Esta actividad se enfoca en comprender y aplicar los fundamentos de las **Word Embeddings** de GloVe en el contexto del Procesamiento de Lenguaje Natural (NLP). 
Implica cargar y serializar un amplio vocabulario de vectores, para luego utilizar **Numpy** y **Scikit-learn** para **reducir la dimensionalidad** mediante PCA y t-SNE, y así poder visualizar las **relaciones semánticas** entre palabras. 

Además, incluye la implementación desde cero (sin librerías de NLP pre-construidas) de dos funcionalidades clave:
* La función de **similitud de coseno** para encontrar palabras conceptualmente cercanas.
* La función de **analogías** para resolver relaciones del tipo $A:B :: C:?$.

## **Activity 2,3:** Code GPT-2

Este notebook documenta la implementación del modelo GPT-2, un Large Language Model (LLM) desde cero, siguiendo la **arquitectura de Transformer** con las modificaciones específicas de GPT-2, como la **Atención Causal** (Masked Self-Attention) y la aplicación de **Layer Normalization** antes de cada sub-capa. El objetivo es comprender cómo la arquitectura autoregresiva del Transformer permite al modelo aprender dependencias de largo alcance para generar texto. 

Tras definir la clase **Config** con hiperparámetros clave (embed_size, num_layers y num_heads), se construyen los móludos más relevantes:
* **SelfAttention**, que usa una máscara triangular baja para evitar ver tokens futuros.
* **FFN** (Feed-Forward Network) con activación GeLU.
* El **bloque Transformer** que integra estos módulos con conexiones residuales.
* Finalmente, la **clase GPT2** ensambla estos bloques, añade los Token and Positional Embeddings, y proyecta la salida al vocabulario.

El modelo se entrena con un dataset basado en el texto de la Biblia y se valida con la función *sample* para generar oraciones.
