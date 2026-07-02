# Visualizing the Hidden Activity of ANNs

Visualización interactiva en **D3 + Flask** basada en Rauber et al. (2017),
*Visualizing the Hidden Activity of Artificial Neural Networks*.

## Protocolo (igual al paper)

- **MLP**: entrada → **4 capas ocultas ReLU de 1000 neuronas** → softmax(10).
- **T1 — Inter-época**: snapshots en épocas **0, 20, 40, 60, 80, 100** (pasos de 20),
  proyección t-SNE única compartida sobre la unión de todas las épocas (última capa oculta).
- **T2 — Inter-capa**: t-SNE de las **4 capas ocultas** (época final), inicializadas con la capa 1.
- **Datasets**: **MNIST**, **SVHN**, **CIFAR-10** (los tres del paper).

## Ejecutar

```bash
python app.py
# abrir http://127.0.0.1:5000
```

Los datos (`static/data/*.json` y `*_sprite.png`) ya están generados.

## Regenerar los datos (opcional)

```bash
python gen/generate_data.py
```

Entrena el MLP, extrae activaciones por época/capa, aplica PCA→t-SNE y exporta las
proyecciones + analíticas de clasificación + hojas de sprites de las miniaturas.

## Interacción (mantra: overview → zoom & filter → details on demand)

- Toggle **T1 / T2** y selector de **dataset**.
- **Animar evolución** (play), slider de etapa y *filmstrip* de pequeñas múltiples (Aₚ[1..4]).
- Herramienta **Pan/Zoom** ↔ **Lazo**: el lazo selecciona puntos a mano alzada.
- **Click** en un punto → lo selecciona, **baja la intensidad de los demás** y traza su trayectoria.
- **Leyenda** clicable para filtrar clases; **Limpiar selección** / **Reset** zoom.
- **Hover** → miniatura de la observación, clase real y predicción (✓/✗).
- Toggles de **trayectorias** (bundling) y **errores** (▲).

## Gráficas analíticas (¿cómo clasifica en cada época/capa?)

Todas están **enlazadas**: una selección en cualquiera (lazo, clic en punto, clic en
clase, clic en celda de confusión) re-filtra a las demás.

1. **Evolución de la calidad** — líneas de *Neighborhood Hit* y *Accuracy*. El **rastro
   se va construyendo** a medida que avanzas: en la etapa 0 no hay línea, sólo el punto.
2. **Matriz de confusión** — clase real vs. predicha; **clic en una celda** selecciona
   exactamente esas instancias (se resaltan en todo el resto). Ejes con **nombres reales
   por dataset** (p. ej. CIFAR-10 = airplane, automobile…), no 0-9.
3. **Recall por clase** — barras; clic en una clase la enfoca.
4. **Squares (estilo Ren et al.)** — abajo, a todo el ancho. Por cada clase predicha,
   distribución de *scores*: barras sólidas = correctos, cajas con contorno = mal
   clasificados (color = **clase real**). Clic en una clase traza dónde aterrizaron sus
   instancias (líneas conectoras).

**Clustering por clase**: el toggle *hulls* encierra cada clase real; los **mal
clasificados** se marcan con un **anillo rojo** distinto del resto.

En **T1** la predicción es la de la **red** (score = softmax) en esa época; en **T2** es un
**k-NN(6) sobre la proyección** de esa capa (score = fracción de votos) — mide cuán
separable/clasificable es la representación capa a capa.
