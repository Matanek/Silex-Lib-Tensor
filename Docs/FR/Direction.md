# Direction de Tensor 0.1.0

Tensor propose un calcul multidimensionnel explicite pour Silex. Une même
valeur immuable se déplace entre CPU et GPU, tandis que l'entraînement eager
emploie des feuilles mutables nommées. Cette direction emprunte des idées
éprouvées sans chercher à reproduire la surface d'un autre framework.

## Trois influences, un contrat Silex

PyTorch inspire l'ordre naturel `forward -> loss -> backward -> step`, les
graphes eager consommés après la rétropropagation, les couches composables et
les conventions numériques de SGD et Adam. Tensor ne reprend ni la mutation
générale des tenseurs, ni les graphes conservés, ni les dérivées d'ordre
supérieur.

TensorFlow rend utile la distinction entre une valeur de calcul et une variable
entraînable, ainsi que le placement explicite sur un runtime ou un device.
Tensor ne reprend pas les graphes différés, les sessions, la compilation de
fonctions ni le placement implicite.

JAX montre l'intérêt de valeurs de tableaux immuables et de transformations
fonctionnelles prévisibles. Tensor ne fournit toutefois ni `jit`, ni `vmap`, ni
transformation fonctionnelle générale de gradient : l'autodifférentiation reste
eager et attachée aux opérations exécutées.

## Ce que garantit la version 0.1.0

- neuf dtypes denses sur CPU, avec vues, broadcasting, réductions et algèbre
  linéaire contrôlée ;
- transfert bit-exact des neuf dtypes entre CPU et GPU ;
- calcul GPU `float32` résident, sans fallback ni readback implicite ;
- autodifférentiation inverse eager de `float32`, paramètres nommés, SGD, Adam
  et écrêtage par norme globale ;
- modèles séquentiels denses, convolutionnels simples et récurrents simples ;
- checkpoints déterministes des paramètres, validés avant mutation.

Les oracles NumPy, PyTorch, TensorFlow et JAX servent à vérifier ce contrat. Ils
ne l'étendent pas automatiquement lorsque leur propre API évolue.

## Limites volontaires

La version 0.1.0 n'inclut pas le calcul entier GPU, les tenseurs clairsemés, la
quantification, les graphes différés, la distribution multi-device, les
dérivées supérieures, AdamW, BatchNorm, les convolutions groupées ou
transposées, LSTM/GRU, embeddings et attention.

Un checkpoint contient uniquement les paramètres : ni code, ni graphe
autograd, ni device, ni état d'optimiseur. L'observation d'une valeur GPU et la
sauvegarde d'un modèle GPU restent des synchronisations et readbacks explicites.

Consultez la [référence de compatibilité](Reference.md) pour choisir un dtype et
un placement, puis les [recettes d'entraînement](Recipes/Training.md) pour
construire un MLP, un CNN ou un RNN.
