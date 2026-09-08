# Tensor

`Tensor` représente des valeurs numériques multidimensionnelles sans exposer
leur stockage. Cette première version exécute immédiatement les opérations sur
CPU et emploie le type Silex `float`.

## Créer un tenseur

Les valeurs sont aplaties par lignes (row-major) et leur nombre doit
correspondre exactement à la forme déclarée :

```sx
use Tensor

let image = Tensor([
    0.1, 0.2, 0.3,
    0.4, 0.5, 0.6
], [2, 3])

assert(image.rank() == 2)
assert(image.at([1, 2]) == 0.6)
```

Des fabriques couvrent les intentions fréquentes :

```sx
let bias = Tensor.ones([3])
let weights = Tensor.zeros([3, 4])
let temperature = Tensor.scalar(0.7)
```

`shape()` et `values()` rendent des copies détachées. Un `Tensor` ne propose
aucune mutation publique et conserve la sémantique de valeur de Silex. Une
future optimisation du stockage ne changera pas cette garantie.

## Calculer

Les opérations élément par élément exigent actuellement des formes identiques.
Une valeur scalaire peut être ajoutée, soustraite, multipliée ou divisée sans
fabriquer un tenseur intermédiaire :

```sx
let left = Tensor.vector([1.0, 2.0, 3.0])
let right = Tensor.vector([4.0, 5.0, 6.0])
let centered = left.add(right).divide(2.0)

assert(centered.sum() == 10.5)
assert(centered.mean() == 3.5)
```

`matmul()` et `transpose()` portent pour l'instant le contrat explicite des
matrices de rang 2 :

```sx
let inputs = Tensor.matrix([1.0, 2.0, 3.0, 4.0], 2, 2)
let weights = Tensor.matrix([2.0, 0.0, 0.0, 3.0], 2, 2)
let outputs = inputs.matmul(weights)

assert(outputs.at([1, 1]) == 12.0)
```

Une forme négative, un nombre de valeurs incohérent, un indice invalide ou des
formes incompatibles terminent le programme avec un diagnostic. `mean()`
refuse un tenseur vide.

## Direction

Le contrat privilégie l'usage immédiat et lisible popularisé par PyTorch, avec
des valeurs immuables proches de l'approche fonctionnelle de JAX. Les
préoccupations de graphe, d'autodifférentiation, de dtype et de device restent
hors de cette première surface afin de pouvoir choisir ensuite un backend sans
le faire transparaître dans les usages élémentaires.

Le broadcasting, les tranches et les tenseurs de rang supérieur dans l'algèbre
linéaire seront ajoutés avec leurs propres preuves consommatrices. `AI` pourra
dépendre de `Tensor`; l'inverse resterait une dépendance de domaine incorrecte.

## Développement

Depuis la racine de `SilexProject` :

```text
silex link Packages/Tensor
silex link Packages/Tensor --workspace Packages/Tensor/Tests/Consumer
silex test Packages/Tensor/Tests/Consumer
```
