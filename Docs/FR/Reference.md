# Référence de compatibilité Tensor 0.2.0

Cette page permet de choisir rapidement un dtype, un placement et une famille
d'opérations. Le [guide Tensor](README.md) détaille les formes, vues, axes et
contrats de chaque opération.

## Matrice d'exécution

| Famille | Dtypes CPU | Dtypes GPU | Différentiable |
| --- | --- | --- | --- |
| Construction, vues, `reshape`, `permute`, `select`, `narrow` | les neuf | métadonnées des neuf | vues `float32` suivies |
| `add`, `subtract`, `multiply`, `divide` | les neuf | `float32` | `float32` |
| `negate` | `float32`, entiers signés | `float32` | `float32` |
| `abs` | les neuf | `float32` | `float32` hors zéro |
| `exp`, `log`, `sqrt` | `float32` | `float32` | oui |
| `sum`, `min`, `max` | les neuf | `float32` | `sum` seulement |
| `mean`, `dot`, `matmul` | `float32`; `dot`/`matmul` aussi entiers | `float32` | `float32` |
| `concatenate`, `stack` | les neuf | les neuf résidents | `float32` |
| `gather` | source quelconque, indices `int32` | source `float32`, indices `int32` | source `float32` |
| activations, pertes, normalisation, convolution, pooling, dropout | `float32` | `float32` | oui selon l'opération |

« Les neuf » désigne `float32`, `int8`, `uint8`, `int16`, `uint16`, `int32`,
`uint32`, `int64` et `uint64`. Une disponibilité de transfert ne signifie pas
qu'un kernel de calcul existe : toute arithmétique entière GPU est refusée avant
création de pipeline ou soumission.

## Conversions et extractions

Les constructeurs déduisent le dtype du tableau Silex. `float`, `int` et `uint`
deviennent respectivement `float32`, `int64` et `uint64`. `cast(dtype)` réalise
une conversion numérique contrôlée sur CPU ; un tenseur GPU suit explicitement
`tensor.cpu().cast(dtype).to(device)`.

`values()`, `item()` et `at()` lisent exclusivement `float32`. Chaque entier
possède ses variantes exactes, par exemple `int32_values()`, `int32_item()` et
`int32_at(indices)`. Une extraction d'un autre dtype échoue sans conversion.
Toute extraction GPU exige d'abord `cpu()`.

## Overflow et valeurs flottantes

Les opérations entières CPU restent dans le dtype de leurs opérandes. Addition,
soustraction, multiplication, négation, division, somme, produit scalaire et
produit matriciel échouent dès qu'un résultat n'est pas représentable. La
division par zéro et le quotient signé minimal divisé par `-1` échouent aussi.

`float32` conserve les règles IEEE observables : NaN et infinis se propagent,
la division par zéro n'est pas convertie en erreur Tensor, et `min`/`max`
distinguent les deux signes de zéro. Les comparaisons CPU/GPU et différentielles
emploient des tolérances absolues et relatives propres à l'opération ; le
[rapport numérique](../../Tools/Oracle/REPORT.md) et le
[rapport neuronal](../../Tools/Oracle/NEURAL_REPORT.md) enregistrent les valeurs
exactes utilisées par les tests.

## Placement, synchronisation et durée de vie

`to(device)` est un upload explicite ; `cpu()` est un download et attend
l'achèvement des commandes dont dépend le résultat. Les opérations GPU restent
ordonnées et résidentes jusqu'à cette frontière. Un changement entre deux
devices s'écrit `tensor.cpu().to(other_device)`.

Un graphe autograd est consommé par un `backward()` réussi. Les gradients
s'accumulent sur les feuilles jusqu'à `zero_grad()`. Les paramètres, gradients
et états d'optimiseur restent sur le même placement ; après création de l'état,
un déplacement partiel est refusé.

## Erreurs structurantes

Tensor diagnostique avant mutation ou soumission les formes non compatibles,
axes invalides ou dupliqués, indices CPU hors limites, mélanges de dtype,
placements ou devices, hyperparamètres invalides, graphes déjà consommés et
checkpoints incompatibles. Sur GPU, `gather` conserve son chemin résident : un
indice hors limites produit un NaN sentinelle au lieu d'un readback caché.

Les limites de modèles et de checkpoints sont résumées dans la
[direction 0.2.0](Direction.md). Les parcours complets figurent dans les
[recettes d'entraînement](Recipes/Training.md).
