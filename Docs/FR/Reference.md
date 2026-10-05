# Référence de compatibilité Tensor 0.2.0

Cette page permet de choisir rapidement un dtype, un placement et une famille
d'opérations. Le [guide Tensor](README.md) détaille les formes, vues, axes et
contrats de chaque opération.

## Matrice d'exécution

| Famille | Dtypes CPU | Dtypes GPU | Différentiable |
| --- | --- | --- | --- |
| Construction, vues, `reshape`, `permute`, `select`, `narrow` | les neuf + booléens | métadonnées des neuf + booléens | vues `float32` suivies |
| `add`, `subtract`, `multiply`, `divide` | les neuf | `float32` | `float32` |
| `negate` | `float32`, entiers signés | `float32` | `float32` |
| `abs` | les neuf | `float32` | `float32` hors zéro |
| `exp`, `log`, `sqrt` | `float32` | `float32` | oui |
| `sum`, `min`, `max` | les neuf | `float32` | `sum` seulement |
| `mean`, `dot`, `matmul` | `float32`; `dot`/`matmul` aussi entiers | `float32` | `float32` |
| `concatenate`, `stack` | les neuf + booléens | les neuf + booléens résidents | `float32` |
| `gather` | source quelconque, indices `int32` | source `float32`, indices `int32` | source `float32` |
| `Neural.LSTM` | `float32`, entrée rang3, hidden/cell rang2 | idem, calcul résident | entrée, deux états et quatre paramètres ; temps0 inutilisés explicites |
| `Neural.GRU` | `float32`, entrée rang3, état rang2 | idem, calcul résident | entrée, état initial et quatre paramètres ; temps0 inutilisés explicites |
| `Neural.TransformerBlock` | bloc `float32`, rang3 | idem, calcul résident | entrée et seize paramètres nommés |
| `Neural.MultiHeadAttention` | projections `float32`, rang3 | idem, calcul résident | Q/K/V et huit paramètres nommés |
| `Neural.attention` | `float32`, rang3/4 | idem, calcul résident | Q/K/V et poids optionnels |
| `Neural.Embedding` | indices `int32`, poids `float32` | idem, calcul résident | poids dense ; padding à dérivée arrêtée |
| `Neural.sequence_mask`, `Neural.causal_mask` | booléens construits depuis métadonnées `int` | transfert explicite, logique résidente | détachés ; `true` autorise |
| comparaisons numériques | les neuf | `float32` | non, résultat booléen détaché |
| égalité booléenne, logique | booléens | booléens | non |
| `where` | branches : les neuf ; masque booléen | branches `float32` ; masque booléen | branches `float32` |
| `masked_select` | les neuf + booléens | refus explicite | source `float32` |
| dimensions unitaires, `expand`, `repeat` | les neuf + booléens | les neuf + booléens | float32 |
| `variance`, `standard_deviation` | `float32` | `float32` résident | oui, correction explicite |
| `topk` | les neuf | `float32` résident | valeurs float32 ; indices int32 détachés |
| `argmin`, `argmax` | les neuf | `float32` résident | non, indices int32 détachés |
| `arange`, `linspace` | int32 / float32, données CPU | transfert explicite | non |
| `Autograd.apply` | règle extérieure `float32` | règle extérieure résidente `float32` | contributions détachées validées avant accumulation |
| `Autograd.grad` | graphes `float32` | graphes `float32` résidents | premier ordre, résultats détachés |
| `Autograd.gradcheck` | entrées/sortie `float32` | contrôle CPU explicite | Jacobien par différences centrales |
| `masked_softmax(mask,axis)` | scores `float32`, masque booléen | idem, calcul résident | scores ; lignes masquées à gradient nul |
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
`bool_values()`, `bool_item()` et `bool_at(indices)` extraient les booléens.
`cast` ne convertit pas entre booléens et nombres. Le [guide des masques](Masks.md)
détaille les comparaisons, gradients et limites de compaction.
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

`Neural.Parameter.replace(value)` remplace une valeur seulement après
`zero_grad()` et conserve dtype, forme et placement. La valeur reçue devient une
nouvelle feuille détachée ; l'état éventuel d'un optimiseur n'est pas réinitialisé.

## Erreurs structurantes

Tensor diagnostique avant mutation ou soumission les formes non compatibles,
axes invalides ou dupliqués, indices CPU hors limites, mélanges de dtype,
placements ou devices, hyperparamètres invalides, graphes déjà consommés et
checkpoints incompatibles. Sur GPU, `gather` conserve son chemin résident : un
indice hors limites produit un NaN sentinelle au lieu d'un readback caché.

Les limites de modèles et de checkpoints sont résumées dans la
[direction 0.2.0](Direction.md). Les parcours complets figurent dans les
[recettes d'entraînement](Recipes/Training.md).

## Profilage GPU

`Tensor.Profile` délimite une période sur un device ; `snapshot()` retourne
`Tensor.ProfileStatistics` immuable et `reset()` recommence la période. Voir
[attribution, compteurs et durée de vie](Profiling.md).

Le [guide des gradients fonctionnels](Gradients.md) décrit les slots préservés,
les seeds, la consommation du graphe et le contrôle du Jacobien.

[Formes, vues et tuiles](Shapes.md) : axes, dimensions vides et gradients.

[Variance et écart-type](Statistics.md) : correction, domaines et gradient constant.

[Progressions et grilles](Sequences.md) : bornes, pas, cardinalité et dtypes.

[Buffers et gel des paramètres](State.md) : détachement, identité, transferts et état conservé des optimiseurs.

[Registre des couches](Registration.md) : enfants nommés, ordre, identités, cycles et collisions.

[État récursif du modèle](ModelState.md) : modes, hooks locaux, gel et transferts prévalidés.

[Lots d’indices reproductibles](BatchSampling.md) : seed, époque, dernier lot et snapshot du curseur.

[Lots supervisés](SupervisedBatches.md) : sources extérieures, collation, erreurs récupérables et placement.

[Reprise des optimiseurs](OptimizerState.md) : moments nommés, restauration atomique et taux courant.

[Décroissance découplée et calendriers](OptimizerUpdates.md) : AdamW, étape explicite et taux du prochain pas.

[Métadonnées CPU du modèle](HostState.md) : buffers à placement CPU explicite et transferts récursifs.


## Checkpoint d’entraînement

[TrainingCheckpoint](TrainingCheckpoint.md) restaure modèle, état SGD/Adam/AdamW,
curseur supervisé, étape/calendrier, modes, gel et dropout ensemble sur CPU.
Format versionné exact pour tous les dtypes de stockage ; save/load récupérables,
validation complète avant mutation, transferts GPU explicites. Le checkpoint de
paramètres historique reste distinct et compatible.

[Préchargement CPU](BatchPrefetch.md) : `BatchCursor(...,prefetch:N)`, éligibilité
`Source.supports_prefetch()`, snapshots consommés, erreurs ordonnées et `close()`.


## Embedding

`Neural.Embedding(name, vocabulary_size, embedding_features, seed, padding_index:null)` : indices int32 de rang1/2, table float32 et gradient dense. Padding à dérivée arrêtée, placements explicites et limites CPU/GPU dans le [guide](Embeddings.md).

## Masques de séquence

`Neural.sequence_mask(lengths:int[],maximum_length:int)` et `Neural.causal_mask(query_length:int,key_length:int,query_offset:int=0)` : masques CPU booléens, longueurs et positions exactes, transfert explicite. Conventions et composition dans le [guide des séquences](SequenceMasks.md).

## Softmax masqué

`tensor.masked_softmax(mask:Tensor,axis:int)` : visibilité booléenne broadcastée vers les scores float32, normalisation des valeurs visibles finies, positions/lignes exclues à sortie et gradient nuls. Formes, placements et limites dans le [guide](MaskedSoftmax.md).

## Attention

`Neural.attention(query,key,value,mask:null,scale:null,return_weights:false)` : calcul batch-first rang3/4, échelle finie, visibilité booléenne et poids optionnels. Formes vides, gradients partagés et placement dans le [guide](Attention.md).

## Attention multi-têtes

`Neural.MultiHeadAttention(name,features,heads,seed,dropout_probability:0.0)` : `attend` self/cross avec masque, échelle, itération et poids optionnels ; `forward` self à itération0. État partagé, dropout et reprise dans le [guide](MultiHeadAttention.md).

## Transformer

`Neural.TransformerBlock(name,features,heads,feed_forward_features,seed,norm_first:true,dropout_probability:0.0,epsilon:0.00001)` : `encode` avec masque/itération/poids optionnels ; `forward` à itération0. Résidus, pre/post-norm, MLP et reprise dans le [guide](Transformer.md).
