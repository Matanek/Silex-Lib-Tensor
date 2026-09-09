# Tensor

`Tensor` représente une valeur numérique dense sans exposer son stockage. Le
même type public couvre `float32`, les entiers signés et non signés de 8, 16,
32 et 64 bits, sur CPU comme sur GPU.

## Créer et inspecter

`scalar`, `vector`, `matrix` et le constructeur forme-valeurs déduisent le
dtype du type Silex fourni. `float` devient `float32`, `int` devient `int64` et
`uint` devient `uint64`.

```sx
use Tensor
use Tensor.DType

var samples:float[] = [1.0, 2.0, 3.0]
var labels:int32[] = [2 as int32, 1 as int32, 0 as int32]

let x = Tensor.vector(samples)
let y = Tensor(labels, [3])

assert(x.dtype() == DType.float32())
assert(y.dtype() == DType.int32())
assert(y.int32_values()[0] == 2 as int32)
```

Les neuf valeurs de `DType` sont `float32`, `int8`, `uint8`, `int16`,
`uint16`, `int32`, `uint32`, `int64` et `uint64`. Elles exposent uniquement
leur largeur et leur famille numérique. `DType` appartient directement au
module `Tensor`, sans sous-module intermédiaire : `use Tensor.DType` sélectionne
cette déclaration exacte. `zeros`, `ones` et les surcharges de `full`
complètent les constructions usuelles.

`values()` et `item()` extraient exclusivement `float32`. Chaque dtype entier
possède les extracteurs correspondants, par exemple `int32_values()` et
`int32_item()`. Une extraction incompatible échoue au lieu de convertir
silencieusement. `cast(dtype)` effectue une conversion numérique contrôlée sur
CPU et conserve la forme.

## Transformer la forme et les vues

Une forme contient des dimensions positives ou nulles. Le scalaire emploie
`[]` et une dimension nulle produit un tenseur vide. `shape()`, `strides()`,
`offset()`, `rank()`, `count()` et `is_contiguous()` inspectent la disposition
sans lire les éléments, y compris sur GPU. Les strides décrivent une
disposition dense row-major pour un nouveau tenseur.

`reshape(shape)` et `flatten()` ne recopient pas un tenseur contigu. Une forme
de `reshape` peut contenir exactement un `-1` à inférer ; les autres dimensions
restent positives ou nulles. Une inférence rendue ambiguë par une cardinalité
nulle est refusée.

`permute(axes)` réordonne tous les axes. `transpose()` est son raccourci pour
les matrices. `select(axis, index)` supprime un axe, tandis que
`narrow(axis, start, count)` conserve l'axe et sélectionne une plage de pas un.
Ces opérations produisent des vues immuables qui partagent le stockage et
gardent celui-ci vivant :

```sx
use Tensor

var values:int32[] = [
    1 as int32, 2 as int32, 3 as int32,
    4 as int32, 5 as int32, 6 as int32,
]

let matrix = Tensor(values, [2, 3])
let columns = matrix.transpose()
let compact = columns.contiguous()

assert(columns.shape()[0] == 3)
assert(columns.strides()[0] == 1)
assert(columns.int32_at([2, 1]) == 6 as int32)
assert(compact.offset() == 0)
```

Les indices sont zéro-based, non négatifs et un accès scalaire fournit un
indice par dimension. `at()` lit `float32`; les variantes typées comme
`int32_at()` suivent le dtype exact. Comme les extracteurs de tableaux, ces
accès sont CPU-only : appelez explicitement `cpu()` avant une lecture GPU.

`contiguous()` conserve un tenseur qui couvre déjà son stockage canonique et
matérialise sinon les valeurs logiques dans un nouveau stockage dense, sans
changer le dtype. Sur GPU, cette matérialisation reste une copie GPU et ne
provoque aucun readback. Les opérations élémentaires de cette version adressent
directement les strides des vues ; `contiguous()` reste disponible lorsqu'un
consommateur exige explicitement un stockage dense canonique.

## Broadcasting et calcul élémentaire

`add`, `subtract`, `multiply` et `divide` alignent les dimensions depuis la
droite. Deux dimensions sont compatibles lorsqu'elles sont égales ou que l'une
vaut `1`; une dimension nulle est donc compatible avec `0` ou `1`. Un scalaire
de forme `[]` suit exactement la même règle :

```sx
use Tensor

var samples:float[] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
var bias:float[] = [10.0, 20.0, 30.0]

let matrix = Tensor(samples, [2, 3])
let shifted = matrix.add(Tensor.vector(bias)).multiply(2.0)

assert(shifted.shape()[0] == 2 && shifted.shape()[1] == 3)
assert(shifted.at([1, 2]) == 72.0)
```

Les deux tenseurs d'une opération binaire gardent le même dtype et le même
placement. Sur CPU, les quatre verbes acceptent les neuf dtypes. Les entiers
conservent leur dtype et leur division entière ; overflow, soustraction non
représentable, division par zéro et quotient signé non représentable échouent
avant le calcul. Un scalaire porte lui aussi son type exact : écrivez par
exemple `tensor.add(1 as int32)` pour un tenseur `int32`.

`negate` accepte `float32` et les entiers signés. `abs` accepte les neuf dtypes
et laisse les non-signés inchangés. `exp`, `log` et `sqrt` sont réservés à
`float32`. Pour `float32`, les règles IEEE restent observables : division par
zéro, NaN et infinis ne sont pas transformés en erreurs Tensor.

Le GPU exécute ces neuf verbes uniquement pour `float32`, y compris sur les
vues stridées et les formes broadcastées de rang 0 à 5. Les entiers peuvent
résider sur le GPU et revenir bit pour bit sur CPU, mais tout calcul entier GPU
échoue avant création de pipeline ou soumission. Les pipelines élémentaires
sont réutilisés au fil d'une chaîne résidente.

## Réductions et algèbre linéaire

`sum`, `mean`, `min` et `max` retournent toujours un `Tensor`. Sans argument,
ils réduisent tous les axes et produisent la forme scalaire `[]`. Le paramètre
optionnel `axes` sélectionne les dimensions à réduire ; `[]` ne réduit rien et
`keep_dimensions:true` remplace chaque dimension réduite par `1` :

```sx
use Tensor

var values:float[] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
let matrix = Tensor(values, [2, 3])

let columns = matrix.sum([0])
let rows = matrix.mean([1], true)

assert(columns.shape()[0] == 3 && columns.values()[2] == 9.0)
assert(rows.shape()[0] == 2 && rows.shape()[1] == 1)
```

`sum`, `min` et `max` acceptent les neuf dtypes ; `mean` est réservé à
`float32`. Une somme de domaine vide vaut zéro. `mean`, `min` et `max` refusent
un domaine vide lorsqu'il existe une valeur de sortie. Les axes hors limites
ou dupliqués sont également refusés. NaN se propage et `min`/`max` choisissent
respectivement `-0.0` et `+0.0` lorsque les deux signes de zéro sont présents.
Les sommes entières restent dans leur dtype et échouent en cas d'overflow.

`dot` accepte deux vecteurs de même longueur et retourne un Tensor scalaire.
`matmul` accepte deux matrices dont les dimensions intérieures correspondent :

```sx
var left:float[] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
var right:float[] = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]

let product = Tensor(left, [2, 3]).matmul(Tensor(right, [3, 2]))
assert(product.at([1, 1]) == 154.0)
```

Les deux opérandes gardent le même dtype et le même placement. Sur CPU,
`dot` et `matmul` acceptent `float32` et les huit dtypes entiers ; les produits
et sommes entiers sont contrôlés dans le dtype. Sur GPU, réductions, `dot` et
`matmul` acceptent `float32` uniquement, rendent un Tensor GPU et ne provoquent
aucun readback implicite. Les comparaisons CPU/GPU flottantes des tests
emploient une tolérance absolue de `1e-5 × nombre de termes` et une tolérance
relative de `1e-5 × |référence|`, en retenant la plus grande.

`matmul` accepte aussi des lots de matrices de rang supérieur ou égal à 2. Les
dimensions de tête suivent le broadcasting habituel et les deux dernières
dimensions portent lignes, colonnes et dimension intérieure. Les versions CPU
et GPU adressent directement les lots et les vues stridées.

## Composer et sélectionner

`Tensor.concatenate(tensors, axis)` joint une liste non vide le long d'un axe.
Les autres dimensions, dtype, placement et device doivent correspondre.
`Tensor.stack(tensors, axis)` ajoute d'abord un axe puis joint des formes
identiques. Les deux opérations restent sur le placement courant et acceptent
les neuf dtypes.

`source.gather(axis, indices)` accepte deux formes d'indices `int32`. La forme
courte omet l'axe choisi et sélectionne une valeur par position restante : des
logits `[batch, classes]` se sélectionnent ainsi avec des classes `[batch]`.
La forme générale donne à `indices` le même rang que la source ; ses dimensions
hors de l'axe choisi doivent correspondre à la source, sa dimension sur l'axe
fixe la taille de sortie, et un même indice peut apparaître plusieurs fois. Sur
GPU, cette opération est l'exception d'adressage `int32` autorisée pour une
source `float32`; elle ne rend pas le calcul entier général disponible.
Les indices doivent appartenir à l'axe sélectionné. Le chemin CPU les valide
avant l'accès ; pour conserver le chemin GPU résident, un indice GPU hors
borne produit un NaN sentinelle dans la sortie au lieu d'un readback caché.

## Calcul neuronal eager

`relu`, `sigmoid`, `tanh`, `softmax(axis)` et `log_softmax(axis)` opèrent sur
des tenseurs `float32`. Softmax et log-softmax soustraient le maximum de l'axe
avant l'exponentielle. Les NaN se propagent ; un axe vide ou invalide échoue
selon les mêmes règles que les réductions.

`layer_norm(dimensions, epsilon)` normalise les dernières dimensions, sans
état entraînable dans cette version du socle. `mse_loss(target)` produit la
MSE scalaire. `cross_entropy(target, axis)` accepte soit une cible dense
`float32` de même forme, soit un Tensor de classes `int32` dont la forme omet
l'axe des classes. Ces opérations composent des Tensor ordinaires et restent
résidentes sur GPU jusqu'à `cpu()`.

La convolution 2D emploie un layout public unique : entrée NCHW et kernel
OIHW. `conv2d(kernel, bias, stride, padding)` accepte un stride scalaire et un
padding symétrique. `max_pool2d` et `average_pool2d` emploient une fenêtre
carrée ; le stride vaut la taille de fenêtre par défaut. Le pooling moyen
compte les cellules de padding comme des zéros dans son diviseur. Dilation,
groupes et convolution transposée ne font pas partie de 0.1.0.

```sx
use Tensor

let logits = Tensor.ones([4, 10]).matmul(Tensor.ones([10, 3]))
let probabilities = logits.softmax(1)
var labels:int32[] = [0 as int32, 1 as int32, 2 as int32, 0 as int32]
let classes = probabilities.gather(1, Tensor.vector(labels))

let features = Tensor.ones([1, 3, 16, 16])
let kernels = Tensor.ones([8, 3, 3, 3])
let pooled = features.conv2d(kernels, stride:1, padding:1).relu().max_pool2d(2)
```

## Initialiser et appliquer le dropout

Les constructeurs `uniform`, `normal`, `xavier_uniform` et `he_uniform`
reçoivent un `STD.Randomizer` explicite. Une même seed et la même séquence
d'appels reproduisent les valeurs ; un appel suivant avance la source. Les
initialisations sont créées sur CPU puis transférées explicitement au besoin.

```sx
use STD.Randomizer
use Tensor

var randomizer = Randomizer(42)
let weights = Tensor.xavier_uniform([64, 128], randomizer)
let noise = Tensor.normal([64], randomizer, standard_deviation:0.01)
```

`dropout(probability, seed, iteration, training)` exige une seed et rend
l'itération explicite. Le même couple seed-itération reproduit le masque ; une
autre itération le renouvelle. En évaluation, `training:false` conserve le
Tensor. Sur GPU, le masque est généré par un compteur privé dans la passe de
calcul : aucun upload de masque ni RNG global n'est caché.

## Différencier un calcul eager

Le module `Tensor.Autograd` fournit l'autodifférentiation inverse de `float32`.
Une `Autograd.Variable` est une feuille mutable qui possède sa valeur Tensor et
son éventuel gradient ; les opérations continuent de produire des `Tensor`
immuables ordinaires :

```sx
use Tensor
use Tensor.Autograd

var samples:float[] = [1.0, 2.0, 3.0, 4.0]
var targets:float[] = [0.0, 1.0]
var weights = Autograd.Variable(Tensor(samples, [2, 2]))

let prediction = Tensor.ones([1, 2]).matmul(weights.value())
let loss = prediction.mse_loss(Tensor.vector(targets))
loss.backward()

if let gradient = weights.gradient() {
    assert(gradient.shape()[0] == 2)
}
weights.zero_grad()
```

`backward()` sans argument exige une sortie scalaire et injecte une seed de
un. Une sortie non scalaire reçoit explicitement un Tensor seed de même forme,
dtype, placement et device. Les contributions d'une feuille répétée ou d'un
graphe ramifié s'additionnent ; les appels issus de graphes distincts
s'accumulent également jusqu'à `zero_grad()`. Une feuille déconnectée conserve
un gradient absent (`null`).

`detach()` rend la même valeur sans provenance. Le graphe d'une sortie est
consommé après un `backward()` réussi : il n'existe ni rétention du graphe ni
dérivée d'ordre supérieur dans cette version. Les changements de dtype ou de
placement sont refusés au milieu d'un graphe suivi ; détachez d'abord la
valeur. Sur GPU, le forward, les gradients et leur accumulation restent
résidents. Pour observer un résultat, employez par exemple
`gradient.detach().cpu()` ; ce `cpu()` demeure le readback explicite.

## Entraîner des paramètres nommés

`Tensor.NN.Parameter` associe un nom stable à une valeur `float32` suivie et à
son gradient éventuel. Le Tensor reste immuable : une étape d'optimisation
remplace la valeur du paramètre par une nouvelle feuille détachée.

```sx
use Tensor
use Tensor.NN
use Tensor.Optim

var initial:float[] = [1.0, -2.0]
var parameters:NN.Parameter[] = [
    NN.Parameter("linear.weight", Tensor.vector(initial))
]
var optimizer = Optim.Adam(parameters, learning_rate:0.01)

optimizer.zero_grad()
parameters[0].value().multiply(parameters[0].value()).sum().backward()
let norm = optimizer.clip_grad_norm(1.0)
optimizer.step()
```

`SGD` accepte `learning_rate`, `momentum` et `weight_decay`. Son momentum suit
la convention PyTorch : le premier buffer reçoit le gradient après décroissance
L2, puis les étapes suivantes calculent `momentum * buffer + gradient`.
`Adam` accepte `learning_rate`, `beta1`, `beta2`, `epsilon` et `weight_decay` ;
il conserve un pas et deux moments par paramètre, applique les corrections de
biais puis une décroissance L2 couplée. Cette opération est Adam, pas AdamW.

`zero_grad()` supprime les accumulateurs sans recréer les valeurs. `step()`
ignore un paramètre dont le gradient est absent et laisse alors son état
d'optimiseur inchangé. Les noms dupliqués, hyperparamètres invalides et
gradients incompatibles échouent avant toute modification de la collection.

`clip_grad_norm(maximum_norm, nonfinite)` calcule une unique norme L2 sur tous
les gradients disponibles et applique le facteur
`min(1, maximum_norm / (norme + 1e-6))`, comme la référence PyTorch. La
politique `NonFiniteGradientPolicy.propagate`, choisie par défaut, conserve les
NaN ou infinis sur le placement courant. La politique `reject` diagnostique la
norme avant toute mutation ; sur GPU, ce choix explicite lit le scalaire et
synchronise donc l'hôte.

Valeurs, gradients, momentum et moments restent sur le même CPU ou device GPU.
Un paramètre peut appeler `to(device)` ou `cpu()` après `zero_grad()` tant que
l'optimiseur n'a encore créé aucun état. Après initialisation de cet état, un
changement de placement est refusé atomiquement à l'étape suivante plutôt que
de migrer silencieusement une partie du modèle.

## Composer et entraîner un réseau

`Tensor.NN.Layer` est le contrat commun des couches et `Sequential` les exécute
dans l'ordre tout en collectant récursivement leurs paramètres nommés. Les
couches 0.1.0 sont `Dense`, `Conv2D`, `MaxPool2D`, `AveragePool2D`, `Flatten`,
`Dropout`, `LayerNorm` et `SimpleRNN`. Les activations restent des opérations
Tensor ; `Activation.relu()`, `sigmoid()` et `tanh()` servent uniquement à les
placer dans une composition.

```sx
var layers:NN.Layer[] = [
    NN.Dense("hidden", 2, 8, 101),
    NN.Activation.tanh(),
    NN.Dense("classifier", 8, 2, 102)
]
var model = NN.Sequential(layers)
var parameters = model.parameters()
var optimizer = Optim.Adam(parameters, learning_rate:0.04)

optimizer.zero_grad()
let loss = model.forward(input).cross_entropy(target, 1)
loss.backward()
optimizer.step()
```

Les poids de `Dense` suivent `[entrées, sorties]`. `Conv2D` attend NCHW et
stocke ses noyaux OIHW. `SimpleRNN` accepte `[batch, temps, features]`, déroule
une récurrence tanh unidirectionnelle et retourne l'état final. `train()` et
`eval()` se propagent dans `Sequential` ; seul `Dropout` change alors son
comportement. `model.to(device)` et `model.cpu()` déplacent tous les paramètres
après `zero_grad()` selon les règles de l'optimiseur.

`NN.Checkpoint.save(model, path)` écrit un document texte déterministe de
schéma 1 avec noms UTF-8, dtype, formes et valeurs. La sauvegarde d'un modèle
GPU effectue nécessairement le readback explicite vers le fichier.
`NN.Checkpoint.load(model, path)` valide le document et l'ensemble complet des
paramètres avant la première mutation. Le modèle doit être sur CPU pendant la
lecture ; appelez ensuite explicitement `model.to(device)` si nécessaire. Le
checkpoint ne contient ni code, ni graphe autograd, ni état d'optimiseur.

La composition personnalisée peut rester un type ou une fonction applicative
qui assemble les opérations Tensor publiques. La version 0.1.0 ne fournit pas
BatchNorm, convolution groupée ou transposée, LSTM/GRU, embedding ni attention.

## Passer sur GPU

Créez le device avec `GFX.GPU`, puis placez explicitement le tenseur :

```sx
use GFX.GPU
use Tensor
use Tensor.DType

var device = GPU.Device()
var values:float[] = [1.0, 2.0, 3.0]

let gpu = Tensor.vector(values).to(device)
let result = gpu.add(2.0).multiply(3.0)

assert(result.is_gpu())
let cpu = result.cpu()
assert(cpu.values()[0] == 9.0)
```

Les métadonnées de forme et de placement ne provoquent aucun transfert.
`cpu()` est le premier point de synchronisation d'une chaîne GPU. Les valeurs,
items et accès scalaires sont CPU-only.

Cette version exécute sur GPU les opérations élémentaires binaires et unaires
décrites ci-dessus pour `float32`. Les neuf dtypes se transfèrent sans perte,
mais un calcul GPU entier est refusé avant soumission. Une migration entre deux
devices reste explicite : `tensor.cpu().to(other_device)`.

## Choisir CPU ou GPU

La [campagne Release publique](https://github.com/Matanek/Silex-Benchmarks/tree/main/Sources/TensorStableCompute)
du 8 septembre 2026 a mesuré Tensor sur un Apple M3 Pro de 18 Gio, sous macOS
26.6.2 et Metal. Ces seuils décrivent uniquement cette machine et les commits
Tensor `bdcd062`, GFX.GPU `bb23787` et Silex `1c310ce` ; ils ne prédisent pas un
autre CPU, GPU, driver ou backend.

Pour un calcul déjà résident, le GPU chaud dépasse le CPU à partir de 65 536
éléments pour l'addition et la somme globale dans la grille mesurée. Le
`matmul` carré dépasse le CPU dès le côté 32, mais son premier passage inclut la
création du pipeline. La chaîne `add -> multiply -> matmul -> sum -> add` reste
plus lente au côté 32, transferts compris ; son croisement observé est le côté
128. À ce point, la médiane vaut 91,889 ms sur CPU, 11,892 ms en calcul GPU
chaud et 18,268 ms avec les deux uploads et le download final, soit 5,03× sur
le parcours complet. Le rapport conserve les plages complètes et les régimes
CPU très dispersés sans retirer d'échantillon ; les facteurs d'accélération ne
doivent donc pas être généralisés.

Pour 262 144 octets, les débits médians observés incluent conversion typée,
allocation, soumission et attente :

| Dtype | Upload Mio/s | Download Mio/s |
| --- | ---: | ---: |
| `float32` | 30,99 | 73,13 |
| `int8` | 22,50 | 23,73 |
| `uint8` | 19,97 | 22,67 |
| `int16` | 25,70 | 39,99 |
| `uint16` | 26,48 | 41,68 |
| `int32` | 36,17 | 62,68 |
| `uint32` | 28,84 | 69,71 |
| `int64` | 41,36 | 89,63 |
| `uint64` | 39,39 | 93,89 |

Gardez donc les petites opérations isolées sur CPU. Le GPU devient intéressant
lorsque les entrées restent résidentes sur plusieurs opérations ou que le
calcul amortit explicitement l'upload et le download. Ces mesures de transfert
ne revendiquent aucune accélération du calcul entier sur GPU.

## Sémantique de valeur

Un `Tensor` peut rester dans un `let`. Une affectation ordinaire partage son
stockage immuable et sa durée de vie sûre. Une copie profonde avec `copy` est
refusée lorsque le tenseur atteint une ressource GPU non clonable ; aucun
handle natif n'est dupliqué.

## Développement

Depuis la racine de `SilexProject` :

```text
silex test Packages/Tensor/Tests/Consumer
silex test Packages/Tensor/Tests/PerformanceGuards.sx
silex check Packages/Tensor
```

La suite ordinaire est hermétique. Elle lit le fixture Silex committé, sans
Python, téléchargement ni accès réseau. Ce corpus couvre les neuf dtypes et
sépare les résultats exacts, les calculs `float32` tolérés, les transferts GPU
bit-exacts et les erreurs attendues. Les comparaisons traitent explicitement
NaN, les infinis et le signe de zéro ; les tolérances absolue et relative sont
fixées par famille d'opérations dans le
[rapport différentiel](../../Tools/Oracle/REPORT.md).

La régénération est une opération de maintenance volontaire. Elle emploie les
versions épinglées de NumPy, PyTorch, TensorFlow et JAX dans un environnement
Python jetable, exige l'accord d'au moins deux oracles par résultat numérique
commun, puis vérifie le diff avant acceptation. Le
[guide du générateur](../../Tools/Oracle/README.md) donne les commandes exactes,
la seed et les règles de licence. Une mise à jour de framework ne modifie jamais
automatiquement le contrat Tensor 0.1.0.
