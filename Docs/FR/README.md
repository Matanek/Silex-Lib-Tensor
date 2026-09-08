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
leur largeur et leur famille numérique. `zeros`, `ones` et les surcharges de
`full` complètent les constructions usuelles.

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
provoque aucun readback. Une vue stridée doit être rendue contiguë avant une
opération GPU qui exige une disposition canonique.

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

Cette version exécute sur GPU seulement `add(float)` et `multiply(float)` pour
`float32`. Les neuf dtypes se transfèrent sans perte, mais un calcul GPU entier
est refusé avant soumission. Une migration entre deux devices reste explicite :
`tensor.cpu().to(other_device)`.

## Sémantique de valeur

Un `Tensor` peut rester dans un `let`. Une affectation ordinaire partage son
stockage immuable et sa durée de vie sûre. Une copie profonde avec `copy` est
refusée lorsque le tenseur atteint une ressource GPU non clonable ; aucun
handle natif n'est dupliqué.

## Développement

Depuis la racine de `SilexProject` :

```text
silex test Packages/Tensor/Tests/Consumer
silex check Packages/Tensor
```
