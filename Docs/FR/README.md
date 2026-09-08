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

`shape()`, `rank()`, `count()`, `dtype()`, `is_cpu()` et `is_gpu()` ne
provoquent aucun transfert. `cpu()` est le premier point de synchronisation
d'une chaîne GPU. Les valeurs et items sont CPU-only.

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
