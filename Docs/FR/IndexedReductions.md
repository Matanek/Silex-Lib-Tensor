# Trouver et utiliser l’indice d’un extremum

`argmin` et `argmax` trouvent la position du minimum ou du maximum. Utilisez
cet indice avec `gather` pour sélectionner une valeur et conserver son gradient.
Le premier indice gagne lorsque plusieurs valeurs sont égales.

Ce programme choisit la première observation maximale :

```sx
use Tensor
use Tensor.Autograd

func main() {
    var observations:float[] = [2.0,7.0,7.0]
    var input = Autograd.Variable(Tensor.vector(observations))
    let index = input.value().argmax().reshape([1])
    let selected = input.value().gather(0,index).sum()
    assert(index.int32_values()[0] == 1 && selected.detach().item() == 7.0)
    var gradients = Autograd.grad(selected,[input.value()])
    let actual = gradients[0].values()
    assert(actual[0] == 0.0 && actual[1] == 1.0 && actual[2] == 0.0)
    assert(input.gradient() == null)
}
```

L’indice vaut 1, la valeur sélectionnée 7 et son gradient vaut 1 uniquement à la
position 1. Les indices sont des données int32 détachées ; le gradient traverse
`gather` vers les observations, sans changer le slot de la Variable.

## Choisir le domaine et la forme

`argmin(axis:int?=null, keep_dimensions:bool=false)` et `argmax` partagent les
mêmes règles. Sans axe, elles retournent l’indice du tenseur aplati dans l’ordre
logique row-major, même pour une vue non contiguë. Avec un axe, elles retournent
la coordonnée sur cet axe, pour chaque combinaison des autres coordonnées.
L’axe doit être non négatif et inférieur au rang.

Les dimensions réduites disparaissent ; `keep_dimensions:true` les conserve
avec une taille de 1. Sans axe, la sortie est scalaire, ou conserve tous les axes
unitaires si demandé. Un scalaire d’entrée donne l’indice 0. Un domaine réduit
vide est refusé, y compris lorsque la sortie serait vide. Une sortie vide est
acceptée si le domaine réduit est non vide : `[0,3]` avec axis 1 donne `[0]`.

## Prévoir les égalités et valeurs non finies

Les égalités conservent le premier indice logique ; +0 et −0 sont égaux.
Si le domaine contient des NaN, son premier NaN est retenu pour les deux
opérations. Sinon les infinis suivent leur ordre numérique habituel.
Cette règle d’indices diffère du choix du signe de zéro de `min` et `max`.

## Choisir le placement

Les neuf dtypes numériques CPU sont acceptés, avec comparaison entière exacte
jusqu’à int64/uint64. Les booléens sont refusés. Sur GPU, l’entrée doit être
float32 et la sortie reste sur le même device, en int32, sans transfert implicite.
Lisez-la par `result.cpu().int32_values()` lorsque nécessaire. Les vues avec
offset, strides et dimensions étendues restent prises en charge ; le GPU garde
sa limite de rang 5 et de métadonnées uint32.

Le dernier indice doit tenir dans int32 : au plus 2^31 éléments dans le domaine.
Les axes, tailles et métadonnées sont contrôlés avant allocation/soumission.
Les indices ne se différencient pas ; les valeurs sélectionnées peuvent se
différencier. Aucune performance supérieure à une réduction de valeurs n’est
promise. Consultez le [guide des gradients](Gradients.md) et les
[formes et vues](Shapes.md) pour composer la sélection.
