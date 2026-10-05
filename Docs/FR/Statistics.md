# Mesurer une dispersion avec correction explicite

`variance` et `standard_deviation` mesurent la dispersion de valeurs float32
sur CPU/GPU. Indiquez toujours la correction statistique : zéro pour la
population, un pour l’estimateur d’échantillon usuel.

Ce programme contrôle une variance et son gradient :

```sx
use Tensor
use Tensor.Autograd
use STD.Math

func main() {
    var data:float[] = [1.0,2.0,3.0]
    var input = Autograd.Variable(Tensor.vector(data))
    let population = input.value().variance(correction:0)
    assert(Math.abs(population.detach().item()-2.0/3.0) < 0.001)
    let sample = input.value().variance(correction:1)
    var gradients = Autograd.grad(sample,[input.value()])
    assert(gradients[0].values()[0] == -1.0 && gradients[0].values()[2] == 1.0)
    assert(input.gradient() == null)
    assert(Tensor.vector(data).standard_deviation(correction:1).item() == 1.0)
}
```

Le domaine contient trois valeurs, de moyenne deux. La somme des écarts au
carré vaut deux : variance de population 2/3, variance d’échantillon un et
écart-type d’échantillon un. `Autograd.grad` conserve le slot absent.

## Choisir le domaine et la forme

`variance(correction:int, axes:null, keep_dimensions:false)` et
`standard_deviation(correction:int, axes:null, keep_dimensions:false)` utilisent
les mêmes conventions d’axes que `sum` et `mean`. Une liste absente réduit tous
les axes ; une liste vide traite chaque valeur comme un domaine singleton.
Les axes explicites sont non négatifs, uniques et dans le rang. Avec
`keep_dimensions:true`, chaque axe réduit garde une taille un.

La correction doit être un entier non négatif. Le nombre N de valeurs par
domaine doit être strictement supérieur à la correction, même quand la forme
résultante est vide. Un domaine vide est donc refusé ; une sortie vide reste
possible lorsque les domaines sont valides, par exemple [0,3] réduit sur l’axe1.
Scalaires et singletons acceptent correction0 et donnent zéro si leurs valeurs
sont finies. Les entiers exigent un cast float32 explicite avant calcul. Les booléens
sont refusés et leur cast vers un nombre reste interdit.

## Valeurs et gradients

La variance calcule d’abord la moyenne, puis les écarts centrés, et divise la
somme de leurs carrés par `N - correction`. Elle évite la formule par différence
de moments, sans promettre une précision supérieure aux calculs float32.
L’écart-type est sa racine carrée. Aucun epsilon, clipping ou correction
implicite n’est ajouté. NaN et infinis suivent les opérations float32 : un
domaine contenant une valeur infinie ou NaN donne ici une dispersion NaN.

Le gradient de variance vaut `2 * (x - mean) / (N - correction)` ; celui
d’écart-type vaut `(x - mean) / ((N - correction) * standard_deviation)` lorsque
l’écart-type est positif. Lorsque la variance calculée vaut zéro, comme sur un domaine constant usuel,
la dérivée de l’écart-type est non unique ; Tensor choisit zéro pour toutes les entrées. Les gradchecks se
font hors de ce point. Une valeur non finie reste observable, sans masquage.

Les vues stridées, offsets, expansions et plusieurs axes sont pris en charge.
Les gradients restent de premier ordre ; `backward` accumule dans les slots,
`Autograd.grad` les préserve. Sur GPU, calculs et gradients float32 restent
résidents, dans les limites existantes de rang jusqu’à cinq, sans upload ni
readback implicite. Observez une valeur suivie avec `detach().cpu()` et un
gradient retourné avec `cpu()`.

Consultez les [formes de lots](Shapes.md), le [contrôle des gradients](Gradients.md)
et la [référence de compatibilité](Reference.md).
