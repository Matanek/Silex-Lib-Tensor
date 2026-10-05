# Sélectionner des valeurs avec un masque

Utilisez une comparaison pour choisir les valeurs qui participent à un calcul.
`Tensor.where` conserve une forme broadcast et différentie les deux branches
numériques. `masked_select` compacte les valeurs sur CPU.

Ce programme conserve les valeurs positives et remplace les autres par zéro :

```sx
use Tensor
use Tensor.Autograd.Variable

func main() {
    var values:float[] = [-2.0, 1.0, 3.0]
    var input = Variable(Tensor.vector(values))
    let mask = input.value().greater_than(0.0)
    let selected = Tensor.where(mask, input.value(), Tensor.zeros([3]))
    selected.sum().backward()
    let result = selected.detach().values()
    assert(result[0] == 0.0 && result[1] == 1.0 && result[2] == 3.0)
    if let gradient = input.gradient() {
        let gradient_values = gradient.values()
        assert(gradient_values[0] == 0.0 && gradient_values[1] == 1.0 && gradient_values[2] == 1.0)
    } else { panic("expected gradient") }
    let positive = input.value().detach().masked_select(mask).values()
    assert(positive.count() == 2 && positive[0] == 1.0 && positive[1] == 3.0)
}
```

Le masque est une valeur booléenne détachée. La comparaison ne crée aucun
chemin de gradient ; les branches numériques de `where` reçoivent le gradient
aux positions où elles sont choisies. Le broadcast additionne les contributions
sur les dimensions répétées, et deux branches partageant une feuille cumulent
leurs contributions. Une valeur NaN non sélectionnée ne contamine ni la sortie
ni les gradients calculés par `where`.

## Construire et combiner les masques

Un tableau Silex `bool[]` construit un Tensor de `DType.boolean()`. Les
constructeurs `scalar`, `vector`, `matrix`, `full` et forme-valeurs acceptent
les booléens. `zeros(shape, DType.boolean())` produit `false`, et `ones`
produit `true`. `bool_values()`, `bool_item()` et `bool_at(indices)` sont les
extractions CPU exactes. Sur GPU, appelez d'abord `cpu()`.

Les comparaisons `equal`, `not_equal`, `less_than`, `less_equal`,
`greater_than` et `greater_equal` acceptent un Tensor ou un scalaire du même
dtype. Les neuf dtypes numériques sont disponibles sur CPU ; les comparaisons
GPU sont disponibles en `float32`. L'égalité et l'inégalité acceptent aussi
les booléens. `logical_not`, `logical_and` et `logical_or` travaillent sur les
masques ; `logical_and/or` acceptent aussi un scalaire `bool`.

Toutes ces opérations suivent le broadcast par alignement à droite et
conservent les vues stridées. Pour les flottants, NaN n'est égal à rien et
rend les comparaisons ordonnées fausses ; `not_equal` est vrai avec NaN.
Les infinis suivent l'ordre usuel, et les deux signes de zéro sont égaux.
Les entiers 64 bits sont comparés exactement, sans passage par `float32`.

Les booléens possèdent un stockage logique privé d'un octet par élément.
`is_boolean()` les distingue des nombres ; `is_integer()` et `is_floating()`
renvoient tous deux `false`. Les vues, transferts exacts, `contiguous`,
`concatenate` et `stack` conservent ce dtype. L'arithmétique, les réductions
numériques et les comparaisons ordonnées refusent les booléens. `cast` ne
convertit pas entre booléens et nombres ; aucune conversion implicite n'existe.

## Choisir une forme et un placement

`Tensor.where(mask, when_true, when_false)` demande un masque booléen et deux
branches numériques de même dtype. Les trois formes sont broadcast ensemble.
Les trois valeurs restent sur le même placement et, sur GPU, le même device.
Les branches CPU acceptent les neuf dtypes numériques ; les branches GPU
acceptent `float32`. Les calculs GPU et les uniformes scalaires n'effectuent
aucun transfert caché. Le GPU accepte les rangs jusqu'à cinq et les tailles de
buffer représentables en `uint32` ; une forme vide ne soumet aucun calcul.

`source.masked_select(mask)` broadcast les deux formes puis retourne un vecteur
des valeurs retenues dans l'ordre logique, même depuis une vue. Un masque tout
faux retourne un vecteur vide. Le gradient de la source `float32` cumule les
répétitions dues au broadcast. Le gather CPU accepte aussi une source booléenne.
La cardinalité de la source doit tenir dans les indices `int32`.

La compaction dynamique reste CPU : sur GPU, elle échoue avant création du
pipeline ou soumission. Pour la demander explicitement, utilisez
`source.cpu().masked_select(mask.cpu())`, avec `detach()` préalable si la source
est suivie. Pour préserver une chaîne GPU différentiable, utilisez `where` avec
une forme connue.

Consultez la [matrice de compatibilité](Reference.md) pour les autres opérations
et le [profilage GPU](Profiling.md) pour observer les transferts explicites.
