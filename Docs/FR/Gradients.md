# Calculer et vérifier un gradient sans modifier les slots

`Autograd.grad` retourne des gradients détachés sans changer les gradients
stockés dans les `Variable` ou les `Parameter`. `Autograd.gradcheck` compare
les dérivées d'une fonction Silex aux différences centrales sur CPU.

Ce programme calcule un gradient, conserve celui déjà stocké et contrôle une
fonction quadratique :

```sx
use Tensor
use Tensor.Autograd

func energy(inputs:Tensor[]) Tensor {
    return inputs[0].multiply(inputs[0]).sum()
}

func main() {
    var data:float[] = [1.0, 2.0, 3.0]
    var input = Autograd.Variable(Tensor.vector(data))
    input.value().multiply(7.0).sum().backward()
    var gradients = Autograd.grad(energy([input.value()]), [input.value()])
    let values = gradients[0].values()
    assert(values[0] == 2.0 && values[1] == 4.0 && values[2] == 6.0)
    if let stored = input.gradient() { assert(stored.values()[0] == 7.0) }
    else { panic("expected stored gradient") }
    let check = Autograd.gradcheck(energy, [Tensor.vector(data)],
        epsilon:0.001, absolute_tolerance:0.003, relative_tolerance:0.001)
    assert(check.passed && check.checked_derivatives == 3)
}
```

## Demander les dérivées d'un graphe

`Autograd.grad(output, inputs, seed:null, allow_unused:false)` reçoit une liste
non vide de Tensor suivis : valeurs d'une Variable, d'un Parameter, ou résultats
intermédiaires. Les résultats respectent l'ordre, la forme et le placement des
entrées demandées. Répéter une entrée répète son gradient ; cela ne multiplie
pas sa dérivée. Les slots existants restent identiques, les slots absents restent
absents, et les résultats ne permettent pas de dérivée d'ordre supérieur.

Sans seed, la sortie doit être scalaire et reçoit un gradient initial de un.
Pour une sortie de forme quelconque, fournissez une seed `float32` de même forme,
placement et device. Le résultat est le produit du Jacobien transposé par cette
seed. La seed est traitée comme une donnée détachée. Le calcul CPU/GPU utilise
les mêmes règles de premier ordre que `backward`, y compris vues, broadcast,
branches partagées et sélection par masque.

Une entrée inutilisée échoue par défaut. Avec `allow_unused:true`, elle reçoit
un vrai zéro sur son propre placement ; une sortie constante devient aussi
possible. Une valeur non suivie n'est pas une entrée différentiable, même avec
cette option. Sur GPU, les calculs restent résidents et n'effectuent aucun
transfert implicite ; utilisez `cpu()` pour observer un résultat.

Le graphe d'opérations est consommé après réussite, comme avec `backward`. Il ne
peut plus servir à un autre `grad` ou `backward`. Les feuilles restent disponibles
pour construire un nouveau graphe. La validation de la provenance, de la seed,
des demandes inutilisées et des nœuds déjà consommés précède toute consommation ;
le calcul fonctionnel ne modifie aucun slot, même en cas de rejet.

## Contrôler une fonction numérique

`Autograd.gradcheck(evaluate, inputs, epsilon, absolute_tolerance,
relative_tolerance)` reçoit un callback `func(Tensor[]) Tensor`, des entrées
`float32` CPU et trois paramètres numériques explicites. Le callback doit être
déterministe, dépendre de ses données d'entrée et garder la même forme de sortie,
sans état externe mutable. La sortie peut être scalaire ou vectorielle, voire
une autre forme : le contrôle porte sur le Jacobien complet.

Le contrôle compare chaque dérivée analytique au quotient central
`(f(x + epsilon) - f(x - epsilon)) / (2 * epsilon)` avec la règle
`abs(a - n) <= absolute_tolerance + relative_tolerance * max(abs(a), abs(n))`.
Choisissez epsilon et les tolérances pour les arrondis `float32`, et restez hors
points non lisses ou frontières de masques. Epsilon doit être fini, positif,
avoir un double fini et produire deux perturbations finies distinctes de la
valeur initiale. Les tolérances sont finies et non négatives ; les données
initiales sont finies.

`GradcheckResult` expose `passed`, `checked_derivatives` et `failure`. Le contrôle
s'arrête au premier écart. `GradcheckFailure` donne les indices logiques `input`,
`element`, `output`, puis les valeurs `analytical` et `numerical`. Les éléments
et sorties sont indexés dans l'ordre logique aplati, dès zéro. Une dérivée non
finie produit un résultat en échec, tandis qu'un dtype, placement, paramètre ou
changement de forme invalide produit un diagnostic. Une sortie vide réussit
avec zéro dérivée contrôlée ; ce résultat n'établit aucune dérivée non vide.

Les sondes et feuilles temporaires sont indépendantes des entrées fournies.
Le contrôle ne change pas leurs slots. Pour N valeurs d'entrée et M valeurs de
sortie, il contrôle jusqu'à N × M dérivées et réévalue le callback pour chaque
sonde : utilisez de petits cas. Aucun Tensor GPU n'est téléchargé implicitement.
Pour comparer des gradients GPU, appelez `grad` sur GPU et observez ses résultats
par transferts explicites, puis comparez-les à un contrôle CPU indépendant.

Consultez les [masques différentiables](Masks.md) et la
[référence de compatibilité](Reference.md) pour choisir les opérations du graphe.
