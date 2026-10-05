# Conserver un buffer et geler un paramètre

Un `Neural.Buffer` conserve une donnée du modèle qui ne doit pas être optimisée.
Un `Neural.Parameter` conserve un poids entraînable ; `freeze()` suspend son
entraînement et `unfreeze()` le reprend avec les moments existants de l’optimiseur.

Ce programme entraîne un poids, le gèle, change son coefficient et reprend :

```sx
use Tensor
use Tensor.Neural
use Tensor.Optim
use STD.Math

func main() {
    var weight = Neural.Parameter("weight",Tensor.scalar(2.0))
    var scale = Neural.Buffer("scale",Tensor.scalar(3.0))
    var optimizer = Optim.SGD([weight],learning_rate:0.1,momentum:0.5)
    weight.value().multiply(scale.value()).sum().backward()
    optimizer.step(); optimizer.zero_grad()
    assert(Math.abs(weight.value().item()-1.7) < 0.00001 && scale.value().item() == 3.0)
    weight.freeze()
    assert(!weight.is_trainable())
    optimizer.step()
    assert(Math.abs(weight.value().item()-1.7) < 0.00001)
    scale.replace(Tensor.scalar(4.0))
    weight.unfreeze()
    weight.value().multiply(scale.value()).sum().backward()
    optimizer.step()
    assert(Math.abs(weight.value().item()-1.15) < 0.00001)
}
```

Le premier pas donne un poids de 1,7. Le gel conserve cette valeur. Après le
remplacement du coefficient et le dégel, SGD reprend avec son momentum conservé
et donne environ 1,15. Le nom et l’identité du poids restent identiques.

## Garder les données hors du graphe

`Buffer(name, value)` exige un nom non vide et détache la valeur reçue. `name()`
et `value()` donnent son nom stable et sa valeur courante. Les neuf dtypes
numériques et les booléens sont acceptés, sur CPU et GPU. Un Buffer n’est pas
un Parameter : les optimiseurs acceptent uniquement des Parameter.

`replace(value)` conserve dtype, forme et placement, y compris le device GPU,
et détache la nouvelle valeur avant de la stocker. La valeur Tensor précédente
reste valide : son stockage est immuable. Plusieurs références au même Buffer
observent le remplacement par son identité partagée.

`to(device)` et `cpu()` déplacent explicitement la valeur du Buffer. Les
transferts restent exacts pour tous les dtypes ; les entiers et booléens ne
bénéficient pas d’un calcul numérique GPU supplémentaire. L’extraction des
éléments exige une valeur CPU.

## Suspendre et reprendre l’entraînement

Un Parameter est entraînable à sa création. `is_trainable()` donne son état.
Changer cet état exige un gradient vide : appelez `parameter.zero_grad()` ou
`optimizer.zero_grad()` avant `freeze()` ou `unfreeze()`. Répéter l’état déjà
établi est sans effet, même si un gradient est présent.

Après le gel, les nouveaux appels à `value()` donnent une valeur détachée.
Les gradients des autres entrées continuent à se calculer. Un graphe construit
avant le gel garde son sens et peut encore remplir le slot du Parameter ;
`gradient()` reste observable et `zero_grad()` le vide normalement.

SGD et Adam ignorent les paramètres gelés, y compris leurs gradients antérieurs :
aucune mise à jour, décroissance, création ou évolution de moments, ni incrément
du compteur Adam. `clip_grad_norm` exclut leurs gradients de la norme et laisse
leurs slots inchangés. Le dégel conserve les états existants et reprend au
prochain pas actif. Les contrôles de placement des états d’optimiseur restent
applicables ; le gel ne permet pas de déplacer séparément des moments existants.

`replace`, `to(device)` et `cpu()` restent des actions explicites sur Parameter,
avec leurs contrôles habituels de gradient vide, dtype, forme et placement.
Pour observer sa valeur GPU, utilisez `parameter.value().detach().cpu()`.

La [composition des couches](Composition.md) présente les hooks à relayer
manuellement. Ces primitives n’enregistrent pas automatiquement l’état d’une
couche ; le checkpoint actuel sauvegarde uniquement ses paramètres, sans
buffers, état de gel ni moments d’optimiseur.
