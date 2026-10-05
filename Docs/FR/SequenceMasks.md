# Construire la visibilité des séquences

Déclarez les longueurs valides de votre lot avec `Neural.sequence_mask`, puis
composez-les avec `Neural.causal_mask`. Les deux fonctions retournent des masques
booléens CPU ; `true` autorise une position. Elles n’extraient aucune donnée d’un
Tensor GPU. Les longueurs proviennent des métadonnées connues par l’application.

L’exemple compose trois séquences de longueurs 0, 2 et 4 avec trois positions de
query. Les clés de chaque séquence forment un préfixe valide ; la causalité
empêche une query de voir une clé future.

```sx
use Tensor
use Tensor.Neural

func main() {
    let lengths:int[]=[0,2,4]
    let keys=Neural.sequence_mask(lengths,maximum_length:4)
    let causal=Neural.causal_mask(query_length:3,key_length:4)
    let visible=keys.unsqueeze(1).logical_and(causal.unsqueeze(0))
    let scores=Tensor.ones([3,3,4])
    let selected=Tensor.where(visible,scores,Tensor.zeros([3,3,4]))
    assert(selected.shape()[0]==3 && selected.shape()[1]==3 && selected.shape()[2]==4)
    assert(selected.sum().item()==11.0)
    let continuation=Neural.causal_mask(query_length:2,key_length:5,query_offset:2)
    assert(continuation.bool_at([0,2]) && !continuation.bool_at([0,3]))
    print("Sequence visibility: lengths and causal positions composed")
}
```

La visibilité obtenue a la forme `[batch,queries,keys]`. Sa sélection contient
11 valeurs égales à 1 ; les autres valent 0. L’offset 2 permet à la première
query du fragment suivant de voir les clés de positions 0, 1 et 2.

## Longueurs et positions explicites

| Appel | Forme du résultat CPU | Position autorisée |
|---|---|---|
| `Neural.sequence_mask(lengths, maximum_length)` | `[lengths.count(),maximum_length]` | `position < lengths[batch]` |
| `Neural.causal_mask(query_length, key_length, query_offset:0)` | `[query_length,key_length]` | `key <= query + query_offset` |

Les dimensions et l’offset doivent être non négatifs. Chaque longueur doit être
comprise entre 0 et `maximum_length`, inclus. Le produit des dimensions et la
dernière position de query doivent tenir dans `int` ; ils sont contrôlés avant
construction. Ce contrôle de position s’applique même avec zéro clé. Une query
unique à l’offset maximal reste valide.

Une longueur nulle produit une ligne entièrement `false`. Un lot vide, zéro
query ou zéro clé préserve les deux axes annoncés avec zéro élément. La causalité
rectangulaire utilise toujours les positions de query `query_offset + query` ;
l’alignement n’est pas déduit de la différence entre nombres de queries et clés.

Le masque de longueurs décrit un préfixe contigu. Si votre padding comporte des
trous, fournissez votre masque booléen explicite plutôt qu’une longueur. Un
[Embedding](Embeddings.md) et son indice de padding ne construisent pas ce masque
automatiquement. Les positions valides des queries peuvent être composées
séparément avec un masque `[batch,queries,1]`.

## Composer et placer le masque

`unsqueeze` ajoute les axes nécessaires au broadcast : les clés deviennent
`[batch,1,keys]`, la causalité `[1,queries,keys]`. Ajoutez un axe de tête pour
obtenir `[batch,1,queries,keys]` si vos scores ont plusieurs têtes. La
[logique booléenne et where](Masks.md) conservent la convention `true`.

Pour un calcul GPU, placez explicitement chaque masque avec `mask.to(device)`
avant sa composition avec les scores résidents. Aucun readback ou cast d’indices
n’est nécessaire. Les contraintes habituelles de formes et de device de Tensor
s’appliquent aux opérations GPU. Les deux factories restent CPU et ne choisissent
aucun device pour l’application.

Les masques sont détachés. `where` transmet le gradient aux branches sélectionnées
et réduit leurs axes broadcast ; une ligne entièrement `false` sélectionne
entièrement la seconde branche. Ces fonctions construisent la visibilité et ne
constituent pas un softmax masqué.
