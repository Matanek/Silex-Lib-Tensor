# Adapter une forme aux lots

Ajoutez ou retirez une dimension unitaire avec `unsqueeze` et `squeeze`.
`expand` expose plusieurs occurrences par une vue ; `repeat` matérialise des
tuiles. Ces opérations conservent le dtype, le placement et les gradients float32.

Ce programme prépare quatre occurrences d’un vecteur et vérifie leur gradient :

```sx
use Tensor
use Tensor.Autograd

func main() {
    var values:float[] = [1.0,2.0,3.0]
    var input = Autograd.Variable(Tensor.vector(values))
    let batch = input.value().unsqueeze(0).expand([2,3])
    let tiled = batch.repeat([2,1])
    assert(batch.strides()[0] == 0 && tiled.is_contiguous())
    var gradients = Autograd.grad(tiled.multiply(tiled).sum(),[input.value()])
    let actual = gradients[0].values()
    assert(actual[0] == 8.0 && actual[1] == 16.0 && actual[2] == 24.0)
    assert(input.gradient() == null)
    assert(input.value().unsqueeze(0).squeeze(0).rank() == 1)
}
```

Chaque valeur intervient quatre fois dans la somme des carrés, donc sa dérivée
vaut huit fois cette valeur. Les gradients retournés laissent le slot absent.

## Ajouter et retirer des unités

`unsqueeze(axis)` insère une dimension de taille un à un axe compris entre zéro
et le rang inclus. `squeeze(axis)` enlève cet axe seulement si sa taille vaut un ;
un autre axe ou une taille différente produit un diagnostic. `squeeze()` enlève
toutes les dimensions unitaires ; un scalaire reste scalaire et un axe vide reste
vide. Les axes négatifs sont refusés.

Ces vues conservent les valeurs, le stockage et l’offset, même après transpose
ou narrow. L’axe ajouté a un stride zéro ; les autres strides restent identiques.
Elles n’exigent aucune copie pour traiter une source non contiguë. Si une opération
ultérieure exige un Tensor contigu, demandez `contiguous()` explicitement.

## Étendre une vue

`expand(shape)` reçoit une forme non négative de rang au moins égal au rang
source. Les axes sont alignés à droite. Une dimension égale est conservée ; une
dimension source de taille un peut devenir une autre taille par stride zéro.
Les nouveaux axes à gauche sont implicitement unitaires. Les dimensions source
différentes de un ne peuvent pas changer. Aucun `-1` n’est accepté.

Une unité peut devenir un axe vide. Une dimension source zéro reste zéro et ne
peut pas devenir non vide. Cardinalité et taille logique en octets sont vérifiées
avant création de la vue. Les vues ne créent ni buffer ni commande GPU et ne
font aucun transfert. Une expansion répétée est réduite aux dimensions source
durant backward ou `Autograd.grad`, y compris un domaine vide qui rend des zéros.

## Répéter des tuiles

`repeat(repeats)` reçoit au moins un facteur non négatif par axe source. Des
facteurs supplémentaires ajoutent des axes unitaires à gauche. Chaque facteur
multiplie la dimension correspondante ; les valeurs se répètent par tuiles, en
ordre logique, plutôt que par duplication de chaque élément adjacent.

Le résultat est matérialisé et contigu, y compris avec tous les facteurs à un.
Un facteur zéro produit une forme vide dont les autres dimensions restent
multipliées. Un scalaire accepte une liste vide ou de nouveaux axes. Toutes les
formes, facteurs et tailles sont validés avant allocation. Les gradients somment
les contributions des occurrences. Les valeurs sources restent immuables ;
`Autograd.grad` préserve leurs slots.

Vues et répétition conservent exactement les neuf dtypes numériques et les
booléens sur CPU/GPU. Les gradients concernent float32 et gardent les limites
de rang des opérations GPU utilisées (jusqu’à cinq pour le calcul). Aucun
readback ni upload implicite : utilisez `cpu()` pour observer un résultat GPU,
et `detach().cpu()` pour une valeur encore suivie. La répétition a le coût d’une
matérialisation ; aucun gain de vitesse n’est promis.

Consultez le [guide des gradients](Gradients.md) et la [référence](Reference.md)
pour composer ces formes avec les calculs différentiables.
