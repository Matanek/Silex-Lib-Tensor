# Sélectionner les meilleurs éléments

`Tensor.topk(k, axis, largest:true)` sélectionne les k meilleures valeurs sur
un axe explicite. Le résultat `TopKResult` contient `values` et `indices` :
les valeurs gardent leur dtype, les indices sont int32 et détachés. Utilisez
`largest:false` pour choisir les plus petites valeurs.

Ce programme retient deux observations maximales et calcule leur gradient :

```sx
use Tensor
use Tensor.Autograd

func main() {
    var observations:float[] = [2.0,7.0,7.0,1.0]
    var input = Autograd.Variable(Tensor.vector(observations))
    let selected = input.value().topk(2,axis:0)
    let indices = selected.indices.int32_values()
    assert(indices[0] == 1 && indices[1] == 2)
    let loss = selected.values.sum()
    assert(loss.detach().item() == 14.0)
    var gradients = Autograd.grad(loss,[input.value()])
    let actual = gradients[0].values()
    assert(actual[0] == 0.0 && actual[1] == 1.0 && actual[2] == 1.0 && actual[3] == 0.0)
    assert(input.gradient() == null)
}
```

Les indices sont [1,2], les valeurs [7,7] et la somme vaut 14. Les deux
observations retenues reçoivent un gradient de 1 ; les autres reçoivent zéro.
`Autograd.grad` préserve le slot de la Variable. Pour annoter un résultat,
importez sa déclaration par `use Tensor.TopKResult`.

## Choisir l’axe et le nombre d’éléments

L’axe est obligatoire, non négatif et inférieur au rang. La shape de sortie
conserve toutes les dimensions et remplace celle de l’axe par k. Pour une entrée
[2,3], topk(2,axis:1) rend des valeurs et indices [2,2]. Chaque indice est une
coordonnée sur cet axe de l’entrée, pas une position dans la sortie.

k doit être compris entre zéro et la taille de l’axe. k égal à zéro rend une sortie vide,
y compris lorsque cet axe a une taille nulle. Une dimension extérieure vide
reste vide. k égal à la taille de l’axe sélectionne toutes les valeurs dans
l’ordre demandé. Les vues non contiguës, offsets et dimensions étendues sont
lus dans leur ordre logique ; leur stockage n’est pas modifié.

## Prévoir l’ordre et les gradients

Les valeurs sont rangées du meilleur au moins bon : décroissantes pour largest,
croissantes sinon. Des valeurs égales conservent l’ordre des indices d’entrée,
y compris +0 et −0, dont les signes originaux restent dans les valeurs rendues.
Les NaN viennent d’abord dans les deux directions, par indice croissant, puis
les nombres et infinis dans l’ordre demandé. Cette convention suit celle des
[extrema indexés](IndexedReductions.md).

Les indices ne se différencient pas. Les valeurs float32 se différencient par
sélection, y compris sur GPU : chaque gradient retourne à la position retenue ;
les positions non retenues reçoivent zéro. Une vue étendue accumule les gradients
vers sa source. La sélection est discontinue lorsque l’ordre change ; utilisez
le [gradcheck](Gradients.md) hors égalités ou changements de rang.

## Choisir le placement et les limites

Les neuf dtypes numériques CPU sont acceptés, avec comparaison entière exacte
jusqu’à int64/uint64. Les booléens sont refusés. Sur GPU, l’entrée doit être
float32 ; valeurs et indices restent sur le même device, sans upload ou readback
implicite. Une observation requiert explicitement `cpu()`.

Le dernier indice possible doit tenir dans int32, soit un axe d’au plus 2^31
éléments. Shapes et tailles des valeurs/indices sont validées avant allocation.
Le GPU conserve ses limites de rang 5 et de métadonnées uint32. La sélection
n’expose aucun tri général ; aucun avantage de vitesse n’est promis pour de
grands domaines ou grands k. Le [profil GPU](Profiling.md) permet de mesurer
votre parcours complet.
