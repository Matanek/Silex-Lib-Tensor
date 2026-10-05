# Construire des indices et une grille numérique

`Tensor.arange` crée une progression int32 à borne finale exclusive.
`Tensor.linspace` construit une grille float32 avec un nombre de points explicite.
Les deux créent des données CPU, que vous pouvez ensuite transférer par `to(device)`.

Ce programme sélectionne des observations et les pondère par une grille :

```sx
use Tensor
use Tensor.Autograd

func main() {
    var data:float[] = [0.0,10.0,20.0,30.0,40.0,50.0]
    var input = Autograd.Variable(Tensor.vector(data))
    let indices = Tensor.arange(1,6,2)
    let weights = Tensor.linspace(0.0,1.0,3)
    let selected = input.value().gather(0,indices)
    let loss = selected.multiply(weights).sum()
    assert(loss.detach().item() == 65.0)
    var gradients = Autograd.grad(loss,[input.value()])
    let actual = gradients[0].values()
    assert(actual[1] == 0.0 && actual[3] == 0.5 && actual[5] == 1.0)
    assert(input.gradient() == null)
}
```

Les indices sont [1,3,5], les valeurs choisies [10,30,50] et les poids [0,0.5,1].
La somme pondérée vaut65. Les indices et poids sont des données détachées ; le
gradient traverse la sélection et préserve le slot de la Variable.

## Construire une progression entière

`arange(stop)` commence à zéro avec un pas de un.
`arange(start, stop, step:1)` accepte des paramètres `int` et retourne un vecteur
int32. La borne stop est exclusive. Un pas positif avance vers une borne plus
grande ; un pas négatif recule vers une borne plus petite. Une direction qui ne
rejoint pas stop, ou deux bornes égales, rend un vecteur vide. Un pas nul est
refusé même pour une progression vide.

La cardinalité et le dernier élément sont calculés sans soustraction signée
susceptible de déborder aux extrêmes int64. Toutes les valeurs réellement rendues
doivent tenir dans int32, et cardinalité/tailles sont validées avant allocation.
Les bornes et pas peuvent dépasser int32 si les valeurs rendues restent valides ;
une progression vide ne contient aucune valeur à vérifier. Ces règles évitent
une conversion silencieuse et une dernière incrémentation inutile hors intervalle.

## Interpoler des valeurs flottantes

`linspace(start, stop, count)` exige des bornes finies et un count non négatif.
Ces bornes sont contrôlées même pour count0. Count0 rend un vecteur vide ; count1
rend start. Pour au moins deux points, start et stop sont conservés exactement,
y compris leurs zéros signés. Une grille peut monter, descendre ou rester constante.

L’interpolation évite le calcul overflow de stop−start pour des bornes de signes
opposés et garde chaque résultat entre les extrémités. Les arrondis float32
peuvent confondre des points voisins ; aucune précision supplémentaire n’est
promise. La taille physique est validée avant de remplir les valeurs.

Ces constructeurs ne suivent aucun paramètre Tensor : ce sont des données
réutilisables, pas une transformation différentiable de bornes scalaires.
Sur GPU, seul `to(device)` demandé par le consommateur effectue l’upload.
Aucun placement automatique ni téléchargement implicite n’est introduit.

Consultez les [formes de lots](Shapes.md) et le [guide des gradients](Gradients.md)
pour utiliser ces données dans un calcul différentiable.
