# Normaliser un lot puis utiliser ses statistiques en évaluation

`Neural.BatchNorm` normalise chaque feature d’un lot dense `[N,F]`, ou chaque
canal d’une entrée NCHW `[N,C,H,W]`. En entraînement, il calcule les statistiques
du lot et actualise ses buffers nommés. En évaluation, il utilise ces buffers
sans les modifier. Les valeurs et gradients float32 sont pris en charge sur
CPU et GPU.

Cet exemple calcule les gradients affines sur un lot, les libère,
puis évalue avec les statistiques courantes. Les observations GPU utilisent
explicitement `detach().cpu()`.

```sx
use STD.Math
use Tensor
use Tensor.Neural
use GFX.GPU

func main() {
    var model=Neural.BatchNorm("normalization",2,momentum:0.5)
    var values:float[]=[1.0,2.0,3.0,6.0];var input=Tensor(values,[2,2])
    if GPU.Device.is_supported() {
        var device=GPU.Device();model.to(device);input=input.to(device)
    }
    let output=model.forward(input)
    output.multiply(output).sum().backward();model.zero_grad()
    let training_values=output.detach().cpu().values()
    assert(Math.abs(training_values[0]+0.999995)<0.00001)
    assert(model.buffers()[2].value().is_cpu() && model.buffers()[2].value().int64_item()==1)
    model.eval();let evaluated=model.forward(input).detach().cpu().values()
    assert(Math.abs(evaluated[0])<0.00001 && Math.abs(evaluated[3]-1.885616)<0.00001)
    assert(model.buffers()[2].value().int64_item()==1)
}
```

## Choisir les conventions du lot

Le constructeur est
`BatchNorm(name,features,epsilon:0.00001,momentum:0.1,affine:true)`.
Le nom doit être non vide, `features` positif et égal à la dimension de l’axe 1,
`epsilon` fini et strictement positif, `momentum` fini dans `[0,1]`.

Le mode initial est train. Pour chaque feature, les axes réduits sont `[0]`
en dense et `[0,2,3]` en NCHW. Avec `m` observations par feature, la sortie est
`(input - moyenne) / sqrt(variance_population + epsilon)`, puis `weight * sortie
+ bias` lorsque `affine` est vrai. La normalisation emploie la variance de
population ; la mise à jour de la variance courante emploie la variance non
biaisée `variance_population * m / (m - 1)`. Train exige donc `m > 1`.

Les mises à jour sont `(1-momentum)*ancien + momentum*lot`. `momentum:0.0`
conserve les statistiques pour un lot fini, mais compte le forward ;
`momentum:1.0` remplace les statistiques par celles du dernier lot. Une valeur
non finie dans les données se propage suivant les opérations Tensor ; le
momentum zéro ne constitue pas un filtre de NaN.

## Conserver l’état nommé

Pour le nom `normalization`, `parameters()` retourne, dans cet ordre,
`normalization.weight` initialisé à un et `normalization.bias` initialisé à zéro.
`affine:false` ne crée aucun paramètre.

`buffers()` retourne, dans cet ordre :

| Nom | Valeur initiale | Placement |
| --- | --- | --- |
| `normalization.running_mean` | float32 `[features]`, zéros | Suit le modèle |
| `normalization.running_variance` | float32 `[features]`, uns | Suit le modèle |
| `normalization.batches_tracked` | int64 scalaire, zéro | CPU explicite |

Les statistiques sont détachées : elles ne conservent aucun graphe autograd.
Le compteur reste exact sur CPU pendant `to(device)` et `cpu()`. Il avance une
fois par forward train réussi, et un débordement int64 est refusé avant toute
mise à jour. La collecte, le partage et les modes suivent le
[registre des couches](Registration.md). Geler les paramètres affines suspend
leurs nouveaux gradients ; cela conserve les gradients de l’entrée et les
mises à jour des statistiques en train.

En eval, les statistiques initiales valent moyenne zéro et variance un.
Le compteur et les statistiques ne changent pas ; les lots vides ou unitaires
sont acceptés. L’entrée reste différentiable et les paramètres affines aussi,
lorsqu’ils sont entraînables. Appelez `eval()` pour l’inférence, indépendamment
du [gel des paramètres](State.md).

Le checkpoint de modèle de schéma 1 conserve uniquement les paramètres : il
ne sauvegarde pas les buffers BatchNorm. Pour conserver ces statistiques avec
l’API courante, capturez leurs valeurs détachées et restaurez-les explicitement
avec `Buffer.replace` sur le même dtype, la même forme et le même placement.

## Garder les calculs sur leur placement

Déplacez explicitement le modèle et l’entrée sur le même CPU ou device GPU.
Le forward prévalide dtype, rang, features, placements de toutes les données,
compteur et population avant de modifier les buffers. Les vues à strides
non négatifs sont acceptées. Le forward et le backward GPU n’effectuent aucun
upload ou download ; seule une observation explicite lit les données sur CPU.
Le compteur CPU est une métadonnée de contrôle, pas une conversion du calcul.
SyncBatchNorm et les entrées de rang autre que 2 ou 4 ne sont pas disponibles.
