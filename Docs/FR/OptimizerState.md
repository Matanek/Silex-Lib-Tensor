# Reprendre un optimiseur

`SGD.state()` et `Adam.state()` capturent le taux courant, les hyperparamètres et
les moments nommés. Après reconstruction des valeurs du modèle, `restore(state)`
reprend le prochain pas avec les mêmes moments et compteurs. Cet exemple copie
explicitement la valeur du paramètre, puis vérifie que les deux entraînements
produisent exactement la même valeur au deuxième pas.

```sx
use Tensor
use Tensor.Neural.Parameter
use Tensor.Optim

func main() {
    var parameter=Parameter("weight",Tensor.scalar(2.0))
    var optimizer=Optim.Adam([parameter],learning_rate:0.1,beta1:0.5,beta2:0.5,epsilon:0.0001)
    parameter.value().multiply(3.0).backward();optimizer.step();optimizer.zero_grad()
    var saved=optimizer.state()
    var restored_parameter=Parameter("weight",parameter.value().detach())
    var restored=Optim.Adam([restored_parameter])
    try restored.restore(saved) else error {panic(error)}
    parameter.value().multiply(4.0).backward();optimizer.step()
    restored_parameter.value().multiply(4.0).backward();restored.step()
    assert(parameter.value().detach().item()==restored_parameter.value().detach().item())
    assert(optimizer.state().parameters[0].step_count==2 && restored.state().parameters[0].step_count==2)
}
```

## Capturer les moments

`SGDState` contient `learning_rate`, `momentum`, `weight_decay` et un tableau
`parameters` de `SGDMomentumState`. Chaque entrée contient `name`, `shape` et
`momentum:Tensor?`. Un buffer absent signifie que le momentum n’est pas initialisé ;
avec un coefficient nul, il reste absent même après des pas actifs.

`AdamState` contient `learning_rate`, `beta1`, `beta2`, `epsilon`, `weight_decay`
et un tableau `parameters` de `AdamMomentState`. Chaque entrée contient `name`,
`shape`, `step_count`, `first_moment:Tensor?` et `second_moment:Tensor?`. À zéro,
les deux moments sont absents ; après un pas actif, les deux sont présents.
Le compteur appartient au paramètre : un gradient absent ou un paramètre gelé
ne le fait pas avancer.

Les snapshots construisent de nouveaux tableaux d’entrées et de dimensions.
Leurs Tensor sont détachés et partagent les valeurs immuables du stockage.
Les pas suivants ne modifient donc pas les valeurs capturées. Les tableaux et
les champs Tensor restent éditables pour construire un état à restaurer.
Les snapshots ne contiennent pas les valeurs du modèle, ses buffers, gradients,
modes, états de gel, lots ou RNG. Le [checkpoint de paramètres](README.md)
reste un format distinct ; ces snapshots ne constituent pas un fichier de
reprise complète de l’entraînement.

## Restaurer sans modifier un état partiel

`restore(snapshot) Result<void,str>` exige le même ensemble de noms uniques et
les mêmes formes, y compris pour les paramètres sans moment. L’ordre du tableau
peut différer. Appelez `zero_grad()` avant restauration : tous les slots de
gradient doivent être absents, même pour un paramètre gelé et un ancien graphe.

L’ensemble des hyperparamètres, noms, formes, compteurs et placements est validé
avant modification de l’optimiseur. Une erreur récupérable laisse tous ses
moments, compteurs et hyperparamètres inchangés. La réussite remplace cet état,
sans modifier les valeurs, gradients, identités ou états de gel des paramètres.
Les moments fournis sont détachés, même s’ils proviennent d’un graphe suivi.

Chaque moment doit être float32, de la forme du paramètre et sur le même CPU ou
device GPU. Le contenu numérique n’est pas inspecté : les NaN/infinis des moments
sont conservés et aucun readback de validation n’est caché. Les hyperparamètres,
qui sont des scalaires hôte, doivent être finis. Taux et epsilon sont strictement
positifs, les beta Adam appartiennent à `[0,1)`, momentum et weight_decay sont
positifs ou nuls. Un Adam refuse un compteur négatif, des moments au compteur
zéro ou un seul moment après zéro ; SGD refuse un buffer de momentum lorsque
le coefficient sauvegardé est nul.

## Choisir le placement et le taux courant

`state()` et `restore()` conservent le placement résident des moments sans
transfert. `snapshot.cpu()` crée explicitement une copie des moments sur CPU ;
les tableaux et dimensions de cette copie sont indépendants. Pour restaurer un
snapshot CPU dans un modèle GPU, transférez explicitement les Tensor de ses
entrées vers le device du modèle avant `restore`. Il n’existe pas de migration
implicite des états lors du déplacement d’un paramètre.

`learning_rate()` lit le taux courant ; `set_learning_rate(value)` exige un
scalaire fini strictement positif et conserve tous les moments et compteurs.
Appliquez le nouveau taux avant le pas auquel il doit servir. Un snapshot
capture ce taux ; sa restauration remplace aussi les autres hyperparamètres.

Adam accepte les compteurs non négatifs jusqu’à la limite de `int`. Sa correction
de biais calcule les puissances par carrés en au plus 63 itérations, même après
une longue reprise. Le regroupement des multiplications peut modifier l’arrondi
flottant par rapport à une multiplication linéaire. Une tentative d’avancer un
compteur maximal avec un gradient actif échoue avant toute mise à jour des
paramètres ; sans gradient actif, ce compteur reste inchangé.
