# Choisir la décroissance et le taux

`Tensor.Optim.AdamW` sépare la décroissance des poids du calcul des moments.
`LearningRateSchedule` calcule un taux depuis une étape explicite. Cet exemple
vérifie d’abord la décroissance seule, puis reprend les mêmes pas AdamW avec un
calendrier cosinus et une valeur de modèle reconstruite.

```sx
use STD.Math
use Tensor
use Tensor.Neural.Parameter
use Tensor.Optim

func main() {
    var parameter=Parameter("weight",Tensor.scalar(2.0))
    var optimizer=Optim.AdamW([parameter],learning_rate:0.1,weight_decay:0.5)
    parameter.value().multiply(0.0).backward();optimizer.step();optimizer.zero_grad()
    assert(Math.abs(parameter.value().detach().item()-1.9)<0.000001)
    var saved=optimizer.state()
    var resumed_parameter=Parameter("weight",parameter.value().detach())
    var resumed=Optim.AdamW([resumed_parameter])
    try resumed.restore(saved) else error {panic(error)}
    let schedule=Optim.LearningRateSchedule.cosine(0.1,duration:4,minimum_rate:0.01)
    var step=1
    while step<6 {
        let rate=schedule.value_at(step)
        optimizer.set_learning_rate(rate);resumed.set_learning_rate(rate)
        parameter.value().multiply(3.0).backward();resumed_parameter.value().multiply(3.0).backward()
        optimizer.step();resumed.step();optimizer.zero_grad();resumed.zero_grad()
        assert(parameter.value().detach().item()==resumed_parameter.value().detach().item())
        step++
    }
}
```

## Choisir Adam ou AdamW

`Adam` conserve sa décroissance L2 couplée : le gradient utilisé par les moments
inclut `weight_decay * parameter`. `AdamW` calcule ses moments depuis le gradient
seul, multiplie l’ancienne valeur par `1 - learning_rate * weight_decay`, puis
soustrait le pas adaptatif corrigé du biais. Un gradient nul reste un gradient
actif : les moments avancent et AdamW applique la décroissance. Un gradient
absent ou un paramètre gelé ne fait avancer ni valeur ni moment ni compteur.

AdamW accepte les mêmes options et domaines qu’Adam ; ses valeurs par défaut
sont `learning_rate:0.001`, `beta1:0.9`, `beta2:0.999`, `epsilon:0.00000001` et
`weight_decay:0.01`. À décroissance nulle, les deux optimiseurs produisent les
mêmes pas. `zero_grad`, `clip_grad_norm`, `learning_rate` et `set_learning_rate`
ont les mêmes contrats. Les calculs GPU conservent valeurs et moments résidents.

`AdamW.state()` retourne un `AdamWState` distinct d’`AdamState`, avec les mêmes
champs et entrées `AdamMomentState`. `restore(AdamWState)` valide l’ensemble avant
mutation et `state.cpu()` lit explicitement les moments. Les règles de noms,
formes, gradients absents et placements sont celles du
[guide de reprise](OptimizerState.md). Ces snapshots ne sauvegardent pas les
valeurs du modèle ni la progression globale d’un entraînement.

## Calculer le taux du prochain pas

Un calendrier est une valeur sans compteur mutable. `value_at(step)` dépend
uniquement de sa configuration et de l’étape entière non négative fournie.
Des appels répétés ou dans un autre ordre ne changent aucun résultat.

| Construction | Taux à l’étape s |
| --- | --- |
| `constant(learning_rate)` | taux initial |
| `step(learning_rate, step_size, gamma)` | initial × gamma puissance division entière s/step_size |
| `exponential(learning_rate, gamma)` | initial × gamma puissance s |
| `cosine(learning_rate, duration, minimum_rate)` | minimum + (initial − minimum) × (1 + cos(π × s/duration))/2, puis minimum dès duration |

Taux initial, gamma et taux minimal doivent être finis strictement positifs.
`step_size` et `duration` sont des entiers strictement positifs. Le minimum
cosinus est fourni explicitement, doit être inférieur ou égal au taux initial,
et reste constant après la durée. Un `gamma > 1` produit un calendrier
croissant ; `gamma == 1` garde le taux constant.

Le résultat est un scalaire hôte float32. Un underflow vers zéro ou un overflow
vers une valeur non finie est diagnostiqué, avant que ce taux puisse être passé
à l’optimiseur. Les puissances entières ont un calcul borné même à la limite de
`int`. Le cosinus retourne exactement le taux initial à zéro et le minimum dès
la durée ; à l’intérieur, son arrondi suit la précision float32.

Pour la reprise, conservez l’étape **du prochain pas**. À l’étape s : calculez
`value_at(s)`, appliquez `set_learning_rate`, exécutez `optimizer.step`, puis
incrémentez s après réussite. Le calendrier n’incrémente rien automatiquement.
Un snapshot d’optimiseur capture le taux courant, mais il vous appartient de
conserver l’étape globale et de reconstruire la même configuration du calendrier.
