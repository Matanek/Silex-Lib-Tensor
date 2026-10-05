# Préparer un forward GPU répété

`model.prepare_forward(example)` prépare le forward du modèle pour une entrée
GPU `float32` de disposition fixe. Appelez ensuite son `forward` habituel : il
utilise le plan compatible, ou le calcul eager avec une raison observable.
Le modèle possède sa préparation ; l’application ne construit aucun graphe.

## Préparer le modèle existant

```sx
use Tensor
use Tensor.Neural
use GFX.GPU

func main() {
    assert(GPU.Device.is_supported())
    var device = GPU.Device()
    var layers:Neural.Layer[] = [
        Neural.Dense("hidden", 2, 4, 101),
        Neural.Activation.tanh(),
        Neural.Dense("output", 4, 2, 102)
    ]
    var model = Neural.Sequential(layers)
    model.to(device)
    let input = Tensor.ones([16, 2]).to(device)
    match model.prepare_forward(input) {
        failure(reason) => { panic(reason) }
        success => {}
    }
    let output = model.forward(input)
    assert(output.shape()[0] == 16 && output.shape()[1] == 2)
    let statistics = model.forward_plan_statistics()
    assert(statistics.captures == 1 && statistics.executions == 1)
    model.clear_forward_plan()
    assert(model.forward_plan_statistics().cached_plans == 0)
}
```

Les trois méthodes appartiennent à `Neural.Layer` et sont héritées par les
couches concrètes. `prepare_forward` retourne `Result<void,str>` ; l’échec
explique pourquoi ce modèle ou cet exemple n’est pas pris en charge.

## Famille prise en charge

La première famille comprend les classes concrètes exactes `Sequential`,
`Dense`, `Activation` (`relu`, `sigmoid`, `tanh`) et `LayerNorm`. Un Sequential
peut les composer et répéter la même couche : chaque occurrence est exécutée,
avec les mêmes handles de paramètres partagés.

Les autres classes et leurs sous-classes sont rejetées avant l’appel à leurs
hooks `children`, `parameters` ou `forward`. La préparation n’exécute aucun
forward applicatif, callback, aléatoire, transfert ou mise à jour de poids.
Les hooks éventuels de construction du modèle sont antérieurs à cette préparation.

L’exemple doit être GPU `float32`, non vide, de rang au plus 5, et respecter les
formes attendues par les couches. La préparation est bornée à 256 couches
uniques, 128 paramètres uniques et 256 opérations. Chaque couche préparée
possède au plus un plan ; aucun storage d’exemple ou de sortie n’est capturé.

## Réutilisation, état et fallback

Le plan vérifie forme, strides, offset, dtype et device de l’entrée, mode du
modèle, puis paramètres courants avant tout encodage. Une signature différente
utilise eager et renseigne `last_fallback`. Ce calcul conserve les règles
habituelles de compatibilité des opérandes : déplacer aussi le modèle avant
un forward CPU ou sur un autre device.

Une nouvelle préparation réussie remplace l’ancienne. Une préparation rejetée
conserve le plan précédent, paramètres, buffers et gradients. Répéter une
préparation encore compatible ne recapture pas. Aucune recapture automatique
n’a lieu lors d’un fallback.

`Parameter.replace` compatible est lu à la reprise suivante, même si les strides
ou l’offset changent. Les nouveaux layouts des paramètres sont pris en compte
sans recapture. `freeze`, `unfreeze` et la provenance de l’entrée déterminent
les gradients du forward courant. Le backward utilise les règles autograd eager ;
le plan de forward ne prépare pas une étape d’entraînement entière.

`clear_forward_plan()` libère le cache de cette couche sans changer son état
numérique. Les sorties conservées restent indépendantes et utilisables.

## Observer et mesurer

`ForwardPlanStatistics` est un instantané de compteurs cumulés :

| Champ | Observation |
|---|---|
| `preparations`, `rejections` | Appels explicites et préparations refusées |
| `captures`, `compilations` | Nouvelles représentations préparées ; compilations du plan, distinctes des pipelines GPU |
| `executions`, `fallbacks` | Reprises du plan et retours à eager lorsqu’un plan existe |
| `cached_plans`, `prepared_operations` | Présence et taille du cache courant |
| `cache_capacity_plans` | Capacité fixe : 1 |
| `last_fallback` | Dernière raison, effacée par une reprise compatible ou une nouvelle capture |

Un modèle sans plan n’incrémente pas `fallbacks`. Les compteurs cumulés ne sont
pas remis à zéro par l’effacement. Utilisez [Profile](Profiling.md) pour observer
les compilations réelles de pipelines, transferts, passes et soumissions.

Le plan conserve les opérations et leur ordre eager, puis regroupe leur soumission.
Le MLP de l’exemple garde cinq passes et utilise une soumission/fence commune ;
LayerNorm garde huit passes, dont six de normalisation. Les valeurs et tous les
gradients sont qualifiés contre eager GPU et CPU.

Sur trois charges Dense–tanh–Dense mesurées en Release sur Metal macOS ARM64,
les médianes chaudes diminuent de 40,15 %, 33,42 % et 19,58 %, avec 16 paires
favorables sur 16 et contrôles A/A valides. La préparation initiale, pipelines
compris, est observée séparément autour de 1,09 ms sur ce banc. Ces mesures
ne constituent ni un gain de démarrage ni une promesse sur tous les modèles
ou toutes les architectures. Le surcoût du profilage est indéterminé dans cette
série. Les sources, contrôles et données brutes vivent dans le banc
`TensorStableCompute/PreparedForward.sx` et sa campagne scellée.

[Revenir à Tensor](README.md) · [État du modèle](ModelState.md) ·
[Gradients fonctionnels](Gradients.md)
