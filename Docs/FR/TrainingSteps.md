# Préparer une étape MSE avec AdamW

`Neural.TrainingStep.mse(model, optimizer, input, target)` prépare une étape
fermée pour un modèle et un AdamW déjà construits. `run(input, target)` exécute
zero-grad, forward, MSE scalaire, backward puis update, dans l’ordre eager.
Cette préparation est explicite : le calcul ordinaire reste disponible.

```sx
use Tensor
use Tensor.Neural
use Tensor.Optim
use GFX.GPU

func main() {
    assert(GPU.Device.is_supported())
    var device=GPU.Device()
    var layers:Neural.Layer[]=[Neural.Dense("hidden",2,4,101),Neural.Activation.tanh(),Neural.Dense("output",4,2,102)]
    var model=Neural.Sequential(layers)
    model.to(device)
    var optimizer=Optim.AdamW(model.parameters(),epsilon:0.01)
    let input=Tensor.ones([4,2]).to(device)
    let target=Tensor.zeros([4,2]).to(device)
    var step=try Neural.TrainingStep.mse(model,optimizer,input,target) else error {panic(error)}
    for iteration in 0...12 {
        var result=try step.run(input,target) else error {panic(error)}
        assert(result.output.shape()[0]==4 && result.loss.rank()==0)
    }
    let statistics=step.statistics()
    assert(statistics.captures==1 && statistics.executions==12 && statistics.rejections==0)
}
```

La préparation retourne `Result<TrainingStep,str>` ; la reprise retourne
`Result<TrainingStepResult,str>`. `output` et `loss` sont les observations avant
mise à jour, détachées du graph. Les conserver ne permet pas de rejouer leur
backward ; leurs valeurs restent intactes après les étapes suivantes.

## Domaine et état courant

La famille comprend les classes exactes Sequential, Dense, Activation
(relu/sigmoid/tanh) et LayerNorm, composées sans callback. AdamW doit être exact
et posséder les mêmes Parameter handles que le modèle. Classes inconnues,
sous-classes, buffers, dropout, autres pertes/optimiseurs et clipping sont des
frontières rejetées. Loader, checkpoint et scheduler restent hors préparation.
Le backward emploie les règles eager ; il n’est pas capturé.

Entrées et cibles GPU float32, non vides, rang ≤ 5, device et disposition fixes.
La cible a la forme exacte de la sortie, sans broadcasting implicite. Les
paramètres gardent leur forme/dtype/device ; leurs strides/offset peuvent changer.
Taux, betas, epsilon, poids, moments et compteurs courants sont relus, ainsi que
la trainabilité. Le passage decay zéro/non-zéro demande une nouvelle préparation.
Au moins une entrée, cible ou un paramètre doit être suivi pour le backward.
La provenance autograd des entrées/cibles doit être pure et non consommée ;
les callbacks personnalisés et les graphes de plus de 4 096 nœuds sont rejetés
avant préparation ou zero-grad. Les variables et vues pures restent admises.

## Rejet, nouvelle préparation et possession

`run` vérifie ses signatures et états avant d’effacer les gradients. Une erreur
ne modifie ni poids, moments, compteurs AdamW ni gradients. Les statistiques
comptent ce rejet et donnent sa raison. Une signature différente demande un
autre plan explicite ; l’application peut aussi reprendre sa boucle eager
habituelle avec des opérandes compatibles. Aucune recapture ou fallback caché.
Les règles ordinaires zero-grad avant changement de trainabilité/placement
et restauration d’état restent applicables.

Chaque TrainingStep possède un seul programme, avec captures/compilations,
exécutions/rejets, nombres d’opérations forward/loss et de paramètres visibles
par `statistics()`. Bornes : 256 couches, 128 paramètres, 256 opérations forward,
trois de perte et au plus 17 opérations AdamW froides par paramètre. Deux plans
restent indépendants. Le plan garde modèle/optimiseur vivants jusqu’à sa
libération ; ceux-ci ne le possèdent pas. Aucun exemple, sortie ou graph passé
n’est conservé par le programme, aucun buffer de sortie n’est écrasé.

## Choisir avec une mesure complète

Le regroupement réduit les soumissions ; il peut augmenter les allocations
intermédiaires et ne garantit pas un gain sur chaque taille. Mesurer l’étape
entière après échauffement, avec synchronisation, séparément de la préparation.
Les résultats qualifiés et contre-épreuves du candidat sont conservés dans
Silex-Benchmarks ; les performances hors macOS arm64/Metal restent ouvertes.
Les six familles sont qualifiées avec eager comme référence et refus explicites
hors du domaine préparé. La gate float16 reste indéterminée, sans API mixte ;
voir la [matrice et la précision](Reference.md). Voir aussi le [forward préparé](ForwardPlans.md)
et le [profilage](Profiling.md).

Sur Apple M3 Pro/Metal en Release, le 6 octobre 2026 : 16 paires et 8 contrôles A/A
par bras admettent 6,21 % sur MLP(2,2,4,2) et 20,66 % sur le modèle normalisé
(4,2,4,2), avec 16/16 paires favorables. MLP(16,16,32,8) régresse de 6,49 %,
0/16 favorable : cette charge doit conserver eager pour la performance.
Les contrôles A/A passent ; préparation et RSS sont séparés. 192 processus,
229 312 scalaires concordants GPU, oracle CPU 2e-5, mêmes sources/binaire.
Ces chiffres concernent l’étape entière, pas le temps GPU isolé. La campagne
recomputable se trouve sous `Silex-Benchmarks/Sources/TensorStableCompute/Baselines/`
dans `2026-10-06-macos-arm64-training-step`.
