# Reprendre un entraînement complet

`Neural.TrainingCheckpoint` capture et restaure ensemble le modèle enregistré,
l’optimiseur et le curseur de lots. `save` retourne `Result<void,str>` ; `load`
retourne `Result<Neural.TrainingProgress,str>`. Les surcharges acceptent `SGD`,
`Adam` ou `AdamW`. Recréez d’abord le même modèle et la même source, puis chargez
sur CPU avant tout déplacement explicite vers le GPU.

Cet exemple effectue un pas, sauvegarde, reconstruit avec une autre seed, puis
compare le prochain pas au parcours continu. Il affiche
`Exact continuation after step 2` et supprime son fichier temporaire.

```sx
use STD.File
use STD.FileSystem
use Tensor
use Tensor.Data
use Tensor.Neural
use Tensor.Optim

func model(seed:int) Neural.Sequential {
    return Neural.Sequential([
        Neural.Dense("dense",2,2,seed:seed),
        Neural.BatchNorm("norm",2),Neural.Dropout(0.25,seed:seed)
    ])
}
func train_step(model:Neural.Layer,optimizer:Optim.AdamW,cursor:Data.BatchCursor,progress:Neural.TrainingProgress) Neural.TrainingProgress {
    var active=model;var optim=optimizer;var reading=cursor
    if let batch=try reading.next() else error {panic(error)} {
        if let schedule=progress.schedule {optim.set_learning_rate(schedule.value_at(progress.step))}
        let error=active.forward(batch.input).subtract(batch.target)
        error.multiply(error).mean().backward();optim.step();optim.zero_grad()
        return Neural.TrainingProgress(step:progress.step+1,schedule:progress.schedule)
    }
    panic("example corpus exhausted")
}
func main() {
    var inputs:float[]=[1.0,2.0,2.0,4.0,3.0,6.0,4.0,8.0]
    var source=Data.TensorSource(Tensor(inputs,[4,2]),Tensor.ones([4,2]))
    var original=model(42);var cursor=Data.BatchCursor(source,2,shuffle:true,seed:42)
    var optimizer=Optim.AdamW(original.parameters(),learning_rate:0.01)
    var progress=Neural.TrainingProgress(step:0,schedule:Optim.LearningRateSchedule.step(0.01,step_size:2,gamma:0.5))
    progress=train_step(original,optimizer,cursor,progress)
    let path="tensor-training-example.sxtc"
    try Neural.TrainingCheckpoint.save(original,optimizer,cursor,progress,path) else error {panic(error)}
    var restored=model(93);var restored_cursor=Data.BatchCursor(source,2,shuffle:true,seed:42)
    var restored_optimizer=Optim.AdamW(restored.parameters())
    var restored_progress=try Neural.TrainingCheckpoint.load(restored,restored_optimizer,restored_cursor,path) else error {panic(error)}
    progress=train_step(original,optimizer,cursor,progress)
    restored_progress=train_step(restored,restored_optimizer,restored_cursor,restored_progress)
    var expected=original.parameters();var actual=restored.parameters();var index=0
    while index<expected.count() {
        let left=expected[index].value().detach().values();let right=actual[index].value().detach().values();var element=0
        while element<left.count() {assert(left[element]==right[element]);element++}
        index++
    }
    assert(progress.step==restored_progress.step && cursor.state()==restored_cursor.state())
    try FileSystem.remove_file(path) else error {panic("cannot remove example checkpoint")}
    print("Exact continuation after step $(restored_progress.step)")
}
```

## Ce qui reprend

Le format d’entraînement version 1 conserve exactement les bits des tenseurs :
float32, entiers signés/non signés 8/16/32/64 et booléens. Les strides ne font pas
partie du fichier : les valeurs sont enregistrées dans l’ordre logique puis
restaurées en tenseurs CPU contigus. Subnormaux, signes de zéro et états numériques
non finis sont préservés ; les hyperparamètres hôte doivent rester valides et finis.

Les paramètres conservent nom, valeur et gel ; les buffers conservent nom, valeur
et politique de placement. Les couches enregistrées conservent leur mode local,
y compris des enfants en eval dans un parent en train. Dropout retrouve sa seed
et sa prochaine itération. BatchNorm retrouve ses paramètres et ses trois buffers,
dont le compteur int64 CPU. Les moments, compteurs et hyperparamètres d’optimiseur,
l’étape et le calendrier facultatif, ainsi que la seed/époque/position du curseur
sont restaurés ensemble. L’optimiseur doit posséder exactement les identités des
paramètres du modèle, y compris ceux qui sont gelés ou sans moment initialisé.

`TrainingProgress(step:...,schedule:...)` décrit la prochaine étape. Définissez
le taux avec `schedule.value_at(progress.step)` avant `optimizer.step()`, puis
incrémentez l’étape après le pas réussi. Le taux courant sauvegardé reste celui du
dernier pas effectué ; le calendrier chargé produit le prochain taux. Le curseur
est capturé à sa position courante : sauvegardez entre deux lots terminés, après
`zero_grad()`, afin de retrouver le prochain lot.

Un curseur peut activer le [préchargement CPU](BatchPrefetch.md) : le checkpoint
conserve sa progression consommée, sans sérialiser les lots préparés. La même
configuration peut être reconstruite avec ou sans préchargement.

## Conditions et erreurs récupérables

Le modèle cible possède les mêmes noms et ordre enregistrés, formes, dtypes,
politiques de buffer, enfants nommés et partage par identité. La configuration
des couches intégrées est comparée ; les seeds de construction des poids peuvent
différer, puisque les valeurs sauvegardées les remplacent. Les paramètres et les
buffers du modèle cible sont CPU et tous ses gradients sont absents. Le curseur
est inactif, sa source annonce toujours le même count et sa configuration
count/batch_size/shuffle/seed/drop_last correspond au fichier.

Le chargement prépare et valide le document entier, puis toutes les cibles,
avant application. Un `failure` laisse paramètres, buffers, modes, gel, RNG,
optimiseur et curseur inchangés. Les erreurs de fichier, version, encodage,
cardinalité, configuration, noms, placements ou moments sont récupérables.
`maximum_bytes` est positif et vaut 16 MiB par défaut ; un fichier plus grand est
refusé. Utilisez modèle, optimiseur et curseur exclusivement depuis le thread de
la boucle pendant save/load. Les effets arbitraires des callbacks et hooks du
code utilisateur, ainsi que les défaillances fatales d’allocation, ne sont pas
annulés par la restauration.

Le fichier ne reconstruit aucun code, graphe autograd, device, source ni callback.
Les signatures internes servent à vérifier les liens enregistrés et configurations,
pas à reconstruire l’architecture. Recréez le même code pour les couches extérieures ;
leur état mutable nécessaire à la continuation doit être dans leurs paramètres
et buffers enregistrés. Le contenu de la source recréée relève de l’appelant :
une configuration de curseur identique ne prouve pas que le dataset est identique.
Les gradients en attente et les états extérieurs non enregistrés ne sont pas sauvés.

## Reprendre sur GPU

La sauvegarde GPU effectue des readbacks explicites. Le chargement reste CPU,
sans transfert caché ni choix automatique de device. Après un chargement réussi :

```sx
var moments=optimizer.state().to(device)
model.to(device)
try optimizer.restore(moments) else error {panic(error)}
```

`SGDState.to(device)`, `AdamState.to(device)` et `AdamWState.to(device)` copient
les tableaux de métadonnées et transfèrent explicitement les moments présents.
Les buffers de politique CPU restent CPU. Ne déplacez pas seulement le modèle :
les moments déjà initialisés doivent être sur son device avant le prochain pas.
L’égalité de reprise est qualifiée séparément sur CPU et Metal avec le même
code et placement ; elle n’est pas une garantie d’identité entre deux backends.

Dropout dispose de 2^32 itérations de train. Le checkpoint conserve aussi la
position épuisée 2^32, qui reste utilisable en eval ; une nouvelle passe train
est refusée avant calcul. Ce compteur ne boucle pas silencieusement.

## Ancien checkpoint de modèle

`Neural.Checkpoint` et son schéma 1 restent compatibles et distincts. Ils stockent
uniquement les paramètres float32, avec leur représentation texte historique ;
ils ne constituent pas une reprise complète et ne promettent pas la fidélité des
bits. Utilisez ce guide pour la continuation, et les [snapshots d’optimiseur](OptimizerState.md)
pour une restauration en mémoire contrôlée par votre propre application.
