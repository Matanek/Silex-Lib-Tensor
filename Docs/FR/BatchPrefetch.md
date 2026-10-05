# Préparer les lots CPU en avance

`Data.BatchCursor(source, batch_size, prefetch:2)` prépare au plus deux lots CPU
en avance avec `STD.Threading`. `prefetch:0`, le défaut, conserve le parcours
synchrone. `Data.TensorSource` est éligible lorsque ses deux Tensor sont CPU et
détachés. Une source extérieure choisit explicitement ce contrat en redéfinissant
`supports_prefetch()` ; le défaut de `Data.Source` retourne `false`.

Cet exemple reprend le deuxième lot depuis la position consommée, puis ferme le
curseur. Il affiche `Prefetched batches resume from consumed position 2`.

```sx
use Tensor
use Tensor.Data

nocopy class Samples:Data.Source {
    init():super(5) {}
    override func supports_prefetch() bool {return true}
    override func observation(index:int) Result<Data.Observation,str> {
        return Result<Data.Observation,str>.success(Data.Observation(input:Tensor.scalar(index),target:Tensor.scalar(index*10)))
    }
}
func main() {
    var source=Samples()
    var cursor=Data.BatchCursor(source,2,prefetch:2)
    assert(cursor.state().position==0)
    if let first=try cursor.next() else error {panic(error)} {assert(first.input.int64_values()[0]==0)}
    let saved=cursor.state()
    assert(saved.position==2)
    if let second=try cursor.next() else error {panic(error)} {assert(second.input.int64_values()[0]==2)}
    cursor.restore(saved)
    if let repeated=try cursor.next() else error {panic(error)} {assert(repeated.input.int64_values()[0]==2)}
    cursor.close()
    print("Prefetched batches resume from consumed position $(saved.position)")
}
```

## Même parcours, progression consommée

Le préchargement commence à la construction. Un seul worker possède les appels
à la source et à la collation pendant la préparation ; il s’arrête au premier
échec ou à la fin d’époque. Les lots et leurs erreurs sont restitués dans l’ordre
du sampler. Une erreur n’avance pas la progression consommée ; `next()` peut
relire ce même lot depuis sa première observation. Mélange, dernier lot incomplet,
`drop_last`, dtypes et valeurs suivent le [parcours synchrone](SupervisedBatches.md).

`state()` désigne uniquement les lots consommés, même si la préparation est plus
avancée. `restore()` et `set_epoch()` attendent le travail accepté, retirent les
lots préparés et relancent depuis la nouvelle position. Une préparation abandonnée
peut avoir effectué des lectures : les effets externes des sources et callbacks
ne sont pas annulés. Ils ne doivent pas dépendre de l’ordre de consommation pour
produire leurs observations.

La borne est `prefetch` lots préparés, plus le lot actuellement retourné ; la
collation utilise aussi les observations transitoires d’un seul lot. Le corpus
source, les indices de mélange et les lots retenus par l’application restent à
sa charge. Cette borne porte sur le nombre de lots, pas sur leur taille en octets.

## Possession et erreurs

Utilisez le curseur depuis un seul thread. Pendant son préchargement, n’utilisez
ni ne modifiez sa source, les captures de la collation ou leurs dépendances depuis
un autre code. Sources et callbacks doivent rester CPU, produire des Tensor
détachés et ne pas créer/manipuler de graphes de gradients ou de devices GPU.
Ils ne doivent pas rappeler le curseur. Le package vérifie les observations avant
la collation et les sorties avant leur consommation ; un callback arbitraire n’est
pas exécuté dans un environnement qui interdit ses effets système.

Une source extérieure retournant `true` promet ce contrat pour `count()` et
`observation()`. La collation personnalisée partage ce contrat dès que le caller
active `prefetch`. Les sources GPU ou suivies de TensorSource sont refusées avant
le démarrage du worker : aucun détachement, transfert ni fallback implicite.
Les erreurs de source, collation et validation restent des `Result.failure`.
Une valeur négative de `prefetch` ou une source non éligible est un diagnostic
fatal de construction.

`close()` attend les travaux acceptés, libère les lots préparés et rejoint les
workers. Il est idempotent et terminal : `next()` retourne ensuite une erreur,
`restore()`/`set_epoch()` sont refusés et un chargement de checkpoint échoue sans
modifier sa cible. La destruction du curseur assure également cet arrêt ; les
références conservées par l’application prolongent sa durée de vie. `close()`
permet de réutiliser sa source immédiatement après un arrêt anticipé. Une lecture
source bloquée doit finir avant que la fermeture puisse rendre la main.

## Reprendre l’entraînement CPU ou GPU

Le [checkpoint complet](TrainingCheckpoint.md) enregistre le snapshot consommé,
sans sérialiser les lots préparés, les workers ou le choix `prefetch`. Il attend
la préparation avant de vérifier le count stable de la source. Un chargement
rejeté conserve les prochains lots préparés et toute la progression ; une
restauration réussie les reconstruit depuis le snapshot sauvegardé. Le caller
peut donc reconstruire un curseur synchrone ou préchargé avec la même configuration.

Les lots restent CPU. Pour entraîner sur GPU, déplacez explicitement `batch.input`
et `batch.target` sur le device dans la boucle d’entraînement, ainsi que le modèle
et ses moments selon le guide. Les tests comparent état complet continu/repris
avec BatchNorm, Dropout, AdamW, calendrier et changements d’époque, séparément
sur CPU et Metal. Aucun gain de débit n’est promis par cette qualification.
