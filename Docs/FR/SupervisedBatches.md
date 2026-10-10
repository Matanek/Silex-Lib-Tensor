# Parcourir des lots supervisés

Cette évolution exige la correction de `try` dans les méthodes modifiables du
compilateur Silex, décrite dans ses notes `Unreleased`.

`Data.TensorSource(inputs, targets)` lit les observations sur l’axe de tête de
deux Tensor. Ils ont le même nombre d’observations et partagent leur placement
CPU ou leur device GPU ; formes restantes et dtypes peuvent différer. Les deux
Tensor possèdent au moins un axe, y compris pour une source vide.

`Data.BatchCursor(source, batch_size, shuffle:false, seed:0, drop_last:false,
collate:Data.stack_batch)` possède son curseur d’indices. `next()` retourne
`Result<Data.Batch?,str>` : réussite avec `Batch(input, target)`, réussite avec
`null` en fin d’époque, ou erreur récupérable. Par défaut, chaque colonne
entrée/cible est empilée sur l’axe batch0. Les observations d’une colonne ont
même forme, dtype et placement ; entrée et cible partagent le placement.

```sx
use Tensor
use Tensor.Data

func same_values<T>(left:T[],right:T[]) bool {
    if left.count()!=right.count() {return false}
    var index=0
    while index<left.count() {
        if left[index]!=right[index] {return false}
        index++
    }
    return true
}

func main() {
    var inputs:float[] = [1.0,2.0,3.0,4.0,5.0,6.0]
    var labels:int32[] = [0 as int32,1 as int32,2 as int32]
    var source = Data.TensorSource(Tensor(inputs,[3,2]),Tensor.vector(labels))
    var cursor = Data.BatchCursor(source,2)
    if let batch = try cursor.next() else error { panic(error) } {
        assert(same_values(batch.input.shape(),[2,2]) && same_values(batch.target.shape(),[2]))
        var expected:float[] = [1.0,2.0,3.0,4.0]
        assert(same_values(batch.input.values(),expected))
        var expected_labels:int32[] = [0 as int32,1 as int32]
        assert(same_values(batch.target.int32_values(),expected_labels))
    } else { panic("missing first supervised batch") }
    let saved = cursor.state()
    var resumed = Data.BatchCursor(source,2)
    resumed.restore(saved)
    if let batch = try resumed.next() else error { panic(error) } {
        var expected:float[] = [5.0,6.0]
        assert(same_values(batch.input.values(),expected) && batch.target.int32_item() == 2 as int32)
    } else { panic("missing final supervised batch") }
    assert((try resumed.next() else error { panic(error) }) == null)
}
```

`state()`, `restore(state)` et `set_epoch(epoch)` reprennent les garanties du
[sampler](BatchSampling.md). Le dernier lot incomplet est conservé par défaut,
le mélange dépend de seed/époque/count et l’épuisement n’avance pas l’époque.
Le snapshot enregistre le prochain lot, sans conserver les Tensor ou une source.
Pour reprendre les mêmes données, reconstruire la même source et collation :
une configuration de sampler identique ne prouve pas que leur contenu est identique.

## Source extérieure et collation personnalisée

Dérivez de `Data.Source`, appelez son constructeur protégé `super(count)`, puis
redéfinissez `observation(index:int) Result<Data.Observation,str>`. Le count
annoncé reste stable pendant le parcours ; l’auteur garantit les bornes et la
stabilité des observations. `Observation` contient deux champs Tensor `input`
et `target`. Les références à Source et BatchCursor utilisent `var` en Silex.

Une fonction nommée `func(Data.Observation[]) Result<Data.Batch,str>` choisit la
collation. Elle reçoit les observations dans l’ordre des indices du sampler.
Cet exemple extérieur ajoute un padding explicite aux entrées de longueur variable :

```sx
use Tensor
use Tensor.Data

func same_values<T>(left:T[],right:T[]) bool {
    if left.count()!=right.count() {return false}
    var index=0
    while index<left.count() {
        if left[index]!=right[index] {return false}
        index++
    }
    return true
}

nocopy class Samples:Data.Source {
    init():super(3) {}
    override func observation(index:int) Result<Data.Observation,str> {
        if index < 0 || index >= self.count() {return Result<Data.Observation,str>.failure("sample index outside source")}
        var shape:int[]=[index+1]
        return Result<Data.Observation,str>.success(Data.Observation(input:Tensor.full(shape,(index+1) as float),target:Tensor.scalar(index)))
    }
}

func padded_batch(observations:Data.Observation[]) Result<Data.Batch,str> {
    var width=0
    for observation in observations {if observation.input.count()>width {width=observation.input.count()}}
    var padded:Data.Observation[]=[]
    for observation in observations {
        var padding_shape:int[]=[width-observation.input.count()]
        var parts:Tensor[]=[observation.input,Tensor.zeros(padding_shape)]
        padded.append(Data.Observation(input:Tensor.concatenate(parts,0),target:observation.target))
    }
    return (try Data.stack_batch(padded))
}

func main() {
    var source=Samples()
    var cursor=Data.BatchCursor(source,2,collate:padded_batch)
    if let batch=try cursor.next() else error {panic(error)} {
        var expected:float[]=[1.0,0.0,2.0,2.0]
        assert(same_values(batch.input.shape(),[2,2]) && same_values(batch.input.values(),expected))
    } else {panic("missing padded batch")}
}
```

La collation doit retourner une entrée et une cible ayant un axe de tête égal
au nombre d’observations reçues et le même placement/device. Elle peut choisir
le padding, les formes restantes et les dtypes. `Data.stack_batch` est disponible
pour terminer une transformation sans réimplémenter la collation par défaut.
Une liste vide passée directement à stack_batch retourne une erreur.

## Erreurs et progression

Les erreurs de source ou collation sont propagées avec leur message `str`.
Les incompatibilités de forme/dtype/placement de stack_batch, le changement du
count et les sorties de collation incohérentes sont récupérables. Le sampler
avance seulement après réussite du lot complet. Une erreur laisse son snapshot
inchangé ; l’appel suivant relit le même lot depuis sa première observation.
Les effets externes d’une lecture ou callback ne sont pas annulés.

Le curseur refuse une lecture récursive. Époque et restauration ne changent pas
pendant une lecture. Les configurations invalides de construction/restauration
et les changements de count lors de ces opérations restent des diagnostics
fatals, comme ceux du sampler. Le curseur s’utilise depuis un seul thread ; les
lectures synchrones et callbacks s’exécutent sur ce thread.

## Placement, gradients et durée de vie

La source Tensor conserve les valeurs fournies ; observation sélectionne une
vue. La collation produit des Tensor selon le contrat de `Tensor.stack`. Les
neuf dtypes numériques et bool sont conservés exactement sur CPU/GPU, avec
les extrêmes int64/uint64 qualifiés. Entrées et cibles ne sont pas converties
pour partager un dtype. Les gradients float32 des entrées restent attachés au
calcul source ; aucun détachement implicite n’est introduit.

Formes, dtypes et placements sont vérifiés sur les deux colonnes avant leurs
allocations de collation. Les lots GPU restent résidents : ni upload d’indices
ni download caché. Les callbacks personnalisés choisissent explicitement leurs
calculs et transferts. Lire un lot GPU nécessite `cpu()` comme tout Tensor.
Le curseur conserve la source et callback ; ses snapshots ne les conservent pas.
Le chemin synchrone ne crée aucun worker. Le [préchargement CPU facultatif](BatchPrefetch.md)
conserve la progression consommée et se ferme explicitement avec `close()`. Le
[checkpoint complet](TrainingCheckpoint.md) restaure l’entraînement et ce curseur.
