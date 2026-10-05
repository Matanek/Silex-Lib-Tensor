# Parcourir et reprendre des lots d’indices

`use Tensor.Data` donne accès à `Data.BatchSampler(count, batch_size,
shuffle:false, seed:0, drop_last:false)`. `next()` retourne une liste `int32[]?`
indépendante : `null` termine l’époque. Le dernier lot incomplet est conservé
par défaut ; `drop_last:true` l’ignore et place le curseur à `count`. Aucun lot
vide n’est rendu. Une source vide termine immédiatement.

`count` est compris entre 0 et 2147483648, pour des indices entre 0 et
2147483647. `batch_size` est strictement positif. Les indices suivent l’ordre
croissant, ou une permutation déterministe avec `shuffle:true`. Une même
configuration et une même époque reproduisent les mêmes lots. `seed` accepte
tout `int`, et l’époque un `int` non négatif. `set_epoch(epoch)` reconstruit
l’ordre et remet la position à zéro ; l’épuisement n’avance jamais l’époque.
Les alias d’un sampler partagent son curseur.

## Reprendre le prochain lot

`state()` retourne un `BatchSamplerState` immuable : `count`, `batch_size`,
`shuffle`, `seed`, `drop_last`, `epoch`, `position`. La position compte les
indices déjà parcourus, et non le numéro du lot. Le snapshot ne change pas
quand le sampler avance. `restore(state)` exige la même configuration, une
époque non négative et une position entre zéro et count, multiple de batch_size
ou égale à count. Tout est validé avant modification du curseur. La permutation
est reconstruite depuis les métadonnées ; elle n’est pas conservée dans l’état.

Cet exemple complet consomme une première liste, sauvegarde la position, puis
reconstruit les lots suivants dans un autre sampler :

```sx
use Tensor
use Tensor.Data

func main() {
    var values:float[] = [10.0,20.0,30.0,40.0,50.0]
    let source = Tensor.vector(values)
    var sampler = Data.BatchSampler(5,2)
    if let indices = sampler.next() {
        let batch = source.gather(0,Tensor.vector(indices))
        assert(batch.values()[0] == 10.0 && batch.values()[1] == 20.0)
    } else { panic("missing first batch") }
    let saved = sampler.state()
    var resumed = Data.BatchSampler(5,2)
    resumed.restore(saved)
    if let indices = resumed.next() {
        let batch = source.gather(0,Tensor.vector(indices))
        assert(batch.values()[0] == 30.0 && batch.values()[1] == 40.0)
    } else { panic("missing resumed batch") }
    if let final = resumed.next() { assert(final.count() == 1 && final[0] == 4 as int32) }
    else { panic("missing final partial batch") }
    assert(resumed.next() == null && resumed.state().position == 5)
}
```

## Coût et placement

Le parcours ordonné crée uniquement les indices du lot demandé ; il n’alloue
pas une liste de taille count. Le mélange reconstruit une permutation CPU
complète, avec mémoire et travail linéaires en count. La génération réutilise
`STD.Randomizer` avec une seed explicite et une permutation Fisher–Yates ;
les bits de seed et époque sont combinés par XOR non signé. Aucun RNG global
ou seed système n’intervient.

Le sampler manipule des métadonnées CPU : aucun Tensor, buffer GPU, upload,
download ou dispatch n’est créé. Le consommateur transforme les indices en
Tensor et choisit explicitement leur placement. Pour une source GPU, écrire
`Tensor.vector(indices).to(device)` avant un gather compatible avec sa forme.
Les listes retournées peuvent être modifiées sans affecter les lots suivants.

Ce snapshot décrit le seul parcours des indices. Il ne sauvegarde ni source,
collation, modèle, optimiseur ou RNG d’un dropout ; le checkpoint de paramètres
existant conserve son propre contrat. Aucune source supervisée ou collation
n’est fournie par cette API.
