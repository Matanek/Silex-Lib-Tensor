# Observer les coûts GPU

`Tensor.Profile` observe le travail Tensor émis sur un device pendant une
période. Il permet de distinguer calcul résident, transferts et pression des
buffers avant d’optimiser un parcours d’entraînement.

```sx
use GFX.GPU
use Tensor
use Tensor.Profile

var device = GPU.Device()
var values:float[] = [1.0, 2.0, 3.0]
let input = Tensor.vector(values).to(device)
var profile = Profile(device)
let output = input.multiply(2.0)
let resident = profile.snapshot()
assert(resident.uploads == 0 && resident.downloads == 0)
assert(resident.dispatches == 1 && resident.active_bytes == 24)

profile.reset()
let observed = output.cpu().values()
let observation = profile.snapshot()
assert(observation.downloads == 1 && observation.downloaded_bytes == 12)
assert(resident.downloads == 0)
```

`use Tensor.Profile` sélectionne le type de profil ; `use Tensor.ProfileStatistics`
importe le type d’instantané lorsqu’une annotation est nécessaire. `Profile` est
une classe non clonable. Sa période commence à la construction et finit à sa
destruction. `reset()` démarre une nouvelle période sur le même device.
Plusieurs profils peuvent observer des périodes chevauchantes ; réinitialiser
ou détruire l’un ne modifie aucun autre profil.

`snapshot()` renvoie une valeur immuable composée uniquement de compteurs
entiers. Les opérations suivantes et les resets n’altèrent pas les instantanés
précédents. Lire, réinitialiser ou détruire un profil ne télécharge aucun élément,
n’ajoute aucune fence et n’attend aucune commande GPU. Pour mesurer une durée
murale incluant l’achèvement GPU, employer explicitement `device.wait_until_idle()`
avant et après le parcours chronométré. Un compteur de soumissions ne mesure
pas la durée d’exécution GPU.

## Compteurs

Tous les événements comptent le travail émis pendant la période. Les buffers et
bytes actifs sont, eux, l’état courant du device, y compris les buffers Tensor
créés avant l’ouverture de la période.

| Champs de `ProfileStatistics` | Signification |
| --- | --- |
| `uploads`, `downloads` | Transferts Tensor CPU→GPU et GPU→CPU. |
| `uploaded_bytes`, `downloaded_bytes` | Octets logiques de ces transferts, selon le dtype. |
| `buffer_allocations`, `buffer_releases` | Créations et libérations de buffers de stockage Tensor pendant la période. Les libérations peuvent concerner des buffers antérieurs. |
| `live_buffers`, `peak_live_buffers` | Buffers Tensor actuellement vivants et maximum depuis le début de la période. |
| `active_bytes`, `peak_bytes` | Somme des tailles demandées de ces buffers et maximum sur la période. |
| `upload_staging_allocations`, `download_staging_allocations` | Créations des buffers de transfert Tensor correspondants. |
| `fence_creations`, `submissions` | Fences créées et commandes soumises par Tensor. |
| `pipeline_compilations` | Créations de pipelines Tensor ; la réutilisation d’un cache n’en ajoute pas. |
| `compute_passes`, `pipeline_bindings`, `dispatches` | Passes compute, liaisons de pipelines et dispatches effectivement émis par Tensor. |

`reset()` met les événements à zéro et repart des buffers et bytes actuellement
vivants pour les pics. Les bytes désignent les tailles demandées des buffers de
stockage, pas la mémoire VRAM physique : ils excluent staging, shaders et mémoire
interne du pilote. Une vue partage un buffer et ne compte pas comme une allocation.
Le stockage vide réserve actuellement un buffer minimal de quatre octets sans
transfert. Une matérialisation GPU par copie peut produire une soumission et une
fence sans passe compute ni dispatch.

## Attribution et durée de vie

Le profil compte uniquement les appels Tensor sur le device sélectionné. Les
commandes GFX externes et `device.reset_command_stats()` ne modifient pas ses
compteurs. Les calculs CPU et les opérations d’un autre device sont exclus.
Une extraction GPU explicite par `cpu()` est comptée comme download.

Un stockage conserve les parents nécessaires jusqu’à l’achèvement des commandes.
Ses bytes restent donc actifs tant que son buffer est retenu. Le profil ne retient
aucun tenseur : il garde son device et ses données de période. L’état interne du
device disparaît quand il n’a plus de stockage Tensor ni de profil vivant ; aucun
historique global de devices ou de périodes n’est conservé.

La période suit les appels du thread appelant ; elle ne constitue pas une garantie
de collecte concurrente. La collecte inactive garde le suivi des buffers nécessaire
pour ouvrir un profil sur des tenseurs déjà résidents. La collecte active ajoute
les incréments des périodes vivantes à chaque événement. Le coût dépend du nombre
de profils simultanés et du parcours ; le benchmark `Neural.sx` distingue
`--measure-mlp`/`--measure-cnn`/`--measure-rnn` (sans profil) de leurs variantes
`--profile-mlp`/`--profile-cnn`/`--profile-rnn`. Les snapshots et leur impression
ont lieu après les sections chronométrées.

La qualification Release du 5 octobre 2026 sur Apple M3 Pro / Metal conserve
75 processus (cinq répétitions par variante/famille et des contrôles A/A).
Sur les neuf étapes résidentes complètes, les écarts médians avec une période
active vont de −3,11% à +5,35% par rapport au candidat sans profil. D’autres
phases présentent des écarts positifs jusqu’à +20,75% ; les contrôles A/A
présentent eux-mêmes des signaux. Cela ne démontre ni coût nul ni non-régression
universelle. Les données brutes, MAD et contre-épreuves sont conservées dans
`Silex-Benchmarks/Sources/TensorStableCompute/Baselines/2026-10-05-macos-arm64-public-profile/`.
