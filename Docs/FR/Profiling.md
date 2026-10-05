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

Tous les événements comptent le travail émis pendant la période. Les nombres
de buffers et les bytes sont, eux, l’état courant du device, y compris les
réservations Tensor créées avant l’ouverture de la période. Les buffers prêts
peuvent être réutilisés par Tensor : distinguer les ressources natives vivantes
des buffers encore utilisés par les valeurs.

| Champs de `ProfileStatistics` | Signification |
| --- | --- |
| `uploads`, `downloads` | Transferts Tensor CPU→GPU et GPU→CPU. |
| `uploaded_bytes`, `downloaded_bytes` | Octets logiques de ces transferts, selon le dtype. |
| `buffer_allocations`, `buffer_releases` | Créations et libérations effectives de buffers de stockage Tensor pendant la période. Une réutilisation ne crée pas de buffer. Les libérations peuvent concerner des buffers antérieurs. |
| `buffer_reuses` | Reprises de buffers prêts, sans nouvelle création native. |
| `cache_evictions` | Buffers retirés du cache, pour respecter sa capacité ou le vider au dernier stockage. |
| `live_buffers`, `peak_live_buffers` | Buffers natifs Tensor vivants, y compris les buffers prêts en cache, et maximum depuis le début de la période. |
| `in_use_buffers`, `peak_in_use_buffers` | Buffers possédés par les stockages des valeurs et maximum sur la période ; les vues partagent leur stockage. |
| `active_bytes`, `peak_bytes` | Somme des tailles demandées des buffers utilisés par les valeurs et maximum sur la période. |
| `allocated_bytes`, `peak_allocated_bytes` | Somme des tailles demandées de tous les buffers natifs vivants, cache inclus, et maximum sur la période. |
| `cached_buffers`, `cached_bytes` | Buffers prêts disponibles et somme de leurs tailles demandées. |
| `cache_capacity_buffers`, `cache_capacity_bytes` | Limites du cache par device : actuellement 32 buffers et 1 MiB. |
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

## Réutilisation et mémoire réservée

Le cache privé de Tensor reprend uniquement un buffer dont la fence est achevée
et dont aucun stockage vivant, vue ou branche ne dépend. La taille et le dtype
doivent correspondre exactement. Les buffers trop grands sont libérés directement.
Le cache respecte simultanément 32 entrées et 1 MiB par device ; l’entrée prête
la plus ancienne est retirée lorsqu’une limite serait dépassée. L’optimisation
réutilise le stockage, sans regrouper de commandes ni éviter les fences du calcul.

`live_buffers == in_use_buffers + cached_buffers` et
`allocated_bytes == active_bytes + cached_bytes` décrivent l’état courant.
Les maximums sont observés séparément : ils ne s’additionnent pas nécessairement.
Après un calcul temporaire, `active_bytes` peut revenir à son niveau précédent
alors que `allocated_bytes` reste plus élevé et que `buffer_releases` ne change
pas : le buffer est disponible pour un prochain calcul. `reset()` ne vide pas
le cache. Il repart de l’état courant pour tous les pics et remet seulement les
événements à zéro ; `snapshot()` ne modifie aucune réservation.

Le cache se vide lorsque le dernier stockage Tensor du device disparaît, même
si un `Profile` reste vivant. Détruire un profil ne vide pas le cache des
valeurs encore vivantes. Les tailles demandées incluent les minimums du provider,
mais ne mesurent ni la mémoire physique Metal ni les allocations internes du
pilote. Le cache n’ajoute pas de transfert, ni d’attente au profilage.

Les nouveaux champs de `ProfileStatistics` ont une valeur par défaut nulle,
pour conserver les constructions nommées qui ne renseignent que les anciens
champs. Un instantané produit par `Profile` renseigne les limites effectives.

La période suit les appels du thread appelant ; elle ne constitue pas une garantie
de collecte concurrente. La collecte inactive garde le suivi des buffers nécessaire
pour ouvrir un profil sur des tenseurs déjà résidents. La collecte active ajoute
les incréments des périodes vivantes à chaque événement. Le coût dépend du nombre
de profils simultanés et du parcours ; le benchmark `Neural.sx` distingue
`--measure-mlp`/`--measure-cnn`/`--measure-rnn` (sans profil) de leurs variantes
`--profile-mlp`/`--profile-cnn`/`--profile-rnn`. Les snapshots et leur impression
ont lieu après les sections chronométrées.

La qualification initiale du profil public, avant la réutilisation de buffers,
en Release le 5 octobre 2026 sur Apple M3 Pro / Metal conserve
75 processus (cinq répétitions par variante/famille et des contrôles A/A).
Sur les neuf étapes résidentes complètes, les écarts médians avec une période
active vont de −3,11% à +5,35% par rapport au candidat sans profil. D’autres
phases présentent des écarts positifs jusqu’à +20,75% ; les contrôles A/A
présentent eux-mêmes des signaux. Cela ne démontre ni coût nul ni non-régression
universelle. Les données brutes, MAD et contre-épreuves sont conservées dans
`Silex-Benchmarks/Sources/TensorStableCompute/Baselines/2026-10-05-macos-arm64-public-profile/`.

## Résultat mesuré de la réutilisation

La comparaison Release du 5 octobre 2026 sur Apple M3 Pro / Metal utilise le
même compilateur exact, 16 paires eager/cache et huit contrôles A/A de chaque
binaire. Trois étapes résidentes Transformer passent de 40,260 ms à 37,853 ms
en médiane ; la différence appariée médiane est −5,89 %, avec MAD 1,20 point
et 14 paires favorables sur 16. Les contrôles sont valides selon la règle
fixée avant mesure. Les 1 700 valeurs complètes des six modèles, gradients et
états compris, restent qualifiées à la tolérance 2e-5 avec AdamW epsilon1e-4.

Les cinq autres modèles et les douze microcas ne démontrent pas de gain selon
la règle d’admission. Les coûts froids restent indéterminés. Les créations
natives Transformer passent de 1 365 à 643 en médiane ; les passes et les
soumissions restent identiques. Cette baisse des créations ne signifie pas
que toutes les charges s’accélèrent. Le RSS médian du processus des six
familles passe de 81 510 400 à 83 869 696 octets ; il inclut le pilote et
l’allocateur, et ne mesure pas la mémoire physique GPU. Le cache garde au
plus 1 MiB de tailles de buffers demandées par device.

Sources, configurations, données brutes, dispersion, contrôles et vérificateur
appartiennent à `Silex-Benchmarks/Sources/TensorStableCompute/Baselines/` sous
`2026-10-05-macos-arm64-ready-buffer-reuse-micro` et
`2026-10-05-macos-arm64-ready-buffer-reuse-models`. Ce résultat concerne ces
charges, ce matériel et ces binaires ; aucune généralisation de performance
aux autres cibles n’est établie.
