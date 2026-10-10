# Notes de version de Tensor

## [Unreleased]

- Adopte la déclaration `Result` unique : l’appel ordinaire rend la valeur
  ou arrête le programme sur erreur ; `try appel()` conserve le `Result`. Les
  anciens appels stockant un résultat et propagations deviennent explicites.
  Cette révision requiert la toolchain de développement avec cette sémantique
  de `try` ; sa publication doit coordonner les versions du compilateur et des
  packages.

- Consolide la matrice des six modèles, le domaine des plans et la conclusion float16 indéterminée ; distingue la baseline publiée du compilateur source requis par les capacités Unreleased. Voir la [référence](Docs/FR/Reference.md).

- `Neural.TrainingStep.mse` prépare explicitement une étape fermée MSE/AdamW : forward et perte préparés, backward de référence et update groupée. Le plan indépendant relit poids, moments, taux et trainabilité ; signatures invalides rejetées avant zero-grad, statistiques et observations détachées. Gains dépendants de la charge, limites et contre-épreuves dans le [guide](Docs/FR/TrainingSteps.md).

- `Neural.Layer.prepare_forward`, `clear_forward_plan` et `forward_plan_statistics` préparent un forward GPU float32 pour Sequential, Dense, Activation et LayerNorm exacts. Cache privé d’un plan, gardes et fallback eager expliqués, paramètres courants et autograd conservés, aucune capture de callback ou de storage d’exemple. Gains chauds locaux de 40,15 %, 33,42 % et 19,58 % sur trois MLP Metal ; préparation et profil séparés ; l’étape MSE/AdamW est décrite dans le [guide d’entraînement préparé](Docs/FR/TrainingSteps.md), les autres familles restent hors plan. 464 tests/110 sources, Debug/Release et six cibles compilées/liées ; voir le [guide](Docs/FR/ForwardPlans.md).

- Fusionne l’addition d’epsilon et sqrt dans LayerNorm GPU float32 : six passes au lieu de sept, gradients et durées de vie qualifiés, réductions inchangées. Sur Apple M3 Pro/Metal, gains admis de 9,50 %, 10,38 % et 9,08 % sur les petites normalisations et la vue stridée ; les six modèles et les autres tailles restent indéterminés. Corpus de 457 tests, six cibles compilées/liées, campagnes et limites documentées dans le [profilage](Docs/FR/Profiling.md).

- Réutilise les buffers GPU prêts à taille/dtype exacts dans un cache privé borné à 32 entrées et 1 MiB par device, vidé au dernier stockage. Le profil distingue créations natives, réutilisations, réservations en cache et mémoire utilisée par les valeurs. Sur Apple M3 Pro/Metal, trois étapes Transformer ont un gain apparié admis de 5,89 % ; les cinq autres familles et microcas restent indéterminés. Les données brutes et le coût mémoire sont conservés dans Silex-Benchmarks.

- Les gardes de durée de vie de MultiHeadAttention, Transformer et de la chaîne résidente vérifient des bornes par variante avec tous les ancêtres retenus, indépendantes de la vitesse de complétion GPU, ainsi que les nombres exacts d’allocations et la libération complète.

- Ajoute LSTM batch-first i/f/g/o avec paire hidden/cell et séquence explicites, quatre paramètres, oracles CPU/Metal, segmentation et reprise exacte.

- Ajoute GRU batch-first reset-after r/z/n avec état et séquence explicites, quatre paramètres nommés, oracles CPU/Metal, segmentation et reprise exacte.

- `Neural.TransformerBlock` compose attention, résidus, LayerNorm et MLP avec pre/post-norm explicite et dropout reproductible. Modèle causal externe avec positions séparées, apprentissage et reprise exacte CPU/Metal qualifiés ; voir le [guide](Docs/FR/Transformer.md).

- `Neural.MultiHeadAttention` ajoute quatre projections entraînables au registre Layer, self/cross-attention et dropout à seed/itération explicites. Queries sans clé neutralisées après projection, poids optionnels appliqués, état partagé et reprise CPU/Metal qualifiés ; voir le [guide](Docs/FR/MultiHeadAttention.md).

- `Neural.attention` compose les produits Q/K/V et le softmax masqué en self/cross-attention batch-first de rang3/4. Poids optionnels différentiables, gradients partagés, formes vides et calcul CPU/Metal résident ; voir le [guide](Docs/FR/Attention.md).

- `Tensor.masked_softmax(mask,axis)` normalise les scores visibles finis et définit des sorties/gradients nuls pour les positions et lignes entièrement masquées. Non-finis exclus ignorés, formes vides et calcul CPU/Metal résident qualifiés par oracle et gradcheck ; voir le [guide](Docs/FR/MaskedSoftmax.md).

- Masques de séquence `Neural.sequence_mask` et `Neural.causal_mask` : booléens CPU, préfixes et positions exactes, offset explicite, composition et transfert GPU explicites ; gradients des sélections qualifiés CPU/Metal.

- `Neural.Embedding` apprend une table float32 seedée à partir d’indices int32 batch-first, avec gradients denses répétés et padding à dérivée arrêtée. Composition Layer, état/reprise et lookup résident CPU/Metal qualifiés ; voir le [guide](Docs/FR/Embeddings.md).

- `Autograd.Operation` et `Autograd.apply` relient une règle extérieure composée de Tensor publics au graphe de premier ordre. Les gradients de feuilles sont préparés avant publication et les valeurs retenues libérées à la consommation/destruction. Voir le [guide](Docs/FR/CustomOperations.md).

- `Data.BatchCursor(...,prefetch:N)` prépare des lots CPU détachés avec STD.Threading, conserve les snapshots consommés et l’ordre des erreurs, et rejoint les workers à la fermeture/destruction. Éligibilité explicite des sources, reprise de checkpoint qualifiée CPU/GPU et borne de lots documentée.

- `Neural.TrainingCheckpoint` sauvegarde et restaure exactement le modèle enregistré, SGD/Adam/AdamW, le curseur, l’étape/calendrier, les modes, le gel et le dropout. Chargement CPU prévalidé avec erreurs récupérables ; les snapshots d’optimiseur exposent un déplacement GPU explicite.

- BatchNorm normalise les lots denses ou NCHW sur CPU/GPU, avec gradients, affine facultatif, statistiques courantes détachées et compteur int64 CPU exact. Conventions population/non biaisée et modes train/eval documentés dans le [guide](Docs/FR/BatchNorm.md).

- Les buffers peuvent déclarer BufferPlacement.cpu pour garder un état exact sur CPU lors des transferts du modèle. La politique model reste le défaut compatible ; un transfert direct incompatible est refusé sans conversion ni readback caché. Voir le [guide](Docs/FR/HostState.md).

- AdamW applique une décroissance découplée des moments, avec état nommé distinct et reprise atomique. LearningRateSchedule fournit constant, paliers, exponentiel et cosinus borné depuis une étape explicite ; taux finis positifs, ordre taux/pas/incrément documenté. Voir le [guide](Docs/FR/OptimizerUpdates.md).

- SGD et Adam capturent et restaurent leurs états nommés avec erreurs récupérables et validation atomique, sans transfert caché. Le taux courant peut changer sans perdre les moments. Correction de biais Adam bornée pour les grands compteurs restaurés et débordement refusé avant mutation. Voir le [guide de reprise](Docs/FR/OptimizerState.md).

- `Tensor.Data.Source`, `TensorSource` et `BatchCursor` produisent des lots supervisés synchrones. Collation stack ou callback nommé avec erreurs récupérables, progression après réussite, reprise du prochain lot et placements/dtypes/gradients préservés. Voir le [guide](Docs/FR/SupervisedBatches.md).

- Ajout de `Tensor.Data.BatchSampler` : lots int32 ordonnés ou mélangés, seed/époque explicites, dernier lot conservé par défaut et snapshot immuable reconstructible. Parcours ordonné sans permutation complète, métadonnées CPU sans travail GPU. Voir le [guide des lots](Docs/FR/BatchSampling.md).

- Layer propage train/eval, placement, gel/dégel et zero_grad aux identités
  enregistrées uniques, y compris en diamant partagé. Hooks locaux protégés de
  mode/placement, is_training commun avec Dropout ; noms/gradients/devices
  prévalidés avant mutation ou transfert. Les anciennes frontières manuelles
  sans registre restent compatibles. Voir le [guide d’état](Docs/FR/ModelState.md).

- Les couches enregistrent explicitement paramètres, buffers et enfants nommés.
  La collecte héritée conserve ordre et noms, déduplique les identités partagées,
  refuse cycles et collisions ; Sequential collecte l’état partagé une fois
  tout en exécutant la couche à chaque occurrence. Voir le
  [guide du registre](Docs/FR/Registration.md).

- Ajout de `Neural.Buffer` détaché, nommé et partagé, avec remplacement contrôlé
  et transferts exacts CPU/GPU. `Parameter.freeze/unfreeze/is_trainable` suspend
  les nouveaux gradients, mises à jour, décroissance, clipping et états SGD/Adam,
  même avec un gradient issu d’un graphe antérieur. Le dégel conserve les moments.
  Voir le [guide des buffers et du gel](Docs/FR/State.md).

- Les classes extérieures peuvent dériver de `Neural.Layer` par son constructeur
  protégé et partager les couches et paramètres existants. Les hooks virtuels
  utilisent l’inférence de mutation du compilateur ; suppression du marqueur
  obsolète de la base. Valeurs et gradients partagés qualifiés CPU/GPU. Voir le
  [guide de composition](Docs/FR/Composition.md).

- Ajout de `Tensor.topk` avec axe explicite, valeurs ordonnées et indices int32
  détachés dans `TopKResult`. Comparaisons numériques CPU exactes et sélection
  float32 GPU résidente ; égalités et NaN conservent l’ordre des indices. Les
  valeurs se différencient par `gather`. k0 et sorties vides définis, formes,
  tailles et indices contrôlés. Voir le [guide top-k](Docs/FR/TopK.md).

- Ajout de `argmin` et `argmax` : indices int32 détachés en ordre logique,
  comparaisons entières CPU exactes et entrée float32 GPU résidente. Égalités et
  NaN retiennent le premier indice ; axes, domaines et largeur d’index contrôlés.
  Les valeurs sélectionnées se différencient par `gather`. Voir le
  [guide des indices](Docs/FR/IndexedReductions.md).

- Ajout de `Tensor.arange` int32 à borne exclusive et `Tensor.linspace` float32
  à count explicite, générés sur CPU. Bornes/pas/cardinalité/valeurs int32 sont
  contrôlés avant allocation ; extrêmes int64 et interpolation de grandes bornes
  finies traités sans overflow artificiel. Transfert GPU explicite. Voir le
  [guide des séquences](Docs/FR/Sequences.md).

- Ajout de `variance` et `standard_deviation` float32 CPU/GPU, avec correction
  entière explicite, axes et domaines validés, calcul centré et gradients résidents.
  L’écart-type choisit un gradient nul pour une variance calculée nulle ; NaN reste visible.
  Voir le [guide des statistiques](Docs/FR/Statistics.md).

- Ajout de `squeeze`, `unsqueeze` et `expand` en vues exactes, et `repeat` en
  tuiles matérialisées, sur les neuf dtypes numériques et booléens CPU/GPU.
  Gradients float32 résidents, axes/facteurs/formes validés et domaines vides
  définis. Le backward de reshape accepte les seeds non contiguës. Voir le
  [guide des formes](Docs/FR/Shapes.md).

- `Tensor.Autograd.grad` retourne des gradients détachés CPU/GPU sans modifier
  les slots, avec seed explicite et zéro optionnel pour les entrées inutilisées.
  Le graphe est consommé après réussite. `Autograd.gradcheck` contrôle le Jacobien
  complet d'une fonction déterministe sur données CPU float32, avec epsilon et
  tolérances explicites et premier écart localisé. Voir le [guide](Docs/FR/Gradients.md).

- Ajout des masques `DType.boolean()`, comparaisons broadcast et logique sur
  CPU/GPU. `Tensor.where` sélectionne des branches numériques avec gradients
  broadcast et sans transfert GPU implicite ; `masked_select` compacte sur CPU
  et différentie la source. Les booléens restent distincts des nombres, avec
  extractions typées et transferts exacts. Voir le [guide des masques](Docs/FR/Masks.md).

- Ajout de `Tensor.Profile` et `Tensor.ProfileStatistics` : coûts GPU propres à
  Tensor, par device et période, instantanés immuables et reset indépendant.
  Transferts, buffers/bytes actifs et de pointe, staging, fences, soumissions,
  compilations, passes, bindings et dispatches sont observables sans readback
  ni attente implicite. Les bytes désignent les tailles de stockage demandées,
  pas la VRAM physique ; les périodes suivent les appels du thread appelant.
  Voir le [guide de profilage](Docs/FR/Profiling.md).

- Adam évite de recalculer les moments nuls après leur première initialisation,
  sur CPU et GPU. Les formules, les valeurs des moments et l’initialisation
  différée des paramètres restent identiques. Sur GPU, les étapes suivantes
  suppriment deux passes et deux allocations temporaires par paramètre actif.

  La [qualification MLP/CNN/RNN](https://github.com/Matanek/Silex-Benchmarks/blob/5599f77a79b065a830740e757501ecad5b5ab167/Sources/TensorStableCompute/Baselines/2026-10-04-macos-arm64-adam-warm-moments/Summary.md) compare Tensor `175c383` et `e742c9a`
  en Release avec Silex 0.47.1 sur macOS arm64 / Metal M3 Pro. Elle conserve
  cinq paires par famille, leur dispersion et un contrôle du même binaire ;
  aucun gain hors de cette configuration n’est revendiqué.
