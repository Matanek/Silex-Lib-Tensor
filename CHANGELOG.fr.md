# Notes de version de Tensor

## [Unreleased]

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
