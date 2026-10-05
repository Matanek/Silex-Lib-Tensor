# Notes de version de Tensor

## [Unreleased]

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
