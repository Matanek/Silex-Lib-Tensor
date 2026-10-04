# Notes de version de Tensor

## [Unreleased]

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
