# Issue: GridFinder silently detects the wrong board size under stone occlusion

## Symptom
On busy/late-game boards (e.g. `tests/vision/resources/game_simple/size_13/move_26.png`, a near-full
13x13 board), `GridFinder::analyseGeometry()` detected only a **9x9** grid, confined to one region of the
photographed board (roughly the least-occluded corner), instead of the true 13x13 grid spanning the whole
board. This was reproduced and confirmed as a genuine pipeline bug (not a ground-truth/test bug) by manually
running the pipeline through `visionTuner` and visually comparing the detected grid against the source image.

## Root cause

### 1. Physical trigger: occlusion breaks Hough line segments
`analyseGeometry()` (`gridFinder.cpp`) detects grid-line candidates with `cv::HoughLinesP` on the Canny-edge
image of the coarse warp ($I_{B_0}$, see `src/vision/core/README.md` for space terminology):

```cpp
cv::HoughLinesP(edges, lines,
                1,                       // rho resolution
                std::numbers::pi / 180., // theta resolution
                80,                      // threshold (votes)
                100,                     // minLineLength
                20                       // maxLineGap
);
```

`minLineLength=100` requires at least one *unbroken* 100px segment of a grid line to be visible. On a
near-full board, a grid line running through a densely-stoned region can have every gap between consecutive
stones shorter than 100px (stone diameter is close to grid spacing), so **the line produces zero Hough
detections and is completely absent** from the candidate list — not just noisy, genuinely missing. Grid
lines through sparser regions of the same board are detected normally. For a 13x13 board this can easily
leave only ~9 of the 13 true lines per axis in the candidate set, clustered in whichever region had fewer
stones on it.

### 2. The actual bug: board-size selection rewarded the smaller, wrong size
When the direct candidate count isn't a clean valid size, `analyseGeometry()` falls back to
`findGrid()` (`gridLatticeFinder.cpp`), which fits a 1D equally-spaced lattice per axis for each candidate
`N ∈ {9, 13, 19}` and then jointly picks the best `N` across both axes.

Given only ~9 detected line candidates (all genuine, evenly-spaced, just a subset of the true 13), the
per-axis fit (`selectGridByLatticeFit`) succeeds *equally well* for `N=9` and `N=13`: both achieve the same
number of inliers and ~0 RMS error — `N=13`'s fit simply places the same 9 real points at higher lattice
indices (e.g. 4–12 instead of 0–8) and reconstructs the remaining 4 positions by extrapolation.

The joint selector, however, scored candidates like this:

```cpp
const double completeness = safeRatio(totalInliers, 2u * N);                            // InliersTotal / (2*N)
const double coverage     = safeRatio(totalInliers, vCenters.size() + hCenters.size());  // InliersTotal / (detections)
const double balanced      = harmonicMean(completeness, coverage);                        // used as the *primary* sort key
```

With 9 detected points on each axis (18 total) matched perfectly by both candidates:

| N  | totalInliers | completeness = inliers/(2N) | coverage | balanced |
|----|-------------:|-----------------------------:|---------:|---------:|
| 9  | 18           | 18/18 = **1.00**             | 1.00     | **1.00** |
| 13 | 18           | 18/26 = **0.69**             | 1.00     | 0.82     |

`completeness` is trivially maximized by picking the **smallest** `N` that happens to equal however many
lines survived occlusion — it has nothing to do with whether that `N` is physically plausible. Since
`balanced` was the dominant comparison key, **`N=9` always won whenever occlusion reduced the detected line
count below the true board size**, regardless of fit quality. This exactly reproduces the reported failure
mode.

## The fix
We know something `findGrid()` previously ignored: the coarse warp $I_{B_0}$ is a **fixed-size canvas**
(`WARP_OUT_SIZE = 1000`) that `BoardFinder` already warps so the physical board fills it (see
`src/vision/core/README.md`, `boardFinder.hpp`). So the *correct* `N`'s reconstructed lattice should span
almost the entire canvas — while a too-small `N` fitted from an occlusion-shrunk subset spans only a
fraction of it, and a too-large `N` would imply a span exceeding it. This is a strong, independent, physical
signal that doesn't depend on how many lines happened to survive occlusion.

`findGrid()` now takes the known canvas extent per axis and scores each candidate by how well its implied
physical span matches it, peaking at a perfect match and falling off symmetrically for over/undershoot:

```cpp
// expectedExtent <= 0 disables the check (returns a neutral 1.0), e.g. for callers without extent knowledge.
auto extentFitScore = [](double span, double expectedExtent) -> double {
    if (!(expectedExtent > 0.0) || !std::isfinite(span)) {
        return 1.0;
    }
    const double ratio = span / expectedExtent;
    return 1.0 / (1.0 + std::abs(ratio - 1.0));
};
...
const double spanV     = vTmp.back() - vTmp.front();
const double spanH     = hTmp.back() - hTmp.front();
const double extentFit = harmonicMean(extentFitScore(spanV, expectedExtentV), extentFitScore(spanH, expectedExtentH));
```

`extentFit` and the existing `balanced` score are **multiplied** into one ranking score, with `rms` /
`inliersTotal` / smaller-`N` left as deterministic tie-breaks:

```cpp
const double score = extentFit * balanced;
```

Both have to hold at once: `extentFit` rejects a smaller `N` that "fully explains" an occlusion-shrunk set
of detections, and `balanced` rejects a larger `N` that only fits by extrapolating lattice lines no
detection supports.

> **Amendment.** `extentFit` was first made the *dominant, lexicographic* criterion, ahead of
> `balanced`/`completeness`/`coverage`. That was wrong and broke `angled_hard/angle_6` (detected 19x19):
> `expectedExtent` is BoardFinder's canvas size, so it silently assumes the grid fills that canvas. It does
> not. `GeometryReport.StageAccuracy` measures the true fraction per image (`gridFill`) and it ranges
> **0.595 to 2.348** across the labelled corpus — wrong on 12 of 18 images, because BoardFinder has no
> consistent output contract (it frames the physical board on some images and the outermost grid lines on
> others) and board margins differ per board. A heuristic whose precondition the caller cannot verify must
> not be able to veto direct line evidence on its own.

The canvas extent is threaded in from `gridFinder.cpp`, which already has the $I_{B_0}$ image at this point:

```cpp
if (!findGrid(vGrid, hGrid, vGridAttempt, hGridAttempt,
              static_cast<double>(input.imageB0.cols), static_cast<double>(input.imageB0.rows))) {
```

### Why this fixes the failure
With true spacing `s` estimated from the 9 real detected points (same for every candidate `N`, since it's
the same underlying data), the implied span for each candidate is `(N-1) * s`. Illustrating with a
representative spacing where the true 13-line board's span nearly fills the ~1000px canvas:

| N  | implied span | extentFit (canvas ≈ 1000px) |
|----|-------------:|------------------------------:|
| 9  | 8s  ≈ 620px  | 1/(1+\|0.62-1\|) ≈ **0.72** (undershoot) |
| 13 | 12s ≈ 920px  | 1/(1+\|0.92-1\|) ≈ **0.93** (near-perfect fill) |
| 19 | 18s ≈ 1380px | 1/(1+\|1.38-1\|) ≈ **0.72** (overshoot) |

`N=13` now wins decisively on `extentFit` alone, before `completeness`/`coverage` even get compared — fixing
the bug at its root instead of special-casing around it.

## Status
Implemented in `src/vision/core/gridLatticeFinder.{hpp,cpp}` and `src/vision/core/gridFinder.cpp`, and
verified by a full rebuild + test run:

- `move_26`, the reported failure, now detects 13x13. ✅
- `angled_hard/angle_6` also went 19x19 → 13x13 (fixed by the amendment above). ✅
- `Process.Find_Board_Hard` now passes; `visionCore.unit` is fully green.
- No image in `GeometryReport.StageAccuracy` regressed; every other row is unchanged.

## Related, not fixed here
- **`move_27` still detects 19x19**, but not for this reason — its spacing estimate collapses to ~39px
  against a true grid spacing of ~78px, i.e. it locks onto a half-spacing harmonic, most likely from
  stone-edge lines appearing between real grid lines on a near-full board. With that spacing every `N` is
  mis-scaled, so `N=19`'s implied span (706px) beats `N=13`'s (470px) under any extent rule. This failed
  identically before and after the fix above. Fixing it means fixing spacing estimation, not selection.
- The underlying occlusion-driven candidate loss (root cause #1) is only *tolerated*, not eliminated. A grid
  line fully hidden behind a dense stone cluster on every axis-window still can't be recovered by this fix if
  it also affects the phase/spacing estimate itself. A future improvement could raise Hough recall (e.g. a
  projection-profile approach less sensitive to a single long occlusion run) to reduce how often lines vanish
  from the candidate set in the first place.
- `Process.Board_Detect_Easy` fails on all 6 `angled_easy` images (stage-1 corner error 118-122px against a
  100px tolerance). Pre-existing and unrelated to grid selection — it is a BoardFinder framing issue.
