# Resources

Images used by the board widget. They are loaded at runtime from this folder (`GUI_RESOURCES_DIR`).

| File               | Used for                                     |
| ------------------ | -------------------------------------------- |
| `anime_black.png`  | Black stone                                  |
| `anime_white.png`  | White stone                                  |
| `anime_shadow.png` | Stone shadow (not drawn yet)                 |
| `anime_*.svg`      | Vector sources of the stone PNGs             |
| `board_textures/`  | Board backgrounds selectable in the Settings |

## Stones

The stones are the "Anime" theme of [OGS](https://online-go.com), taken from its board library [online-go/goban][goban].

| File               | Source                                | License                              | Modified                        | Accessed   |
| ------------------ | ------------------------------------- | ------------------------------------ | ------------------------------- | ---------- |
| `anime_black.png`  | [goban: anime_black.svg][ogs-black]   | [Apache 2.0](LICENSE-Apache-2.0.txt) | Yes: SVG rendered to 512 px PNG | 2025-07-26 |
| `anime_white.png`  | [goban: anime_white.svg][ogs-white]   | [Apache 2.0](LICENSE-Apache-2.0.txt) | Yes: SVG rendered to 512 px PNG | 2025-07-26 |
| `anime_shadow.png` | [goban: anime_shadow.svg][ogs-shadow] | [Apache 2.0](LICENSE-Apache-2.0.txt) | Yes: SVG rendered to 512 px PNG | 2026-09-30 |
| `anime_black.svg`  | [goban: anime_black.svg][ogs-black]   | [Apache 2.0](LICENSE-Apache-2.0.txt) | No                              | 2026-09-30 |
| `anime_white.svg`  | [goban: anime_white.svg][ogs-white]   | [Apache 2.0](LICENSE-Apache-2.0.txt) | No                              | 2026-09-30 |
| `anime_shadow.svg` | [goban: anime_shadow.svg][ogs-shadow] | [Apache 2.0](LICENSE-Apache-2.0.txt) | No                              | 2026-09-30 |

Copyright (C) Online-Go.com. Licensed under the Apache License, Version 2.0; see [LICENSE-Apache-2.0.txt](LICENSE-Apache-2.0.txt).
The access date is when the file was added to this repository.
The source links are pinned to the later goban commit the PNGs were verified against on 2026-09-29; the SVGs are byte-identical to it.

The renderer loads the PNGs rather than the SVGs: the stones use SVG masks, filters and clip paths, which Qt SVG only supports from Qt 6.7.
To change the stone resolution, re-render the SVGs (e.g. with Inkscape) instead of scaling the PNGs.

[goban]: https://github.com/online-go/goban
[ogs-black]: https://github.com/online-go/goban/blob/e61c56e246726481ab39bdddbfc886a4df06b786/assets/img/anime_black.svg
[ogs-white]: https://github.com/online-go/goban/blob/e61c56e246726481ab39bdddbfc886a4df06b786/assets/img/anime_white.svg
[ogs-shadow]: https://github.com/online-go/goban/blob/e61c56e246726481ab39bdddbfc886a4df06b786/assets/img/anime_shadow.svg

## Board textures

Every image in `board_textures/` shows up in **Settings → Style**, listed by its file name without extension.
Without a selected texture the board is drawn in a plain colour.

| Texture                      | Source                                    | License                                                   | Format | Modified | Accessed   |
| ---------------------------- | ----------------------------------------- | --------------------------------------------------------- | ------ | -------- | ---------- |
| `1K-wood_fine_8-diffuse.jpg` | [ShareTextures: Wood Fine 8][wood-fine-8] | [CC0][cc0] ([ShareTextures terms][sharetextures-license]) | 1K JPG | No       | 2026-09-29 |
| `2K_afromosia_basecolor.png` | [ShareTextures: Afromosia][afromosia]     | [CC0][cc0] ([ShareTextures terms][sharetextures-license]) | 2K PNG | No       | 2026-09-29 |
| `Wood094_2K-PNG_Color.png`   | [ambientCG: Wood094][wood094]             | [CC0][cc0] ([ambientCG license][ambientcg-license])       | 2K PNG | No       | 2026-09-29 |
| `Wood095_2K-PNG_Color.png`   | [ambientCG: Wood095][wood095]             | [CC0][cc0] ([ambientCG license][ambientcg-license])       | 2K PNG | No       | 2026-09-29 |
| `anime_board.svg`            | [goban: anime_board.svg][ogs-board]       | [Apache 2.0](LICENSE-Apache-2.0.txt) (Online-Go.com)      | SVG    | No       | 2026-09-30 |

[ogs-board]: https://github.com/online-go/goban/blob/e61c56e246726481ab39bdddbfc886a4df06b786/assets/img/anime_board.svg
[wood-fine-8]: https://www.sharetextures.com/textures/wood/wood-fine-8
[afromosia]: https://www.sharetextures.com/textures/wood/afromosia
[wood094]: https://ambientcg.com/view?id=Wood094
[wood095]: https://ambientcg.com/view?id=Wood095
[cc0]: https://creativecommons.org/publicdomain/zero/1.0/
[sharetextures-license]: https://www.sharetextures.com/p/terms
[ambientcg-license]: https://docs.ambientcg.com/license/

### Adding a texture

- **Format:** Any format Qt can read. JPEG keeps photos small; PNG of the same image is several times larger.
- **Which file:** Texture sites ship PBR sets with several maps. Only the colour map is needed (named _Color_, _Albedo_, _Diffuse_ or _BaseColor_).
- **Size:** About 2K is plenty; the board is never drawn larger than the screen.
- **Crop:** The board uses the centre square of the image, so non-square images work but lose their ends.
- **License:** Textures are shipped with the application. Prefer CC0 sources such as [ambientCG](https://ambientcg.com/), [ShareTextures](https://www.sharetextures.com/) or [Poly Haven](https://polyhaven.com/textures) and fill in a table row with the texture's own page, the downloaded variant, whether you changed the file and the access date.
