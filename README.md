# Texture Rotation Coordinate Finder

Finds your coordinates in a Minecraft world from the **texture rotations** of
blocks around you.

Many blocks (grass, sand, stone, netherrack, concrete powder, ...) pick their
texture variant from a hash of their absolute position. That hash depends only
on `(x, y, z)` -- not on the world seed. So a handful of observed rotations
pins down where you are, on any server, without knowing the seed.

Reading ~12 blocks is usually enough to get a unique hit inside a 10k x 10k
region. The GPU matcher scans about **150 billion candidate positions per
second** on an RTX 4060.

## Contents

| File | What it is |
|---|---|
| `texture_finder_gui.py` | Tkinter GUI. Enter observations, pick CPU or GPU, run. |
| `tex_match.c` | CPU matcher (multithreaded). |
| `tex_match_gpu.cu` | CUDA matcher. Field + sieve; ~150 G candidates/s on a 4060. |
| `build_tex.bat` | Builds both binaries on Windows. |

## Build

Requires **gcc** (MinGW/MSYS2) for the CPU matcher. The GPU matcher additionally
needs the **CUDA Toolkit** and **Visual Studio 2019-2022** (nvcc's host compiler).

```bat
build_tex.bat
```

The GPU step is skipped automatically if `nvcc` is not on PATH -- the GUI still
works, CPU-only.

The build produces a fatbin with native code for Turing (`sm_75`, RTX 20xx /
GTX 16xx), Ampere (`sm_86`, RTX 30xx), Ada (`sm_89`, RTX 40xx) and Blackwell
(`sm_120`, RTX 50xx), plus `compute_120` PTX so newer cards can JIT. The
committed `tex_match_gpu.exe` is that same fatbin, so it runs on any of those
without rebuilding.

To build for just your own card (faster to compile):

```bat
set CUDA_GENCODE=-arch=sm_89  &&  build_tex.bat
```

## Run

```bat
python texture_finder_gui.py
```

Needs Python 3 with Tkinter (included in the standard python.org installer).

## Observations

One per line, relative to any block you pick as the origin:

```
dx,dy,dz,rot,mod[,modeff]
```

- `dx,dy,dz` -- offset from your reference block
- `rot` -- the variant you observed
- `mod` -- number of variants the block has (4 for most, 16 for netherrack)
- `modeff` -- optional; use `2` on mod-4 blocks where you can only tell
  *normal* from *mirrored* (stone, bedrock, deepslate, sculk) rather than the
  full rotation

There is also a mask form for when one face narrows the variant to a *set*
rather than a single value (netherrack):

```
dx,dy,dz,mHEX,mod        e.g.  0,0,0,m807,16  = variants {0,1,2,11}
```

If you do not know which way you were facing, the GUI's **try all 4 directions**
option searches every rotation of your observation set.

The GUI's built-in help lists every supported block and how many observations
you need for a given search volume.

## CLI

The GUI is a wrapper; both matchers run standalone.

```
tex_match.exe     <cx> <cy> <cz> <radius> <threads> [--ymin N] [--ymax N]
                  [--facing 0,1,...|all4|all] [--view-relative] <obs>...

tex_match_gpu.exe <cx> <cy> <cz> <radius> [--ymin N] [--ymax N]
                  [--facing 0,1,...|all4|all] [--stop N] [--naive]
                  [--view-relative] <obs>...
```

`--stop N` halts once N matches are confirmed nearer than anything unsearched,
which is much faster when you only want the closest hit.

## How the GPU matcher works

The search is split into two kernels:

1. **Field kernel** -- computes the rotation value once per absolute cell and
   packs it into a bitfield (16-32 cells per 64-bit word). Each cell is hashed
   exactly once, which is the information-theoretic minimum.
2. **Sieve kernel** -- matches candidates as shifted 64-bit AND masks over that
   field. One `xor`/`and`/`fold` tests **32 candidates at once**, and a word
   drops out as soon as every candidate in it is dead.

So the per-candidate cost is a few logic ops rather than a hash, with no
per-candidate branch. Tiles are processed in expanding rings from the search
centre so near matches surface first.

Because a word drops out after about three observations on average, throughput
barely moves with the size of the observation set. Measured on an RTX 4060,
one facing:

| observations | 12 | 20 | 30 | 45 | 57 |
|---|---|---|---|---|---|
| G candidates/s | 150 | 153 | 153 | 153 | 154 |

Searching all 8 facings is *cheaper per candidate*, not more expensive, because
the field is built once and reused by every transform: ~415 G candidates/s with
12 observations.

## Version

Targets the 26.1.2 block-variant tables and hash. Earlier versions used the
same coordinate hash but different blockstate variant lists.

## Licence

MIT -- see [LICENSE](LICENSE).
