# Texture Rotation Coordinate Finder

Finds your coordinates in a Minecraft world from the **texture rotations** of
blocks around you.

Many blocks (grass, sand, stone, netherrack, concrete powder, ...) pick their
texture variant from a hash of their absolute position. That hash depends only
on `(x, y, z)` - not on the world seed. So a handful of observed rotations
pins down where you are, on any server, without knowing the seed.

Reading about 20 blocks pins you down inside a 10k x 10k region, and about 33
blocks does it across the entire world (both 95% confidence, full y range of
-64 to 320). The GPU matcher scans about **150 billion candidate positions per
second** on an RTX 4060.

## Contents

| File | What it is |
|---|---|
| `texture_finder_gui.py` | Tkinter GUI. Enter observations, pick CPU or GPU, run. |
| `tex_match.c` | CPU matcher (multithreaded). |
| `tex_match_gpu.cu` | CUDA matcher. Field + sieve; ~150 G candidates/s on a 4060. |
| `tex_conf.py` | Loads CoordsFinder `.conf` scan files into the GUI. |
| `build_tex.bat` | Builds both binaries on Windows. |

## Build

Requires **gcc** (MinGW/MSYS2) for the CPU matcher. The GPU matcher additionally
needs the **CUDA Toolkit** and **Visual Studio 2019-2022** (nvcc's host compiler).

```bat
build_tex.bat
```

The GPU step is skipped automatically if `nvcc` is not on PATH - the GUI still
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

**Loading a `.conf`.** If you already have observations saved in a CoordsFinder
`.conf` file, `Load .conf...` (or `Load latest .conf`, which takes the newest one
in the script folder) fills in the centre, radius, Y range, facings and
observations. Nothing runs until you press Run, so the values can be checked
first. Settings with no equivalent here are skipped and listed in the log.

## Observations

One per line, relative to any block you pick as the origin:

```
dx,dy,dz,rot,mod[,modeff]
```

- `dx,dy,dz` - offset from your reference block
- `rot` - the variant you observed
- `mod` - number of variants the block has (4 for most, 16 for netherrack)
- `modeff` - optional; use `2` on mod-4 blocks where you can only tell
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

1. **Field kernel** - computes the rotation value once per absolute cell and
   packs it into a bitfield (16-32 cells per 64-bit word). Each cell is hashed
   exactly once, I could not achieve any form of useful lazy-hashing with
   small enough overhead to be worth implementing.
2. **Sieve kernel** - matches candidates as shifted 64-bit AND masks over that
   field. One `xor`/`and`/`fold` tests **32 candidates at once**, and a word
   drops out as soon as every candidate in it is dead.

So the per-candidate cost is a few logic ops rather than a hash, with no
per-candidate branch. Tiles are processed in expanding rings from the search
centre so near matches surface first.

Because a word drops out after about three observations on average, throughput
barely moves with the size of the observation set. Billions of candidate
positions per second on an RTX 4060:

| observations | 12 | 20 | 30 | 45 | 57 |
|---|---|---|---|---|---|
| known direction | 150 | 154 | 149 | 154 | 153 |
| unknown, 4 rotations (`--facing all4`) | 338 | 358 | 354 | 356 | 374 |
| unknown, 8 transforms (`--facing all`) | 428 | 460 | 423 | 466 | 468 |

Searching an unknown direction is *cheaper per candidate*, not more expensive:
the field is built once and every transform reuses it, so the one-off hashing
cost is amortised over 4 or 8 times as many candidates. Wall-clock time still
goes up, just sub-linearly.

## Version

Targets the 26.1.2 block-variant tables and hash. Earlier versions used the
same coordinate hash but different blockstate variant lists.

## Credits

The `.conf` scan file format read by `tex_conf.py` comes from
[CoordsFinder](https://github.com/ALaggyDev/CoordsFinder) by ALaggyDev.
Thanks to that project for the format, which lets their `.conf` files be
loaded here directly.

## Licence

MIT - see [LICENSE](LICENSE).
