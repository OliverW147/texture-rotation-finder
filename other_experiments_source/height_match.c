// Grass / surface-height pattern matcher for Minecraft 1.21 / 26.1.2.
//
// You observe the surface (grass-top) Y at a handful of (x,z) offsets, e.g.
//   -8,0:73  -4,0:77  0,0:75  4,0:71  8,0:69
// This tool scans the world for every (x,z) whose exact surface-height profile
// matches that pattern, in two modes:
//   - relative (default): matches the RELIEF (shape), ignoring absolute Y. The
//     pattern and each candidate are levelled by their anchor height before
//     comparison, so it works even if your absolute Y reading is off.
//   - absolute (--absolute): heights must match the given Y values exactly
//     (within tolerance).
//
// Each column's height comes from the exact density engine (height_exact.c),
// which is bit-exact vs Minecraft on intact terrain (caves not yet modelled, so
// a column broken by a surface cave may differ; pick pattern points on solid
// grass and/or use a small tolerance).
//
// Modelled on ore_match.c: center-out tiled scan, OpenMP, 8 orientations,
// matches written to matches.txt. Build:
//   gcc -O2 -fwrapv -fopenmp -D_WIN32 -I cubiomes height_match.c height_exact.c \
//     cubiomes/noise.c cubiomes/biomes.c cubiomes/layers.c cubiomes/biomenoise.c \
//     cubiomes/generator.c cubiomes/finders.c cubiomes/util.c cubiomes/quadbase.c \
//     -lm -o height_match.exe

#include "height_exact.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

typedef int64_t  i64;
typedef uint64_t u64;

#define MAX_PATTERN 256

// ---- pattern ----
static int   PAT_DX[MAX_PATTERN];
static int   PAT_DZ[MAX_PATTERN];
static int   PAT_H [MAX_PATTERN];
static int   PATTERN_COUNT = 0;

// 8 XZ transforms (4 rotations x 2 reflections). Applied to the (dx,dz) offset;
// the height stays attached to the offset.
static void applyTransformXZ(int t, int dx, int dz, int *ox, int *oz) {
    switch (t) {
        case 0: *ox =  dx; *oz =  dz; break;  // identity
        case 1: *ox = -dz; *oz =  dx; break;  // rot 90
        case 2: *ox = -dx; *oz = -dz; break;  // rot 180
        case 3: *ox =  dz; *oz = -dx; break;  // rot 270
        case 4: *ox = -dx; *oz =  dz; break;  // reflect X
        case 5: *ox =  dx; *oz = -dz; break;  // reflect Z
        case 6: *ox =  dz; *oz =  dx; break;  // transpose
        case 7: *ox = -dz; *oz = -dx; break;  // anti-transpose
    }
}

// transformed offsets for each orientation (heights unchanged)
static int TPAT_DX[8][MAX_PATTERN];
static int TPAT_DZ[8][MAX_PATTERN];

// ---- tile task (center-out) ----
typedef struct { int x0, z0, x1, z1; double distSq; } Tile;
static int cmpTile(const void *a, const void *b) {
    double da = ((const Tile*)a)->distSq, db = ((const Tile*)b)->distSq;
    return (da < db) ? -1 : (da > db);
}

// ---- shared output ----
static FILE *g_matchFile = NULL;
static long long g_matchCount = 0;
static omp_lock_t g_matchLock;

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s <seed> <specDir> <cx> <cz> <radiusBlocks> [options] <dx,dz:height>...\n"
        "\n"
        "  <cx> <cz>         search center (block coords)\n"
        "  <radiusBlocks>    search radius in blocks (e.g. 1000, up to ~30000000)\n"
        "  <dx,dz:height>    pattern point: surface Y at offset (dx,dz) from center\n"
        "\n"
        "Options:\n"
        "  --absolute        match exact Y values (default: relative relief/shape)\n"
        "  --tol N           per-column tolerance in blocks (default 1)\n"
        "  --orient N        0..7 single orientation, or -1 for all 8 (default -1)\n"
        "  --threads N       OpenMP threads (default: all cores)\n"
        "  --max N           stop after N matches (default: unlimited)\n"
        "\n"
        "Example (relative shape, +-1 block):\n"
        "  %s -377264746167088810 wgspec 0 0 5000 -8,0:73 -4,0:77 0,0:75 4,0:71 8,0:69\n",
        prog, prog);
}

int main(int argc, char **argv) {
    if (argc < 7) { usage(argv[0]); return 1; }

    u64  seed       = (u64) strtoll(argv[1], NULL, 10);
    const char *spec= argv[2];
    int  centX      = atoi(argv[3]);
    int  centZ      = atoi(argv[4]);
    i64  radius     = (i64) strtoll(argv[5], NULL, 10);

    int  absolute   = 0;
    int  tol        = 1;
    int  orientArg  = -1;
    int  nthreads   = omp_get_max_threads();
    long long maxMatches = -1;

    // parse options + pattern
    int i = 6;
    for (; i < argc; i++) {
        if (strcmp(argv[i], "--absolute") == 0) { absolute = 1; }
        else if (strcmp(argv[i], "--tol") == 0)     { tol = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--orient") == 0)  { orientArg = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--threads") == 0) { nthreads = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--max") == 0)     { maxMatches = strtoll(argv[++i], NULL, 10); }
        else {
            // pattern point dx,dz:height
            int dx, dz, h;
            if (sscanf(argv[i], "%d,%d:%d", &dx, &dz, &h) != 3) {
                fprintf(stderr, "bad pattern point '%s' (want dx,dz:height)\n", argv[i]);
                return 1;
            }
            if (PATTERN_COUNT >= MAX_PATTERN) { fprintf(stderr, "too many points\n"); return 1; }
            PAT_DX[PATTERN_COUNT] = dx;
            PAT_DZ[PATTERN_COUNT] = dz;
            PAT_H [PATTERN_COUNT] = h;
            PATTERN_COUNT++;
        }
    }
    if (PATTERN_COUNT < 2) { fprintf(stderr, "need at least 2 pattern points\n"); return 1; }

    omp_set_num_threads(nthreads);

    // The anchor for relative mode is pattern point 0; level all heights by it.
    int anchorH = PAT_H[0];

    // precompute transformed offsets
    int tStart = (orientArg < 0) ? 0 : orientArg;
    int tEnd   = (orientArg < 0) ? 7 : orientArg;
    for (int t = 0; t < 8; t++)
        for (int p = 0; p < PATTERN_COUNT; p++)
            applyTransformXZ(t, PAT_DX[p], PAT_DZ[p], &TPAT_DX[t][p], &TPAT_DZ[t][p]);

    // pattern reach (max |offset|) -> tile halo so a match centred at a tile
    // edge can still read all its offsets.
    int reach = 0;
    for (int p = 0; p < PATTERN_COUNT; p++) {
        int a = abs(PAT_DX[p]), b = abs(PAT_DZ[p]);
        if (a > reach) reach = a;
        if (b > reach) reach = b;
    }

    printf("Surface-height pattern matcher (C)\n");
    printf("Seed: %lld  Center: (%d,%d)  Radius: %lld blocks\n",
           (long long)seed, centX, centZ, radius);
    printf("Pattern: %d points  mode: %s  tol: +-%d  orient: %d..%d  reach: %d\n",
           PATTERN_COUNT, absolute ? "absolute" : "relative", tol, tStart, tEnd, reach);
    printf("Threads: %d\n\n", nthreads);
    fflush(stdout);

    // ---- build center-out tiles ----
    const int TILE = 256; // blocks per tile side
    i64 rSq = radius * radius;
    int nTilesSide = (int)((2 * radius) / TILE + 2);
    long long maxTiles = (long long)(nTilesSide + 2) * (nTilesSide + 2);
    Tile *tiles = malloc(sizeof(Tile) * maxTiles);
    int nTiles = 0;
    for (i64 tz = -radius; tz <= radius; tz += TILE) {
        for (i64 tx = -radius; tx <= radius; tx += TILE) {
            i64 x0 = centX + tx, z0 = centZ + tz;
            i64 x1 = x0 + TILE - 1, z1 = z0 + TILE - 1;
            if (x1 > centX + radius) x1 = centX + radius;
            if (z1 > centZ + radius) z1 = centZ + radius;
            if (x0 > x1 || z0 > z1) continue;
            // nearest-point distance from center to tile box
            i64 dx = (centX < x0) ? (x0 - centX) : (centX > x1 ? centX - x1 : 0);
            i64 dz = (centZ < z0) ? (z0 - centZ) : (centZ > z1 ? centZ - z1 : 0);
            if (dx*dx + dz*dz > rSq) continue;
            double mx = (x0 + x1 + 1) * 0.5 - centX;
            double mz = (z0 + z1 + 1) * 0.5 - centZ;
            tiles[nTiles].x0 = (int)x0; tiles[nTiles].z0 = (int)z0;
            tiles[nTiles].x1 = (int)x1; tiles[nTiles].z1 = (int)z1;
            tiles[nTiles].distSq = mx*mx + mz*mz;
            nTiles++;
        }
    }
    qsort(tiles, nTiles, sizeof(Tile), cmpTile);
    printf("Tiles: %d (%dx%d blocks each)\n\n", nTiles, TILE, TILE);
    fflush(stdout);

    omp_init_lock(&g_matchLock);
    g_matchFile = fopen("matches.txt", "w");
    if (g_matchFile) {
        fprintf(g_matchFile, "--- HEIGHT MATCH LOG ---\n");
        fprintf(g_matchFile, "Seed: %lld  Center: (%d,%d)  Radius: %lld\n",
                (long long)seed, centX, centZ, (long long)radius);
        fprintf(g_matchFile, "Mode: %s  tol: +-%d\n\n", absolute ? "absolute":"relative", tol);
        fflush(g_matchFile);
    }

    int processed = 0;
    double t0 = omp_get_wtime();
    volatile int stop = 0;

    #pragma omp parallel
    {
        // Per-thread engine instance (Generator + HeightGen).
        Generator g;
        setupGenerator(&g, MC_1_21_3, 0);
        applySeed(&g, DIM_OVERWORLD, seed);
        HeightGen hg;
        if (heightGenInit(&hg, &g, spec) != 0) {
            #pragma omp critical
            fprintf(stderr, "heightGenInit failed (thread %d)\n", omp_get_thread_num());
        }

        // Per-tile height cache: covers tile core + reach halo.
        int cacheW = TILE + 2 * reach;
        int *hcache = malloc(sizeof(int) * (size_t)cacheW * cacheW);

        #pragma omp for schedule(dynamic, 1)
        for (int ti = 0; ti < nTiles; ti++) {
            if (stop) continue;
            Tile tile = tiles[ti];
            int bx0 = tile.x0 - reach, bz0 = tile.z0 - reach;
            int cw = (tile.x1 - tile.x0 + 1) + 2 * reach;
            int ch = (tile.z1 - tile.z0 + 1) + 2 * reach;

            // fill cache via the shared-lattice region pass (far faster than a
            // per-column heightAt loop; bit-identical results). The cache row
            // stride is cacheW but the filled width is cw, so fill row by row
            // into a packed temp then scatter, OR fill directly when cw==cacheW.
            if (cw == cacheW) {
                heightRegion(&hg, bx0, bz0, cw, ch, hcache);
            } else {
                int *tmp = malloc(sizeof(int) * (size_t)cw * ch);
                heightRegion(&hg, bx0, bz0, cw, ch, tmp);
                for (int lz = 0; lz < ch; lz++)
                    for (int lx = 0; lx < cw; lx++)
                        hcache[lz * cacheW + lx] = tmp[lz * cw + lx];
                free(tmp);
            }

            // test every candidate center in the tile core
            for (int cz = tile.z0; cz <= tile.z1; cz++) {
                i64 ddz = (i64)cz - centZ;
                for (int cx = tile.x0; cx <= tile.x1; cx++) {
                    i64 ddx = (i64)cx - centX;
                    if (ddx*ddx + ddz*ddz > rSq) continue;

                    for (int t = tStart; t <= tEnd; t++) {
                        // anchor height of this candidate (offset 0 of pattern,
                        // transformed). Pattern point 0's transformed offset:
                        int a0x = cx + TPAT_DX[t][0];
                        int a0z = cz + TPAT_DZ[t][0];
                        int candAnchor = hcache[(a0z - bz0) * cacheW + (a0x - bx0)];

                        int ok = 1;
                        for (int p = 0; p < PATTERN_COUNT; p++) {
                            int hx = cx + TPAT_DX[t][p];
                            int hz = cz + TPAT_DZ[t][p];
                            int hv = hcache[(hz - bz0) * cacheW + (hx - bx0)];
                            int want = PAT_H[p];
                            int diff;
                            if (absolute) diff = hv - want;
                            else          diff = (hv - candAnchor) - (want - anchorH);
                            if (diff < -tol || diff > tol) { ok = 0; break; }
                        }
                        if (ok) {
                            omp_set_lock(&g_matchLock);
                            g_matchCount++;
                            if (g_matchFile) {
                                fprintf(g_matchFile, "MATCH (%d,%d) orient=%d anchorY=%d\n",
                                        cx, cz, t, candAnchor);
                                fflush(g_matchFile);
                            }
                            if (maxMatches > 0 && g_matchCount >= maxMatches) stop = 1;
                            omp_unset_lock(&g_matchLock);
                        }
                    }
                }
            }

            #pragma omp atomic
            processed++;
            if ((processed & 31) == 0) {
                #pragma omp critical
                {
                    double el = omp_get_wtime() - t0;
                    printf("\r  tiles %d/%d  matches %lld  %.0fs   ",
                           processed, nTiles, g_matchCount, el);
                    fflush(stdout);
                }
            }
        }
        free(hcache);
    }

    double el = omp_get_wtime() - t0;
    printf("\n\nDone. %lld matches in %.1fs. Written to matches.txt\n",
           g_matchCount, el);
    if (g_matchFile) fclose(g_matchFile);
    omp_destroy_lock(&g_matchLock);
    free(tiles);
    return 0;
}
