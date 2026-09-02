// ore_match_gpu.cu -- CUDA ore pattern finder for Minecraft 1.21
// Overlapping-tile, dense-grid, scatter approach:
//   generateTile: 1 thread per SEEDING chunk, scatters each vein's blocks into
//                 whichever chunk grid they land in (cross-border bleed included)
//   matchTile:    1 thread per (anchor chunk, interior cell); scans the anchor
//                 chunk's dense grid and tests each pattern offset by direct read
// The search region is tiled in BOTH X and Z into fixed TILE_COLS x TILE_ROWS
// chunk blocks, so resident memory is bounded by the stripe budget regardless of
// search radius (a tile no longer spans the full search width). Tiles overlap
// neighbours by 2*margin chunks in each dimension so no vein bleed or pattern
// lookup is lost across a tile boundary.
// Compile (Linux):  nvcc -O3 -arch=sm_89 -o ore_match_gpu ore_match_gpu.cu -lm
// Compile (Windows): nvcc -O3 -arch=sm_89 -o ore_match_gpu.exe ore_match_gpu.cu

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <chrono>

// Phase timing (host wall-clock around each phase's terminating sync)
static double g_genSeconds = 0.0;
static double g_matchSeconds = 0.0;

// Match output file. The binary owns matches.txt; the GUI tails it for display.
static FILE *g_matchFile = NULL;
static double wallSeconds() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// ---- Basic types ----
typedef uint64_t u64;
typedef int64_t  i64;
typedef uint32_t u32;
typedef int32_t  i32;

#define PI_D 3.14159265358979323846
#define PI_F 3.14159265358979323846f

// Maximum ore size (copper is largest at 20)
#define MAX_ORE_SIZE 20

// Maximum pattern offsets
#define MAX_PATTERN 64

// Maximum distinct ore types in one pattern (per-chunk dense grid per slot)
#define MAX_ORE_SLOTS 7

// Maximum results to collect per match kernel launch (drained after each tile).
#define MAX_RESULTS 2000000

// ---- Ore type constants (same as CPU) ----
#define ORE_IRON     0
#define ORE_COAL     1
#define ORE_COPPER   2
#define ORE_GOLD     3
#define ORE_REDSTONE 4
#define ORE_LAPIS    5
#define ORE_DIAMOND  6

// ---- Result struct ----
struct MatchResult {
    i32 x, y, z;
    i32 transform;
    i32 score;
};

// ---- Device-side sin table (matches CPU float->double promotion) ----
// SIN_TABLE_D[sz][i] = sin(PI * (float)i/(float)sz)   for sz in [1,20], i in [0,sz)
__constant__ double SIN_TABLE_D[21][21];

// ---- Xoroshiro128++ (device) ----
struct Xoro { u64 lo, hi; };

__device__ __forceinline__ u64 rotl64(u64 x, int b) {
    return (x << b) | (x >> (64 - b));
}

__device__ __forceinline__ void xSetSeed(Xoro *xr, u64 v) {
    const u64 XL = 0x9e3779b97f4a7c15ULL, XH = 0x6a09e667f3bcc909ULL;
    const u64 A  = 0xbf58476d1ce4e5b9ULL, B  = 0x94d049bb133111ebULL;
    u64 l = v ^ XH, h = l + XL;
    l = (l ^ (l >> 30)) * A; h = (h ^ (h >> 30)) * A;
    l = (l ^ (l >> 27)) * B; h = (h ^ (h >> 27)) * B;
    xr->lo = l ^ (l >> 31); xr->hi = h ^ (h >> 31);
}

__device__ __forceinline__ u64 xNext(Xoro *xr) {
    u64 l = xr->lo, h = xr->hi;
    u64 n = rotl64(l + h, 17) + l;
    h ^= l;
    xr->lo = rotl64(l, 49) ^ h ^ (h << 21);
    xr->hi = rotl64(h, 28);
    return n;
}

__device__ __forceinline__ i32 next31(Xoro *xr) {
    return (i32)(xNext(xr) >> 33);
}

__device__ __forceinline__ i32 xNextInt(Xoro *xr, u32 n) {
    i32 bits, val;
    u32 m = n - 1;
    if ((m & n) == 0)
        return (i32)(((i64)next31(xr) * (i64)n) >> 31);
    do {
        bits = next31(xr);
        val  = bits % (i32)n;
    } while ((i32)((u32)bits - (u32)val + m) < 0);
    return val;
}

__device__ __forceinline__ float xNextFloat(Xoro *xr) {
    return (i32)(xNext(xr) >> (64 - 24)) * (1.0f / (1 << 24));
}

__device__ __forceinline__ double xNextDouble(Xoro *xr) {
    i64 a = (i32)(xNext(xr) >> (64 - 26));
    i64 b = (i32)(xNext(xr) >> (64 - 27));
    return ((a << 27) + b) * (1.0 / (1ULL << 53));
}

__device__ __forceinline__ u64 xNextLongJ(Xoro *xr) {
    i32 a = (i32)(xNext(xr) >> 32);
    i32 b = (i32)(xNext(xr) >> 32);
    return (u64)(((i64)a << 32) + (i64)b);
}

// ---- Population seed ----
__device__ __forceinline__ u64 populationSeed(u64 worldSeed, u64 POP_A, u64 POP_B,
                                               i32 blockX, i32 blockZ) {
    return (u64)blockX * POP_A + (u64)blockZ * POP_B ^ worldSeed;
}

// ---- Height providers ----
__device__ __forceinline__ i32 heightTrapezoid(Xoro *xr, i32 minH, i32 inner, i32 shoulder) {
    return minH + xNextInt(xr, (u32)(inner + 1)) + xNextInt(xr, (u32)(shoulder + 1));
}

__device__ __forceinline__ i32 heightUniform(Xoro *xr, i32 minH, i32 maxH) {
    return minH + xNextInt(xr, (u32)(maxH - minH + 1));
}

// Specific height functions matching CPU exactly
__device__ __forceinline__ i32 heightIronMiddle(Xoro *xr)    { return -24 + xNextInt(xr,41) + xNextInt(xr,41); }
__device__ __forceinline__ i32 heightIronSmall(Xoro *xr)     { return -64 + xNextInt(xr,137); }
__device__ __forceinline__ i32 heightCoalUpper(Xoro *xr)     { return xNextInt(xr,193); }
__device__ __forceinline__ i32 heightCoalLower(Xoro *xr)     { return xNextInt(xr,97) + xNextInt(xr,97); }
__device__ __forceinline__ i32 heightGold(Xoro *xr)          { return -64 + xNextInt(xr,49) + xNextInt(xr,49); }
__device__ __forceinline__ i32 heightGoldLower(Xoro *xr)     { return -64 + xNextInt(xr,9) + xNextInt(xr,9); }
__device__ __forceinline__ i32 heightRedstone(Xoro *xr)      { return -64 + xNextInt(xr,80); }
__device__ __forceinline__ i32 heightRedstoneLower(Xoro *xr) { return -32 + xNextInt(xr,5) + xNextInt(xr,5); }
__device__ __forceinline__ i32 heightDiamond(Xoro *xr)       { return -64 + xNextInt(xr,31) + xNextInt(xr,31); }
__device__ __forceinline__ i32 heightLapis(Xoro *xr)         { return -32 + xNextInt(xr,33) + xNextInt(xr,33); }
__device__ __forceinline__ i32 heightCopper(Xoro *xr)        { return -16 + xNextInt(xr,65) + xNextInt(xr,65); }

// ---- Per-chunk dense occupancy bit-grid ----
// Each chunk+ore is a dense 16x384x16 bitset (one bit per block cell), one bit
// array per active ore type. This replaces the old per-chunk hash table.
//
// Why dense instead of a hash table: it lets the generate kernel SCATTER bled
// blocks directly into whatever chunk they physically land in. A vein near a
// chunk border writes some cells in this chunk and some in its neighbours; with
// a shared grid those cross-border writes just land in the neighbour's bits.
// The old hash table required either clipping the vein (the bleed bug: ~17%
// missed matches) or per-insert atomicCAS with probing. The dense grid drops
// the probing entirely: a write is a single direct-indexed set, and a lookup is
// a single direct-indexed test. We still need a light atomicOr on the bit word
// (32 distinct cells share one u32, so two threads setting different bits in the
// same word must not lose an update) -- but that is far cheaper than the hash
// CAS loop and contends only on shared words, which is rare for sparse veins.
// Cost: ~12 KB/chunk/ore vs ~1 KB, which the per-stripe sizing bounds.
//
// Y-CLAMPED layout: each ore stores only its real vertical band, not the full
// world Y range [-64,319]. Iron lives in ~Y[-72,80], diamond ~[-72,4], lapis
// ~[-40,40], etc., so a per-ore band cuts grid memory (and the per-tile zeroing
// + memory traffic) by ~4x. Cell ordering within an ore's band:
//   bitIdx = (lx * ySpan + (Y - yLo)) * 16 + lz
// Each ore SLOT s has its own yLo/ySpan/words; a chunk's per-slot grids are
// concatenated with cumulative offset slotOff[s], total = chunkStride words.
// These tables are uploaded once to __constant__ (indexed by ore SLOT, not ORE_*).
__constant__ i32 SLOT_Y_LO[MAX_ORE_SLOTS];   // world Y of band row 0 for slot s
__constant__ i32 SLOT_Y_SPAN[MAX_ORE_SLOTS]; // band height in blocks for slot s
__constant__ i32 SLOT_OFF[MAX_ORE_SLOTS];    // word offset of slot s in a chunk
__constant__ i32 SLOT_WORDS[MAX_ORE_SLOTS];  // band words for slot s
__constant__ i32 CHUNK_STRIDE_WORDS;         // total words per chunk (all slots)

__device__ __forceinline__ i32 cellIndexS(i32 slot, i32 localX, i32 Y, i32 localZ) {
    return (localX * SLOT_Y_SPAN[slot] + (Y - SLOT_Y_LO[slot])) * 16 + localZ;
}

// Set a cell bit in ore-slot s's band grid (grid points at the slot grid base).
// atomicOr guards lost updates on the 32 cells sharing a u32 word; same-value
// races are benign.
__device__ __forceinline__ void gridSetS(u32 *grid, i32 slot, i32 localX, i32 Y, i32 localZ) {
    i32 idx = cellIndexS(slot, localX, Y, localZ);
    atomicOr(&grid[idx >> 5], 1u << (idx & 31));
}

// Test a cell bit in ore-slot s's band grid.
__device__ __forceinline__ bool gridGetS(const u32 *grid, i32 slot, i32 localX, i32 Y, i32 localZ) {
    i32 idx = cellIndexS(slot, localX, Y, localZ);
    return (grid[idx >> 5] >> (idx & 31)) & 1u;
}

// Generate one ore vein and insert blocks into the chunk's hash map.
// chunkBX, chunkBZ: world block coordinates of the chunk origin (cx*16, cz*16)
// baseX, baseY, baseZ: vein origin in world block coordinates
// size: vein size
//
// SCATTER model: this vein is seeded in some chunk, but its blocks may bleed
// into neighbouring chunks. We compute the true world (X,Y,Z) of each block and
// set the corresponding bit in WHICHEVER resident chunk it lands in. Blocks that
// fall outside the resident grid band (off the stripe edge, or off the X range)
// are dropped here -- the adjacent stripe regenerates that seeding chunk and
// scatters them correctly (stripes overlap by 1 row so nothing is lost).
//
// gridBase:    base of the tile grid (row 0, chunk 0, slot 0).
//              Chunk (cxIdx, zRowIdx) ore-slot s grid is at
//              gridBase + ((u64)zRowIdx*numChunksX + cxIdx)*chunkStride + slotOff
// slot:        ore SLOT index (selects the Y band via SLOT_Y_LO/SLOT_Y_SPAN).
// slotOff:     word offset of this slot's grid within a chunk.
// cxMinRow:    world chunk X of grid column 0.
// numChunksX:  grid width in chunks.
// czBandMin:   world chunk Z of grid row 0.
// bandRows:    number of chunk rows resident in the grid band.
// chunkStride: u32 words per chunk (sum of all slot band words).
__device__ void generateVeinGPU(
    u32 *gridBase,
    i32 cxMinRow, i32 numChunksX,
    i32 czBandMin, i32 bandRows,
    i32 chunkStride, i32 slot, i32 slotOff,
    Xoro *xr,
    i32 baseX, i32 baseY, i32 baseZ,
    i32 size
) {
    // angle -- float, matching CPU
    float angle = xNextFloat(xr) * PI_F;
    float szf   = (float)size / 8.0f;

    double s = sin((double)angle);
    double c = cos((double)angle);

    double oxp = baseX + s * (double)szf;
    double oxn = baseX - s * (double)szf;
    double ozp = baseZ + c * (double)szf;
    double ozn = baseZ - c * (double)szf;
    double oyp = baseY + xNextInt(xr, 3) - 2;
    double oyn = baseY + xNextInt(xr, 3) - 2;

    double sz_div_16 = (double)size / 16.0;

    // Store sphere params (max size 20 for copper)
    double sx_arr[MAX_ORE_SIZE], sy_arr[MAX_ORE_SIZE];
    double sz_arr[MAX_ORE_SIZE], off_arr[MAX_ORE_SIZE];

    for (i32 i = 0; i < size; i++) {
        double len = xNextDouble(xr) * sz_div_16;
        // SIN_TABLE_D[size][i] matches CPU's SIN_TABLE[size][i]
        double sinv = SIN_TABLE_D[size][i];
        double off  = ((sinv + 1.0) * len + 1.0) * 0.5;
        double pct  = (double)(float)((float)i / (float)size); // float pct as in CPU
        double x    = oxp + pct * (oxn - oxp);
        double y    = oyp + pct * (oyn - oyp);
        double z    = ozp + pct * (ozn - ozp);
        sx_arr[i]   = x;
        sy_arr[i]   = y;
        sz_arr[i]   = z;
        off_arr[i]  = off;
    }

    // Containment pruning
    for (i32 i = 0; i < size - 1; i++) {
        if (off_arr[i] <= 0.0) continue;
        for (i32 j = i + 1; j < size; j++) {
            if (off_arr[j] <= 0.0) continue;
            double ddx  = sx_arr[i] - sx_arr[j];
            double ddy  = sy_arr[i] - sy_arr[j];
            double ddz  = sz_arr[i] - sz_arr[j];
            double doff = off_arr[i] - off_arr[j];
            if (doff * doff <= ddx * ddx + ddy * ddy + ddz * ddz) continue;
            if (doff > 0.0) off_arr[j] = -1.0; else off_arr[i] = -1.0;
        }
    }

    // World-block extent of the resident grid band, used to drop out-of-band
    // bleed cheaply (those cells belong to an adjacent stripe).
    i32 bandMinX = cxMinRow * 16;
    i32 bandMaxX = (cxMinRow + numChunksX) * 16 - 1;
    i32 bandMinZ = czBandMin * 16;
    i32 bandMaxZ = (czBandMin + bandRows) * 16 - 1;

    for (i32 i = 0; i < size; i++) {
        double off = off_arr[i];
        if (off <= 0.0) continue;

        double cx2 = sx_arr[i], cy2 = sy_arr[i], cz2 = sz_arr[i];
        i32 minX = (i32)floor(cx2 - off);
        i32 maxX = (i32)floor(cx2 + off);
        i32 minY = (i32)floor(cy2 - off);
        i32 maxY = (i32)floor(cy2 + off);
        i32 minZ = (i32)floor(cz2 - off);
        i32 maxZ = (i32)floor(cz2 + off);

        // Clamp Y to this ore slot's band (cells outside the band are never
        // stored; matchTile only ever looks up within the same band, so dropping
        // them is exact for this ore).
        i32 yLo = SLOT_Y_LO[slot];
        i32 yHi = yLo + SLOT_Y_SPAN[slot] - 1;
        if (minY < yLo) minY = yLo;
        if (maxY > yHi) maxY = yHi;

        // Clamp X/Z to the resident grid band (drop out-of-band bleed).
        if (minX < bandMinX) minX = bandMinX;
        if (maxX > bandMaxX) maxX = bandMaxX;
        if (minZ < bandMinZ) minZ = bandMinZ;
        if (maxZ > bandMaxZ) maxZ = bandMaxZ;

        if (minX > maxX || minZ > maxZ) continue;

        double inv_off = 1.0 / off;

        for (i32 X = minX; X <= maxX; X++) {
            double xsl  = ((double)X + 0.5 - cx2) * inv_off;
            double xsl2 = xsl * xsl;
            if (xsl2 >= 1.0) continue;

            double r2_xz = 1.0 - xsl2;
            i32 cxIdx = (X >> 4) - cxMinRow; // grid column
            i32 lx    = X & 15;

            for (i32 Y = minY; Y <= maxY; Y++) {
                double ysl  = ((double)Y + 0.5 - cy2) * inv_off;
                double ysl2 = ysl * ysl;
                double r2_xy = r2_xz - ysl2;
                if (r2_xy <= 0.0) continue;

                for (i32 Z = minZ; Z <= maxZ; Z++) {
                    double zsl  = ((double)Z + 0.5 - cz2) * inv_off;
                    double zsl2 = zsl * zsl;
                    if (xsl2 + ysl2 + zsl2 >= 1.0) continue;

                    i32 zRowIdx = (Z >> 4) - czBandMin; // grid row
                    i32 lz      = Z & 15;

                    // Scatter into the chunk+slot grid this block lands in.
                    u32 *chunkGrid = gridBase
                        + ((u64)zRowIdx * numChunksX + cxIdx) * chunkStride
                        + slotOff;
                    gridSetS(chunkGrid, slot, lx, Y, lz);
                }
            }
        }
    }
}

// ---- Generate kernel: 1 thread per SEEDING chunk in the tile ----
// Each thread rolls the veins seeded in its chunk and SCATTERS the resulting
// blocks into whichever resident chunk grid they land in (its own, or a
// neighbour's via cross-border bleed). The whole tile grid must be zeroed by the
// caller first (we OR bits in). The host calls this once per active ore type.
//
// gridBase:   base of the tile grid (row 0, chunk 0, slot 0).
// cxBase:     world chunk X of grid column 0.
// czBandMin:  world chunk Z of grid row 0.
// tileRows:   number of seeding chunk rows in this tile (= bandRows).
// numChunksX: tile width in chunks.
// chunkStride: u32 words per chunk (sum of all slot band words).
// slot/slotOff: ore slot index + its word offset within a chunk.
__global__ void __launch_bounds__(256, 4) generateTile(
    u32 *gridBase,
    u64  worldSeed,
    u64  POP_A,
    u64  POP_B,
    i32  czBandMin,
    i32  cxBase,
    i32  numChunksX,
    i32  tileRows,
    i32  chunkStride,
    i32  slot,
    i32  slotOff,
    i32  oreType
) {
    i32 tid = (i32)(blockIdx.x * blockDim.x + threadIdx.x);
    i32 total = numChunksX * tileRows;
    if (tid >= total) return;

    i32 col = tid % numChunksX;
    i32 row = tid / numChunksX;
    i32 cx = cxBase + col;
    i32 cz = czBandMin + row;
    i32 blockX = cx << 4;
    i32 blockZ = cz << 4;

    u64 popSeed = populationSeed(worldSeed, POP_A, POP_B, blockX, blockZ);

#define GEN_VEIN(idx, count, size, heightCallExpr) \
    { Xoro xr; xSetSeed(&xr, popSeed + (idx) + 10000ULL * 6); \
      for (i32 _i = 0; _i < (count); _i++) { \
          i32 bx = blockX + xNextInt(&xr, 16); \
          i32 bz = blockZ + xNextInt(&xr, 16); \
          i32 by = (heightCallExpr); \
          generateVeinGPU(gridBase, cxBase, numChunksX, czBandMin, tileRows, \
                          chunkStride, slot, slotOff, &xr, bx, by, bz, (size)); \
      } }

    switch (oreType) {
        case ORE_IRON:
            GEN_VEIN(12, 10, 9,  heightIronMiddle(&xr))
            GEN_VEIN(13, 10, 4,  heightIronSmall(&xr))
            break;
        case ORE_COAL:
            GEN_VEIN( 9, 20, 17, heightCoalUpper(&xr))
            GEN_VEIN(10, 20, 17, heightCoalLower(&xr))
            break;
        case ORE_COPPER:
            GEN_VEIN(24, 16, 20, heightCopper(&xr))
            break;
        case ORE_GOLD:
            GEN_VEIN(14, 4, 9,   heightGold(&xr))
            GEN_VEIN(15, 4, 9,   heightGoldLower(&xr))
            break;
        case ORE_REDSTONE:
            GEN_VEIN(16, 8, 8,   heightRedstone(&xr))
            GEN_VEIN(17, 8, 8,   heightRedstoneLower(&xr))
            break;
        case ORE_LAPIS:
            GEN_VEIN(22, 2, 7,   heightLapis(&xr))
            GEN_VEIN(23, 4, 7,   heightLapis(&xr))
            break;
        case ORE_DIAMOND:
            GEN_VEIN(18, 7, 8,   heightDiamond(&xr))
            GEN_VEIN(19, 2, 8,   heightDiamond(&xr))
            GEN_VEIN(20, 1, 12,  heightDiamond(&xr))
            GEN_VEIN(21, 4, 4,   heightDiamond(&xr))
            break;
    }

#undef GEN_VEIN
}

// ---- Match kernel: 1 thread per (anchor chunk col, interior anchor row) ----
// Grid layout: chunk (col, zRowIdx), ore slot s grid is at
//   grid + ((u64)zRowIdx*numChunksX + col)*CHUNK_STRIDE_WORDS + SLOT_OFF[s]
// Anchor candidates are the SET cells of the anchor ore's band for this chunk;
// each pattern offset p is tested by a direct-indexed bit read in its ore's band.
__global__ void matchTile(
    u32 *grid,            // [tileRows * numChunksX * CHUNK_STRIDE_WORDS]
    i32  tileRows,
    i32  cxBase,
    i32  czBandMin,
    i32  matchRowStart,   // first interior anchor row index (in grid rows)
    i32  matchRowEnd,     // one past last interior anchor row index
    i32  matchColStart,   // first interior anchor col index (in grid cols)
    i32  matchColEnd,     // one past last interior anchor col index
    i32  numChunksX,
    i32  anchorOreSlot,
    i64  rSq,
    i32  centX, i32  centZ,
    i32  tStart, i32  tEnd,
    i32 *patterns,        // [8][patCount][3]
    i32 *patOre,          // [patCount] -> ore slot per offset
    i32  patCount,
    i32  minScore,
    MatchResult *results,
    i32 *resultCount
) {
    i32 tid = (i32)(blockIdx.x * blockDim.x + threadIdx.x);
    i32 numMatchRows = matchRowEnd - matchRowStart;
    i32 numMatchCols = matchColEnd - matchColStart;
    i32 total = numMatchCols * numMatchRows;
    if (tid >= total) return;

    i32 col     = matchColStart + tid % numMatchCols;
    i32 zRowIdx = matchRowStart + tid / numMatchCols;
    i32 cx = cxBase + col;
    i32 cz = czBandMin + zRowIdx;

    // Quick chunk-level radius cull: skip only if the WHOLE chunk is outside the
    // radius. Use the chunk's nearest XZ point to the centre (clamp centre to the
    // chunk's [x0,x0+15] x [z0,z0+15] block extent), not the chunk centre -- a
    // chunk whose centre is just outside the radius can still hold in-radius
    // blocks at its near corner (this previously dropped edge matches).
    i32 x0 = cx * 16, z0 = cz * 16;
    i64 nx = centX < x0 ? x0 : (centX > x0 + 15 ? x0 + 15 : centX);
    i64 nz = centZ < z0 ? z0 : (centZ > z0 + 15 ? z0 + 15 : centZ);
    i64 ndx = nx - centX, ndz = nz - centZ;
    if (ndx * ndx + ndz * ndz > rSq) return;

    i32 stride      = CHUNK_STRIDE_WORDS;
    u64 chunkBase   = ((u64)zRowIdx * numChunksX + col) * stride;
    const u32 *anchorGrid = grid + chunkBase + (u64)SLOT_OFF[anchorOreSlot];

    // Anchor band geometry for decoding cell indices.
    i32 aYLo   = SLOT_Y_LO[anchorOreSlot];
    i32 aYSpan = SLOT_Y_SPAN[anchorOreSlot];
    i32 aWords = SLOT_WORDS[anchorOreSlot];

    // Scan the anchor chunk's band for set cells (candidate anchor blocks).
    for (i32 w = 0; w < aWords; w++) {
        u32 word = anchorGrid[w];
        while (word) {
            i32 b   = __ffs(word) - 1;      // lowest set bit
            word   &= word - 1;             // clear it
            i32 idx = (w << 5) + b;         // cell index within the band
            // Decode cell: idx = (lx*aYSpan + (Y-aYLo))*16 + lz
            i32 lz = idx & 15;
            i32 t2 = idx >> 4;              // = lx*aYSpan + (Y-aYLo)
            i32 Y  = (t2 % aYSpan) + aYLo;
            i32 lx = t2 / aYSpan;

            i32 wx = cx * 16 + lx;
            i32 wz = cz * 16 + lz;

            // Precise radius check on this block
            i64 dx = wx - centX, dz = wz - centZ;
            if (dx * dx + dz * dz > rSq) continue;

            for (i32 t = tStart; t <= tEnd; t++) {
                i32 score = 0;
                i32 maxMiss = patCount - minScore;
                i32 miss = 0;

                for (i32 p = 0; p < patCount; p++) {
                    i32 odx = patterns[t * patCount * 3 + p * 3 + 0];
                    i32 ody = patterns[t * patCount * 3 + p * 3 + 1];
                    i32 odz = patterns[t * patCount * 3 + p * 3 + 2];
                    i32 oslot = patOre[p];

                    i32 tx2 = wx + odx;
                    i32 ty2 = Y  + ody;
                    i32 tz2 = wz + odz;

                    // Out of the target ore's stored Y band -> cannot be present.
                    i32 oYLo = SLOT_Y_LO[oslot];
                    if (ty2 < oYLo || ty2 >= oYLo + SLOT_Y_SPAN[oslot]) {
                        miss++;
                        if (miss > maxMiss) goto next_transform;
                        continue;
                    }

                    i32 tcx = tx2 >> 4;
                    i32 tcz = tz2 >> 4;
                    i32 tlx = tx2 & 15;
                    i32 tlz = tz2 & 15;

                    i32 tCol  = tcx - cxBase;
                    i32 tRow  = tcz - czBandMin;
                    if (tCol < 0 || tCol >= numChunksX || tRow < 0 || tRow >= tileRows) {
                        miss++;
                        if (miss > maxMiss) goto next_transform;
                        continue;
                    }

                    const u32 *tGrid = grid
                        + ((u64)tRow * numChunksX + tCol) * stride
                        + (u64)SLOT_OFF[oslot];

                    if (gridGetS(tGrid, oslot, tlx, ty2, tlz)) {
                        score++;
                    } else {
                        miss++;
                        if (miss > maxMiss) goto next_transform;
                    }
                }

                if (score >= minScore) {
                    i32 ridx = atomicAdd(resultCount, 1);
                    if (ridx < MAX_RESULTS) {
                        results[ridx].x         = wx;
                        results[ridx].y         = Y;
                        results[ridx].z         = wz;
                        results[ridx].transform = t;
                        results[ridx].score     = score;
                    }
                }
                next_transform:;
            }
        }
    }
}

// ---- Host helpers ----

static void checkCuda(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

static void applyTransform(i32 t, i32 dx, i32 dy, i32 dz,
                           i32 *ox, i32 *oy, i32 *oz) {
    *oy = dy;
    switch (t) {
        case 0: *ox =  dx; *oz =  dz; break;
        case 1: *ox =  dx; *oz = -dz; break;
        case 2: *ox = -dx; *oz =  dz; break;
        case 3: *ox = -dx; *oz = -dz; break;
        case 4: *ox =  dz; *oz =  dx; break;
        case 5: *ox =  dz; *oz = -dx; break;
        case 6: *ox = -dz; *oz =  dx; break;
        case 7: *ox = -dz; *oz = -dx; break;
    }
}

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s <seed> <cx> <cy> <cz> <radiusBlocks> <orient> [--ore TYPE] [--stripe-mb N] <dx,dy,dz[,oretype]>...\n"
        "  orient: -1=all 8, or 0..7\n"
        "  Offsets may carry a per-offset ore type: dx,dy,dz,oretype\n"
        "\n"
        "  Ore types: iron, gold, redstone, lapis, diamond (coal/copper: CPU engine only)\n"
        "\nExample (mixed):\n"
        "  %s -377264746167088810 205 17 -88 1000 -1 --ore iron 0,0,0 5,1,6 7,1,-6,lapis\n",
        prog, prog);
}

// Map an ore name to ORE_* (-1 unknown). coal/copper excluded (overflow 256 slots).
static int parseOreName(const char *s) {
    if (!strcmp(s,"iron"))     return ORE_IRON;
    if (!strcmp(s,"gold"))     return ORE_GOLD;
    if (!strcmp(s,"redstone")) return ORE_REDSTONE;
    if (!strcmp(s,"lapis"))    return ORE_LAPIS;
    if (!strcmp(s,"diamond"))  return ORE_DIAMOND;
    if (!strcmp(s,"coal"))     return ORE_COAL;     // recognized -> clear error later
    if (!strcmp(s,"copper"))   return ORE_COPPER;
    return -1;
}

// Generation parameters per ORE_* type (idx,count,size + which height fn), used to
// dispatch generateTile per ore slot.
static int oreGenCostRank(int t) { // lower = generate first (cheaper); rare = better anchor
    switch (t) { case ORE_LAPIS: return 0; case ORE_GOLD: return 1; case ORE_REDSTONE: return 2;
                 case ORE_DIAMOND: return 3; case ORE_IRON: return 4; default: return 9; }
}

int main(int argc, char **argv) {
    if (argc < 8) { usage(argv[0]); return 1; }

    i64 WORLD_SEED_I = strtoll(argv[1], NULL, 10);
    u64 worldSeed    = (u64)WORLD_SEED_I;
    i32 centX        = atoi(argv[2]);
    i32 centY        = atoi(argv[3]); (void)centY;
    i32 centZ        = atoi(argv[4]);
    i32 radiusBlks   = atoi(argv[5]);
    i32 orientArg    = atoi(argv[6]);

    i32 oreType = ORE_IRON;
    // Default stripe budget. ~1GB is the sweet spot on RTX 4060: as fast as a
    // full-VRAM stripe but with smooth frequent progress and low TDR risk.
    // Override with --stripe-mb (0 = use free VRAM - 512MB).
    i32 stripeMbCap = 1024;

    i32 argOff = 7;
    while (argOff < argc) {
        if (strcmp(argv[argOff], "--ore") == 0) {
            if (argOff + 1 >= argc) { fprintf(stderr, "--ore needs a value\n"); return 1; }
            const char *s = argv[argOff + 1];
            if      (strcmp(s, "iron")     == 0) oreType = ORE_IRON;
            else if (strcmp(s, "coal")     == 0) oreType = ORE_COAL;
            else if (strcmp(s, "copper")   == 0) oreType = ORE_COPPER;
            else if (strcmp(s, "gold")     == 0) oreType = ORE_GOLD;
            else if (strcmp(s, "redstone") == 0) oreType = ORE_REDSTONE;
            else if (strcmp(s, "lapis")    == 0) oreType = ORE_LAPIS;
            else if (strcmp(s, "diamond")  == 0) oreType = ORE_DIAMOND;
            else { fprintf(stderr, "Unknown ore '%s'\n", s); return 1; }
            argOff += 2;
        } else if (strcmp(argv[argOff], "--stripe-mb") == 0) {
            if (argOff + 1 >= argc) { fprintf(stderr, "--stripe-mb needs a value\n"); return 1; }
            stripeMbCap = atoi(argv[argOff + 1]);
            argOff += 2;
        } else {
            break;
        }
    }

    // Parse pattern offsets, each optionally carrying its own ore type
    // (dx,dy,dz or dx,dy,dz,oretype). Offsets without a type use --ore.
    i32 patCount = 0;
    i32 pattern[MAX_PATTERN][3];
    i32 patOreType_h[MAX_PATTERN];     // ORE_* per offset
    for (i32 i = argOff; i < argc && patCount < MAX_PATTERN; i++) {
        i32 dx, dy, dz;
        char extra[64] = "";
        i32 nf = sscanf(argv[i], "%d,%d,%d,%63s", &dx, &dy, &dz, extra);
        if (nf < 3) { fprintf(stderr, "Bad offset '%s'\n", argv[i]); return 1; }
        i32 ot = oreType;
        if (nf == 4) {
            ot = parseOreName(extra);
            if (ot < 0) { fprintf(stderr, "Unknown ore '%s' in offset '%s'\n", extra, argv[i]); return 1; }
        }
        pattern[patCount][0] = dx;
        pattern[patCount][1] = dy;
        pattern[patCount][2] = dz;
        patOreType_h[patCount] = ot;
        patCount++;
    }
    if (patCount == 0) { fprintf(stderr, "No offsets given\n"); return 1; }

    // Build the list of distinct active ore types (slots). coal/copper are
    // disabled on the GPU: their veins are very dense/large and the kernel is
    // work-bound on them (the CPU engine is the supported path). The dense grid
    // itself could store them, so this is a policy guard, not a capacity limit.
    i32 activeOre[MAX_ORE_SLOTS]; i32 numOreSlots = 0;
    i32 patOreSlot_h[MAX_PATTERN];
    for (i32 p = 0; p < patCount; p++) {
        i32 ot = patOreType_h[p];
        if (ot == ORE_COAL || ot == ORE_COPPER) {
            fprintf(stderr,
                "ERROR: '%s' is not supported by the GPU engine (very dense veins,\n"
                "work-bound on GPU). Use the CPU engine (ore_match4.exe).\n",
                ot == ORE_COAL ? "coal" : "copper");
            return 1;
        }
        i32 slot = -1;
        for (i32 s = 0; s < numOreSlots; s++) if (activeOre[s] == ot) { slot = s; break; }
        if (slot < 0) {
            if (numOreSlots >= MAX_ORE_SLOTS) { fprintf(stderr, "Too many distinct ore types\n"); return 1; }
            slot = numOreSlots; activeOre[numOreSlots++] = ot;
        }
        patOreSlot_h[p] = slot;
    }

    // Anchor on the ore of the FIRST offset (the search frame is relative to the
    // anchor block, matching the existing GPU single-ore semantics: the reported
    // centre is the anchor block and every offset is applied from it). The first
    // offset is normally (0,0,0) of the dropdown ore. (Rare-ore re-anchoring like
    // the CPU engine is a future improvement.)
    i32 anchorOreSlot = patOreSlot_h[0];
    (void)oreGenCostRank;

    // Build all 8 transforms of the pattern
    // patterns_h[t * patCount * 3 + p * 3 + (0|1|2)] = (dx,dy,dz)
    i32 *patterns_h = (i32*)malloc(8 * patCount * 3 * sizeof(i32));
    for (i32 t = 0; t < 8; t++) {
        for (i32 p = 0; p < patCount; p++) {
            i32 ox, oy, oz;
            applyTransform(t, pattern[p][0], pattern[p][1], pattern[p][2], &ox, &oy, &oz);
            patterns_h[t * patCount * 3 + p * 3 + 0] = ox;
            patterns_h[t * patCount * 3 + p * 3 + 1] = oy;
            patterns_h[t * patCount * 3 + p * 3 + 2] = oz;
        }
    }

    i32 tStart = (orientArg < 0) ? 0 : orientArg;
    i32 tEnd   = (orientArg < 0) ? 7 : orientArg;

    // Compute pattern reach
    i32 patReachBlocks = 0;
    for (i32 p = 0; p < patCount; p++) {
        i32 r = abs(pattern[p][0]) > abs(pattern[p][2]) ? abs(pattern[p][0]) : abs(pattern[p][2]);
        if (r > patReachBlocks) patReachBlocks = r;
    }
    i32 patReachChunks = (patReachBlocks + 15) >> 4;

    // Vein bleed reach in chunks: a vein seeded in one chunk can place blocks in
    // neighbours. The reach is bounded by the sphere-centre spread (size/8) plus
    // the max sphere offset (< size). For the supported ores (max size 12) this
    // is < 16 blocks = 1 chunk; we derive it from the largest active vein size to
    // stay safe if larger ores are ever enabled.
    i32 maxVeinSize = 0;
    for (i32 s = 0; s < numOreSlots; s++) {
        i32 ms = 0;
        switch (activeOre[s]) {
            case ORE_IRON:     ms = 9;  break;
            case ORE_GOLD:     ms = 9;  break;
            case ORE_REDSTONE: ms = 8;  break;
            case ORE_LAPIS:    ms = 7;  break;
            case ORE_DIAMOND:  ms = 12; break;
            default:           ms = 12; break;
        }
        if (ms > maxVeinSize) maxVeinSize = ms;
    }
    i32 bleedChunks = ((maxVeinSize / 8 + maxVeinSize) + 15) >> 4;
    if (bleedChunks < 1) bleedChunks = 1;

    // Per-tile margin: bleed rows (under-filled at the tile edge) plus pattern
    // reach (target chunks an anchor's offsets may point into). Interior anchor
    // rows are the rows that have both fully resident. Tiles overlap by 2*margin.
    i32 margin     = bleedChunks + patReachChunks;
    i32 haloChunks = margin;          // X/Z padding around the search region

    // ---- Per-ore-slot Y bands (the Y-clamp optimisation) ----
    // Each ore only generates within a known seed-Y range; the vein then bleeds a
    // few blocks in Y. We store only [yLo,yHi] per ore instead of the full
    // [-64,319], shrinking grid memory + the per-tile zeroing/bandwidth ~4x.
    // Seed-Y ranges below come straight from the height functions; we pad by
    // Y_BLEED on each side for vein offset and floor/ceil expansion, then clamp to
    // the world [-64,319]. These must be conservative (too-tight drops blocks).
    const i32 Y_BLEED = 8;
    i32 slotYLo_h[MAX_ORE_SLOTS], slotYSpan_h[MAX_ORE_SLOTS];
    i32 slotOff_h[MAX_ORE_SLOTS], slotWords_h[MAX_ORE_SLOTS];
    i32 chunkStrideWords = 0;
    for (i32 s = 0; s < numOreSlots; s++) {
        i32 lo, hi;
        switch (activeOre[s]) {
            // seed-Y min .. max across that ore's height functions
            case ORE_IRON:     lo = -64; hi =  72; break; // mid -24..56, small -64..72
            case ORE_GOLD:     lo = -64; hi =  32; break; // -64..32, lower -64..-48
            case ORE_REDSTONE: lo = -64; hi =  15; break; // -64..15, lower -32..-24
            case ORE_LAPIS:    lo = -32; hi =  32; break;
            case ORE_DIAMOND:  lo = -64; hi =  -4; break;
            case ORE_COPPER:   lo = -16; hi = 112; break;
            case ORE_COAL:     lo =   0; hi = 192; break;
            default:           lo = -64; hi = 319; break;
        }
        lo -= Y_BLEED; hi += Y_BLEED;
        if (lo < -64) lo = -64;
        if (hi > 319) hi = 319;
        i32 span = hi - lo + 1;
        i32 cells = 16 * span * 16;
        i32 words = (cells + 31) / 32;
        slotYLo_h[s]   = lo;
        slotYSpan_h[s] = span;
        slotOff_h[s]   = chunkStrideWords;
        slotWords_h[s] = words;
        chunkStrideWords += words;
    }
    checkCuda(cudaMemcpyToSymbol(SLOT_Y_LO,   slotYLo_h,   numOreSlots * sizeof(i32)), "SLOT_Y_LO");
    checkCuda(cudaMemcpyToSymbol(SLOT_Y_SPAN, slotYSpan_h, numOreSlots * sizeof(i32)), "SLOT_Y_SPAN");
    checkCuda(cudaMemcpyToSymbol(SLOT_OFF,    slotOff_h,   numOreSlots * sizeof(i32)), "SLOT_OFF");
    checkCuda(cudaMemcpyToSymbol(SLOT_WORDS,  slotWords_h, numOreSlots * sizeof(i32)), "SLOT_WORDS");
    checkCuda(cudaMemcpyToSymbol(CHUNK_STRIDE_WORDS, &chunkStrideWords, sizeof(i32)),  "CHUNK_STRIDE_WORDS");

    // Precompute sin table on host, upload to __constant__
    // Match CPU: float pct = (float)i / (float)sz; SIN_TABLE[sz][i] = sin(PI * (double)pct)
    double sinTableHost[21][21] = {};
    for (i32 sz = 1; sz <= 20; sz++) {
        for (i32 i = 0; i < sz; i++) {
            float pct = (float)i / (float)sz;
            sinTableHost[sz][i] = sin((double)pct * PI_D);
        }
    }
    checkCuda(cudaMemcpyToSymbol(SIN_TABLE_D, sinTableHost, sizeof(sinTableHost)),
              "sinTable upload");

    // Compute POP_A, POP_B
    // Same as CPU initPopSeedMults
    u64 POP_A, POP_B;
    {
        Xoro xr_h;
        // Host-side xSetSeed / xNext (replicate inline)
        const u64 XL = 0x9e3779b97f4a7c15ULL, XH = 0x6a09e667f3bcc909ULL;
        const u64 A  = 0xbf58476d1ce4e5b9ULL, B  = 0x94d049bb133111ebULL;
        u64 v = worldSeed;
        u64 l = v ^ XH, h = l + XL;
        l = (l ^ (l >> 30)) * A; h = (h ^ (h >> 30)) * A;
        l = (l ^ (l >> 27)) * B; h = (h ^ (h >> 27)) * B;
        xr_h.lo = l ^ (l >> 31); xr_h.hi = h ^ (h >> 31);

        auto xn = [&]() -> u64 {
            u64 ll = xr_h.lo, hh = xr_h.hi;
            u64 n  = ((ll + hh) << 17 | (ll + hh) >> 47) + ll;
            hh ^= ll;
            xr_h.lo = (ll << 49 | ll >> 15) ^ hh ^ (hh << 21);
            xr_h.hi = (hh << 28 | hh >> 36);
            return n;
        };
        auto nextLongJ = [&]() -> u64 {
            i32 a = (i32)(xn() >> 32);
            i32 b = (i32)(xn() >> 32);
            return (u64)(((i64)a << 32) + (i64)b);
        };
        POP_A = nextLongJ() | 1ULL;
        POP_B = nextLongJ() | 1ULL;
    }

    // Search dimensions
    i32 cx0         = centX >> 4;
    i32 cz0         = centZ >> 4;
    i32 chunkRadius = (radiusBlks + 15) >> 4;
    i32 cxMin       = cx0 - chunkRadius - haloChunks;
    i32 cxMax       = cx0 + chunkRadius + haloChunks;
    i32 czMin       = cz0 - chunkRadius - haloChunks;
    i32 czMax       = cz0 + chunkRadius + haloChunks;
    i32 numChunksX  = cxMax - cxMin + 1;
    i32 numChunksZ  = czMax - czMin + 1;
    i64 rSq         = (i64)radiusBlks * radiusBlks;

    static const char *ORE_NAMES[] = {
        "iron", "coal", "copper", "gold", "redstone", "lapis", "diamond"
    };
    printf("Ore pattern finder (CUDA)\n");
    printf("Seed: %lld  Centre: (%d,%d,%d)  Radius: %d blocks\n",
           (long long)WORLD_SEED_I, centX, (i32)centY, centZ, radiusBlks);
    printf("Pattern: %d offsets, %d ore type(s)  Orient: %d..%d  Anchor ore: %s\n",
           patCount, numOreSlots, tStart, tEnd, ORE_NAMES[activeOre[anchorOreSlot]]);
    printf("Pattern reach: %d blocks (%d chunks)  Bleed: %d  Margin: %d chunks\n",
           patReachBlocks, patReachChunks, bleedChunks, margin);
    printf("Search area: %d x %d chunks (%d x %d blocks)\n",
           numChunksX, numChunksZ, numChunksX * 16, numChunksZ * 16);
    fflush(stdout);

    // The binary owns matches.txt. The GUI tails it for the match list.
    g_matchFile = fopen("matches.txt", "w");
    if (g_matchFile) {
        fprintf(g_matchFile, "--- MATCH LOG ---\n");
        fprintf(g_matchFile, "Seed: %lld\nCenter: (%d,%d,%d)\nRadius: %d\n\n",
                (long long)WORLD_SEED_I, centX, (i32)centY, centZ, radiusBlks);
        fflush(g_matchFile);
    } else {
        fprintf(stderr, "warning: could not open matches.txt\n");
    }

    // Pattern device buffer
    i32 *d_patterns = nullptr;
    checkCuda(cudaMalloc(&d_patterns, 8 * patCount * 3 * sizeof(i32)), "alloc patterns");
    checkCuda(cudaMemcpy(d_patterns, patterns_h, 8 * patCount * 3 * sizeof(i32),
                         cudaMemcpyHostToDevice), "copy patterns");

    // Per-offset ore slot buffer
    i32 *d_patOre = nullptr;
    checkCuda(cudaMalloc(&d_patOre, patCount * sizeof(i32)), "alloc patOre");
    checkCuda(cudaMemcpy(d_patOre, patOreSlot_h, patCount * sizeof(i32),
                         cudaMemcpyHostToDevice), "copy patOre");

    // Results
    MatchResult *d_results  = nullptr;
    i32         *d_resCount = nullptr;
    checkCuda(cudaMalloc(&d_results,  MAX_RESULTS * sizeof(MatchResult)), "alloc results");
    checkCuda(cudaMalloc(&d_resCount, sizeof(i32)),                       "alloc resCount");
    checkCuda(cudaMemset(d_resCount, 0, sizeof(i32)), "zero resCount");

    MatchResult *h_results  = (MatchResult*)malloc(MAX_RESULTS * sizeof(MatchResult));

    i32 blockSize  = 256;

    i32 bestScore  = 0;
    i32 totalMatches = 0;

    // Interior anchor extents to match (the search region minus the outer halo).
    // Anchor cols run cxMin+margin..cxMax-margin; anchor rows czMin+margin..czMax-margin.
    i32 totalAnchorRows = numChunksZ - 2 * margin;
    i32 totalAnchorCols = numChunksX - 2 * margin;
    if (totalAnchorRows < 1) totalAnchorRows = 0;
    if (totalAnchorCols < 1) totalAnchorCols = 0;

    // TILE sizing (2D). A tile is a TILE_COLS x TILE_ROWS chunk block, generated
    // once (scatter) and matched on its interior [margin, .-margin) cols/rows.
    // Tiles step by the interior size and overlap neighbours by 2*margin so no
    // vein bleed or pattern lookup is lost. Resident memory is
    //   TILE_COLS * TILE_ROWS * chunkStrideWords u32  (Y-clamped),
    // BOUNDED INDEPENDENTLY OF SEARCH RADIUS because both dimensions are tiled.
    // (Previously a tile spanned the full search width, so memory grew linearly
    // with radius and blew past the budget / VRAM at large radii.)
    size_t freeVram = 0, totalVram = 0;
    cudaMemGetInfo(&freeVram, &totalVram);

    size_t bytesPerChunk = (size_t)chunkStrideWords * sizeof(u32);
    size_t hardCap  = (freeVram > 512ULL*1024*1024) ? (freeVram - 512ULL*1024*1024) : 0;
    size_t softBudget = (stripeMbCap > 0 ? (size_t)stripeMbCap : 1024ULL) * 1024ULL * 1024ULL;
    size_t budget = softBudget;
    if (budget > hardCap && hardCap > 0) budget = hardCap;

    // Keep overlap re-gen overhead modest: each interior dim should be >> 2*margin.
    i32 minInterior = (2 * margin * 100) / 15;   // interior >= ~13x(2*margin)
    if (minInterior < 16) minInterior = 16;

    // Aim for a roughly square tile: side = sqrt(budget / bytesPerChunk) chunks.
    // Clamp each side's interior to the actual search extent so small searches
    // still use a single tile, and to minInterior so overlap stays cheap.
    i32 maxChunksPerTile = (bytesPerChunk > 0) ? (i32)(budget / bytesPerChunk) : (1<<30);
    if (maxChunksPerTile < (minInterior + 2*margin) * (minInterior + 2*margin)) {
        // Budget too small for a square min tile; fall back to a thin band that
        // still fits, preferring full rows of the (now bounded) interior width.
        maxChunksPerTile = (minInterior + 2*margin) * (minInterior + 2*margin);
    }
    i32 sideChunks = (i32)floor(sqrt((double)maxChunksPerTile));
    if (sideChunks < 1) sideChunks = 1;

    i32 interiorCols = sideChunks - 2 * margin;
    i32 interiorRows = sideChunks - 2 * margin;
    if (interiorCols < minInterior) interiorCols = minInterior;
    if (interiorRows < minInterior) interiorRows = minInterior;
    if (interiorCols > totalAnchorCols) interiorCols = totalAnchorCols;
    if (interiorRows > totalAnchorRows) interiorRows = totalAnchorRows;
    if (interiorCols < 1) interiorCols = 1;
    if (interiorRows < 1) interiorRows = 1;

    i32 TILE_COLS = interiorCols + 2 * margin;
    i32 TILE_ROWS = interiorRows + 2 * margin;
    size_t tileBytes = (size_t)TILE_COLS * TILE_ROWS * bytesPerChunk;

    i32 numColTiles = (totalAnchorCols + interiorCols - 1) / interiorCols;
    i32 numRowTiles = (totalAnchorRows + interiorRows - 1) / interiorRows;
    if (numColTiles < 1) numColTiles = 1;
    if (numRowTiles < 1) numRowTiles = 1;

    printf("VRAM free: %.0f MB  tile: %.0f MB  (%d x %d interior chunks/tile, "
           "%d x %d tiles)\n",
           (double)freeVram / 1024 / 1024,
           (double)tileBytes / 1024 / 1024,
           interiorCols, interiorRows, numColTiles, numRowTiles);
    fflush(stdout);

    u32 *d_tile = nullptr;
    checkCuda(cudaMalloc(&d_tile, tileBytes), "alloc tile");

    i32 chunkStride = chunkStrideWords;
    i32 processedTiles = 0;
    i32 totalTiles = numColTiles * numRowTiles;

    // Collect results helper
    auto collectResults = [&]() {
        i32 nRes = 0;
        checkCuda(cudaMemcpy(&nRes, d_resCount, sizeof(i32), cudaMemcpyDeviceToHost), "copy resCount");
        if (nRes > MAX_RESULTS) nRes = MAX_RESULTS;
        if (nRes > 0) {
            checkCuda(cudaMemcpy(h_results, d_results, (size_t)nRes * sizeof(MatchResult),
                                 cudaMemcpyDeviceToHost), "copy results");
            for (i32 ri2 = 0; ri2 < nRes; ri2++) {
                MatchResult &res = h_results[ri2];
                printf("MATCH score=%d/%d transform=%d centre=(%d,%d,%d)\n",
                       res.score, patCount, res.transform, res.x, res.y, res.z);
                if (g_matchFile)
                    fprintf(g_matchFile,
                            "MATCH score=%d/%d transform=%d centre=(%d,%d,%d)\n",
                            res.score, patCount, res.transform, res.x, res.y, res.z);
                if (res.score > bestScore) bestScore = res.score;
                totalMatches++;
            }
            fflush(stdout);
            if (g_matchFile) fflush(g_matchFile);
            checkCuda(cudaMemset(d_resCount, 0, sizeof(i32)), "reset resCount");
        }
    };

    // Center-out 1D ordering helper: visit the centre tile index first, then
    // alternate outward, so the closest matches surface first.
    auto buildOrder = [](i32 n, i32 total, i32 interior) {
        i32 *ord = (i32*)malloc((size_t)(n < 1 ? 1 : n) * sizeof(i32));
        i32 center = (total / 2) / (interior < 1 ? 1 : interior);
        i32 k = 0;
        if (n >= 1) ord[k++] = center;
        for (i32 d = 1; k < n; d++) {
            i32 hi = center + d, lo = center - d;
            if (hi < n) ord[k++] = hi;
            if (lo >= 0 && k < n) ord[k++] = lo;
        }
        return ord;
    };
    i32 *rowOrder = buildOrder(numRowTiles, totalAnchorRows, interiorRows);
    i32 *colOrder = buildOrder(numColTiles, totalAnchorCols, interiorCols);

    // 2D tile sweep: centre-out in Z (outer) and X (inner). Each tile generates
    // its bounded TILE_COLS x TILE_ROWS block then matches its interior window.
    for (i32 rIdx = 0; rIdx < numRowTiles; rIdx++) {
        i32 ai = rowOrder[rIdx] * interiorRows;        // first interior anchor row
        i32 rEnd = ai + interiorRows;
        if (rEnd > totalAnchorRows) rEnd = totalAnchorRows;
        i32 rBatch = rEnd - ai;
        if (rBatch <= 0) continue;

        i32 czBandMin = czMin + ai;                    // grid row 0 world chunk Z
        i32 bandRows  = rBatch + 2 * margin;
        if (czBandMin < czMin) czBandMin = czMin;
        i32 bandEndZ = czBandMin + bandRows;
        if (bandEndZ > czMax + 1) bandEndZ = czMax + 1;
        bandRows = bandEndZ - czBandMin;
        if (bandRows > TILE_ROWS) bandRows = TILE_ROWS;

        i32 matchRowStart = (czMin + margin + ai) - czBandMin;
        i32 matchRowEnd   = matchRowStart + rBatch;
        if (matchRowStart < 0) matchRowStart = 0;
        if (matchRowEnd > bandRows) matchRowEnd = bandRows;
        if (matchRowEnd <= matchRowStart) continue;

        for (i32 cIdx = 0; cIdx < numColTiles; cIdx++) {
            i32 aj = colOrder[cIdx] * interiorCols;    // first interior anchor col
            i32 cEnd = aj + interiorCols;
            if (cEnd > totalAnchorCols) cEnd = totalAnchorCols;
            i32 cBatch = cEnd - aj;
            if (cBatch <= 0) continue;

            i32 cxBandMin = cxMin + aj;                // grid col 0 world chunk X
            i32 bandCols  = cBatch + 2 * margin;
            if (cxBandMin < cxMin) cxBandMin = cxMin;
            i32 bandEndX = cxBandMin + bandCols;
            if (bandEndX > cxMax + 1) bandEndX = cxMax + 1;
            bandCols = bandEndX - cxBandMin;
            if (bandCols > TILE_COLS) bandCols = TILE_COLS;

            i32 matchColStart = (cxMin + margin + aj) - cxBandMin;
            i32 matchColEnd   = matchColStart + cBatch;
            if (matchColStart < 0) matchColStart = 0;
            if (matchColEnd > bandCols) matchColEnd = bandCols;
            if (matchColEnd <= matchColStart) continue;

            // The resident grid for this tile is bandRows x bandCols chunks, row
            // pitch = bandCols (NOT the full search width). The kernels index a
            // chunk as (row*numChunksX + col)*stride, so pass numChunksX=bandCols.
            size_t usedBytes = (size_t)bandRows * bandCols * bytesPerChunk;

            // Phase G: zero the used region, then scatter-generate seeding chunks.
            double tg0 = wallSeconds();
            checkCuda(cudaMemset(d_tile, 0, usedBytes), "zero tile grid");

            i32 genThreads = bandCols * bandRows;
            i32 genGrid    = (genThreads + blockSize - 1) / blockSize;
            for (i32 os = 0; os < numOreSlots; os++) {
                generateTile<<<genGrid, blockSize>>>(
                    d_tile,
                    worldSeed, POP_A, POP_B,
                    czBandMin, cxBandMin, bandCols, bandRows,
                    chunkStride, os, slotOff_h[os], activeOre[os]);
            }
            checkCuda(cudaGetLastError(), "generateTile");
            checkCuda(cudaDeviceSynchronize(), "sync generateTile");
            double tg1 = wallSeconds();
            g_genSeconds += tg1 - tg0;

            // Phase M: match this tile's interior anchor window. matchTile scans
            // all columns of rows [matchRowStart,matchRowEnd); restrict to the
            // interior column window via the new matchColStart/End bounds.
            double tm0 = wallSeconds();
            i32 matchThreads = (matchColEnd - matchColStart) * (matchRowEnd - matchRowStart);
            i32 matchGrid    = (matchThreads + blockSize - 1) / blockSize;
            matchTile<<<matchGrid, blockSize>>>(
                d_tile,
                bandRows,
                cxBandMin, czBandMin,
                matchRowStart, matchRowEnd,
                matchColStart, matchColEnd,
                bandCols,
                anchorOreSlot,
                rSq,
                centX, centZ,
                tStart, tEnd,
                d_patterns, d_patOre, patCount,
                patCount,
                d_results, d_resCount);
            checkCuda(cudaGetLastError(), "matchTile");
            checkCuda(cudaDeviceSynchronize(), "sync matchTile");
            double tm1 = wallSeconds();
            g_matchSeconds += tm1 - tm0;

            collectResults();

            processedTiles++;
            printf("Progress: %d/%d tiles done  best=%d/%d\n",
                   processedTiles, totalTiles, bestScore, patCount);
            fflush(stdout);
        }
    }

    printf("\nAll done. Best score: %d/%d  Total matches: %d\n",
           bestScore, patCount, totalMatches);
    printf("Matches found: %d (written to matches.txt)\n", totalMatches);
    printf("Phase split: generate %.2fs  match %.2fs\n", g_genSeconds, g_matchSeconds);

    if (g_matchFile) { fclose(g_matchFile); g_matchFile = NULL; }

    free(rowOrder);
    free(colOrder);
    cudaFree(d_tile);
    cudaFree(d_patterns);
    cudaFree(d_patOre);
    cudaFree(d_results);
    cudaFree(d_resCount);
    free(h_results);
    free(patterns_h);
    return 0;
}
