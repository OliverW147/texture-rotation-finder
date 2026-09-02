// Ore pattern matcher for Minecraft 1.21 / 26.1.2
// Processes the search region in a parallelized grid of fixed-size square slabs.
// Slabs are sorted to process from the center outward to find nearby matches first.
// Uses an inlined flat 3D Bitset pipeline to write ores directly to memory during generation.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <omp.h>

#define PI 3.14159265358979323846

typedef int64_t  i64;
typedef uint64_t u64;
typedef int32_t  i32;
typedef uint32_t u32;

static int g_centX = 0;
static int g_centZ = 0;
static size_t g_bitset_words = 0;
// Match output: write full matches to a file to avoid flooding/dropping stdout
// under many threads. stdout gets only progress + a summary.
static FILE *g_matchFile = NULL;
static long long g_matchCount = 0;
static omp_lock_t g_matchLock;

// Precomputed Sine Table matching exact float-to-double promotion rules of Minecraft
static double SIN_TABLE[64][64];

void initSinTable(void) {
    for (int sz = 1; sz < 64; sz++) {
        for (int i = 0; i < sz; i++) {
            float pct = (float)i / (float)sz;
            SIN_TABLE[sz][i] = sin(PI * (double)pct);
        }
    }
}

// ---- Xoroshiro128 ----
typedef struct { u64 lo, hi; } Xoro;

static inline u64 rotl64(u64 x, int b) { return (x<<b)|(x>>(64-b)); }

static inline void xSetSeed(Xoro *xr, u64 v) {
    const u64 XL=0x9e3779b97f4a7c15ULL, XH=0x6a09e667f3bcc909ULL;
    const u64 A=0xbf58476d1ce4e5b9ULL, B=0x94d049bb133111ebULL;
    u64 l=v^XH, h=l+XL;
    l=(l^(l>>30))*A; h=(h^(h>>30))*A;
    l=(l^(l>>27))*B; h=(h^(h>>27))*B;
    xr->lo=l^(l>>31); xr->hi=h^(h>>31);
}

static inline u64 xNext(Xoro *xr) {
    u64 l=xr->lo, h=xr->hi;
    u64 n=rotl64(l+h,17)+l;
    h^=l; xr->lo=rotl64(l,49)^h^(h<<21); xr->hi=rotl64(h,28);
    return n;
}

static inline int next31(Xoro *xr) { return (int)(xNext(xr) >> 33); }

static inline int xNextInt(Xoro *xr, u32 n) {
    int bits, val;
    u32 m = n - 1;
    if ((m & n) == 0)
        return (int)(((i64)next31(xr) * (i64)n) >> 31);
    do {
        bits = next31(xr);
        val  = bits % (int)n;
    } while ((i32)((u32)bits - (u32)val + m) < 0);
    return val;
}

// Optimized fast-path modulos for height generators to avoid slow div assembly instructions
static inline int xNextInt3(Xoro *xr) {
    int bits, val;
    do { bits = next31(xr); val = bits % 3; } while (bits - val + 2 < 0);
    return val;
}

static inline int xNextInt9(Xoro *xr) {
    int bits, val;
    do { bits = next31(xr); val = bits % 9; } while (bits - val + 8 < 0);
    return val;
}

static inline int xNextInt17(Xoro *xr) {
    int bits, val;
    do { bits = next31(xr); val = bits % 17; } while (bits - val + 16 < 0);
    return val;
}

static inline int xNextInt25(Xoro *xr) {
    int bits, val;
    do { bits = next31(xr); val = bits % 25; } while (bits - val + 24 < 0);
    return val;
}

static inline int xNextInt33(Xoro *xr) {
    int bits, val;
    do { bits = next31(xr); val = bits % 33; } while (bits - val + 32 < 0);
    return val;
}

static inline int xNextInt41(Xoro *xr) {
    int bits, val;
    do { bits = next31(xr); val = bits % 41; } while (bits - val + 40 < 0);
    return val;
}

static inline int xNextInt65(Xoro *xr) {
    int bits, val;
    do { bits = next31(xr); val = bits % 65; } while (bits - val + 64 < 0);
    return val;
}

static inline int xNextInt80(Xoro *xr) {
    int bits, val;
    do { bits = next31(xr); val = bits % 80; } while (bits - val + 79 < 0);
    return val;
}

static inline int xNextInt97(Xoro *xr) {
    int bits, val;
    do { bits = next31(xr); val = bits % 97; } while (bits - val + 96 < 0);
    return val;
}

static inline int xNextInt129(Xoro *xr) {
    int bits, val;
    do { bits = next31(xr); val = bits % 129; } while (bits - val + 128 < 0);
    return val;
}

static inline int xNextInt137(Xoro *xr) {
    int bits, val;
    do { bits = next31(xr); val = bits % 137; } while (bits - val + 136 < 0);
    return val;
}

static inline int xNextInt192(Xoro *xr) {
    int bits, val;
    do { bits = next31(xr); val = bits % 192; } while (bits - val + 191 < 0);
    return val;
}

static inline int xNextInt193(Xoro *xr) {
    int bits, val;
    do { bits = next31(xr); val = bits % 193; } while (bits - val + 192 < 0);
    return val;
}

static inline int xNextInt305(Xoro *xr) {
    int bits, val;
    do { bits = next31(xr); val = bits % 305; } while (bits - val + 304 < 0);
    return val;
}

static inline int xNextIntBetween(Xoro *xr, int a, int b) {
    if (a>=b) return a;
    return a+xNextInt(xr,(u32)(b-a+1));
}

static inline double xNextDouble(Xoro *xr) {
    i64 a=(i32)(xNext(xr)>>(64-26));
    i64 b=(i32)(xNext(xr)>>(64-27));
    return ((a<<27)+b)*(1.0/(1ULL<<53));
}

static inline float xNextFloat(Xoro *xr) {
    return (i32)(xNext(xr)>>(64-24))*(1.0f/(1<<24));
}

static inline u64 xNextLongJ(Xoro *xr) {
    i32 a=(i32)(xNext(xr)>>32);
    i32 b=(i32)(xNext(xr)>>32);
    return (u64)(((i64)a<<32)+(i64)b);
}

// Precomputed multipliers for populationSeed (constant per world seed)
static u64 POP_A, POP_B;

static void initPopSeedMults(u64 worldSeed) {
    Xoro xr; xSetSeed(&xr, worldSeed);
    POP_A = xNextLongJ(&xr)|1ULL;
    POP_B = xNextLongJ(&xr)|1ULL;
}

static inline u64 populationSeed(u64 worldSeed, int blockX, int blockZ) {
    return (u64)blockX*POP_A + (u64)blockZ*POP_B ^ worldSeed;
}

// Trapezoid height helpers: range=max-min, shoulder=range/2, inner=range-shoulder
// min + nextInt(inner+1) + nextInt(shoulder+1)
// iron_middle: range=80, shoulder=40, inner=40  -> -24 + [0..40] + [0..40]
static inline int heightIronMiddle(Xoro *xr)   { return -24 + xNextInt41(xr) + xNextInt41(xr); }
// iron_upper: range=304, shoulder=152, inner=152 -> 80 + [0..152] + [0..152]
static inline int heightIronUpper(Xoro *xr)    { return  80 + xNextInt(xr,153) + xNextInt(xr,153); }
// coal_upper: uniform [0..192]
static inline int heightCoalUpper(Xoro *xr)    { return xNextInt193(xr); }
// coal_lower: range=192, shoulder=96, inner=96 -> 0 + [0..96] + [0..96]
static inline int heightCoalLower(Xoro *xr)    { return xNextInt97(xr) + xNextInt97(xr); }
// gold: range=96, shoulder=48, inner=48 -> -64 + [0..48] + [0..48]
static inline int heightGold(Xoro *xr)         { return -64 + xNextInt(xr,49) + xNextInt(xr,49); }
// gold_lower: range=16, shoulder=8, inner=8 -> -64 + [0..8] + [0..8]
static inline int heightGoldLower(Xoro *xr)    { return -64 + xNextInt9(xr) + xNextInt9(xr); }
// redstone: uniform [-64..15] = -64 + [0..79]
static inline int heightRedstone(Xoro *xr)     { return -64 + xNextInt80(xr); }
// redstone_lower: range=8, shoulder=4, inner=4 -> -32 + [0..4] + [0..4]
static inline int heightRedstoneLower(Xoro *xr){ return -32 + xNextInt(xr,5) + xNextInt(xr,5); }
// diamond: range=60, shoulder=30, inner=30 -> -64 + [0..30] + [0..30]
static inline int heightDiamond(Xoro *xr)      { return -64 + xNextInt(xr,31) + xNextInt(xr,31); }
// lapis: range=64, shoulder=32, inner=32 -> -32 + [0..32] + [0..32]
static inline int heightLapis(Xoro *xr)        { return -32 + xNextInt33(xr) + xNextInt33(xr); }
// copper_large: range=128, shoulder=64, inner=64 -> -16 + [0..64] + [0..64]
static inline int heightCopper(Xoro *xr)       { return -16 + xNextInt65(xr) + xNextInt65(xr); }
// iron_small: uniform -64 + [0..136]
static inline int heightIronSmall_(Xoro *xr)   { return -64 + xNextInt137(xr); }

#define MAX_ORE_TYPES 8

// ---- Per-thread/Slab state ----
// One bitset + key list per active ore type
typedef struct {
    u64 *bitsets[MAX_ORE_TYPES]; // one bitset per active ore type
    u64 *keys;                   // anchor-ore core blocks (search candidates)
    u32  nkeys;
    u32  kcap;
    int  collectKeys;            // per-thread: collect keys only for the anchor ore
} ThreadState;

typedef struct {
    int cxA, cxB, czA, czB;
    double distSq; // Distance from the search center for sorting
} SlabTask;

static inline u64 packXYZ(int x, int y, int z) {
    u64 dx = (u64)(x - g_centX + 8388608) & 0xFFFFFFULL;
    u64 dz = (u64)(z - g_centZ + 8388608) & 0xFFFFFFULL;
    u64 dy = (u64)(y + 64) & 0x3FFULL;
    return (dx << 34) | (dz << 10) | dy;
}

// Fast branchless containment check assuming coordinate bounds are verified
static inline int bitsetContainsFast(const u64 *bitset, int x, int y, int z, 
                                     int min_bx, int min_bz, int range_z) {
    int dx = x - min_bx;
    int dz = z - min_bz;
    int dy = y + 64;

    if (dy < 0 || dy >= 384) {
        return 0;
    }

    u64 bit_idx = ((u64)dx * 384 + (u64)dy) * range_z + dz;
    return (bitset[bit_idx >> 6] & (1ULL << (bit_idx & 63))) != 0;
}

// ---- Ore vein generation directly into the bitset ----
static void generateVein(ThreadState *ts, u64 *bitset, Xoro *xr, int baseX, int baseY, int baseZ, int size,
                         int min_bx, int min_bz, int range_x, int range_z,
                         int core_min_x, int core_max_x, int core_min_z, int core_max_z) {
    float angle=xNextFloat(xr)*(float)PI;
    float szf=(float)size/8.0f;

    double s, c;
    __builtin_sincos(angle, &s, &c);

    double oxp = baseX + s * (double)szf;
    double oxn = baseX - s * (double)szf;
    double ozp = baseZ + c * (double)szf;
    double ozn = baseZ - c * (double)szf;
    double oyp = baseY + xNextInt3(xr) - 2;
    double oyn = baseY + xNextInt3(xr) - 2;

    double sz_div_16 = (double)size / 16.0;

    double store[20*4];
    for (int i=0;i<size;i++) {
        double len = xNextDouble(xr) * sz_div_16;
        double off = ((SIN_TABLE[size][i] + 1.0) * len + 1.0) * 0.5;
        double pct=(float)i/(float)size;
        double x=oxp+pct*(oxn-oxp), y=oyp+pct*(oyn-oyp), z=ozp+pct*(ozn-ozp);
        store[i*4]=x; store[i*4+1]=y; store[i*4+2]=z; store[i*4+3]=off;
    }

    for (int i=0;i<size-1;i++) {
        if (store[i*4+3]<=0) continue;
        for (int j=i+1;j<size;j++) {
            if (store[j*4+3]<=0) continue;
            double dx=store[i*4]-store[j*4], dy=store[i*4+1]-store[j*4+1], dz=store[i*4+2]-store[j*4+2];
            double doff=store[i*4+3]-store[j*4+3];
            if (doff*doff<=dx*dx+dy*dy+dz*dz) continue;
            if (doff>0) store[j*4+3]=-1; else store[i*4+3]=-1;
        }
    }

    for (int i=0;i<size;i++) {
        double off=store[i*4+3];
        if (off<=0.0) continue;
        double sx=store[i*4], sy=store[i*4+1], sz=store[i*4+2];
        int minX=(int)floor(sx-off), maxX=(int)floor(sx+off);
        int minY=(int)floor(sy-off), maxY=(int)floor(sy+off);
        int minZ=(int)floor(sz-off), maxZ=(int)floor(sz+off);

        int clamp_minY = minY > -64 ? minY : -64;
        int clamp_maxY = maxY < 319 ? maxY : 319;

        // Clip X-bounds to the active task range to remove branches inside loop
        int startX = minX;
        int endX = maxX;
        if (startX < min_bx) startX = min_bx;
        if (endX > min_bx + range_x - 1) endX = min_bx + range_x - 1;

        double inv_off = 1.0 / off;

        for (int X=startX; X<=endX; X++) {
            double xsl=((double)X+0.5-sx) * inv_off;
            double xsl2 = xsl*xsl;
            if (xsl2>=1.0) continue;

            int dx = X - min_bx;

            double r2_left_Y = 1.0 - xsl2;
            double dy_limit = off * sqrt(r2_left_Y);
            int y_min = (int)floor(sy - 0.5 - dy_limit) + 1;
            int y_max = (int)ceil(sy - 0.5 + dy_limit) - 1;
            if (y_min < clamp_minY) y_min = clamp_minY;
            if (y_max > clamp_maxY) y_max = clamp_maxY;

            int is_core_X = (X >= core_min_x && X <= core_max_x);

            for (int Y=y_min; Y<=y_max; Y++) {
                double ysl=((double)Y+0.5-sy) * inv_off;
                double ysl2 = ysl*ysl;
                double r2_left = r2_left_Y - ysl2;
                if (r2_left<=0.0) continue;

                int dy = Y + 64;

                double dz_limit = off * sqrt(r2_left);
                int z_min = (int)floor(sz - 0.5 - dz_limit) + 1;
                int z_max = (int)ceil(sz - 0.5 + dz_limit) - 1;
                if (z_min < minZ) z_min = minZ;
                if (z_max > maxZ) z_max = maxZ;

                // Precompute index constant part (removes multiplication inside Z loop)
                u64 const_idx = ((u64)dx * 384 + (u64)dy) * range_z;

                for (int Z=z_min; Z<=z_max; Z++) {
                    int dz = Z - min_bz;

                    u64 bit_idx = const_idx + dz;
                    u64 word_idx = bit_idx >> 6;
                    u64 mask = 1ULL << (bit_idx & 63);

                    if ((bitset[word_idx] & mask) == 0) {
                        bitset[word_idx] |= mask;

                        int is_core = ts->collectKeys && is_core_X && (Z >= core_min_z && Z <= core_max_z);
                        if (is_core) {
                            if (ts->nkeys == ts->kcap) {
                                ts->kcap = ts->kcap ? ts->kcap * 2 : 65536;
                                ts->keys = realloc(ts->keys, ts->kcap * sizeof(u64));
                            }
                            ts->keys[ts->nkeys++] = packXYZ(X, Y, Z);
                        }
                    }
                }
            }
        }
    }
}

// Ore type constants -- individual placements and combined groups
// Combined (group all sub-placements for that ore mineral):
#define ORE_IRON          0  // iron_middle(12) + iron_small(13)
#define ORE_COAL          1  // coal_upper(9) + coal_lower(10)
#define ORE_COPPER        2  // copper_large(24)
#define ORE_GOLD          3  // gold(14) + gold_lower(15)
#define ORE_REDSTONE      4  // redstone(16) + redstone_lower(17)
#define ORE_LAPIS         5  // lapis(22) + lapis_buried(23)
#define ORE_DIAMOND       6  // diamond(18-21) all four placements
// Individual placements:
#define ORE_IRON_UPPER    7  // idx 11: trapezoid  80..384  size=9  count=90
#define ORE_IRON_MIDDLE   8  // idx 12: trapezoid -24..56   size=9  count=10
#define ORE_IRON_SMALL    9  // idx 13: uniform   -64..72   size=4  count=10
#define ORE_COAL_UPPER   10  // idx  9: uniform     0..192  size=17 count=20
#define ORE_COAL_LOWER   11  // idx 10: trapezoid   0..192  size=17 count=20
#define ORE_GOLD_NORMAL  12  // idx 14: trapezoid -64..32   size=9  count=4
#define ORE_GOLD_LOWER   13  // idx 15: trapezoid -64..-48  size=9  count=4
#define ORE_REDSTONE_N   14  // idx 16: uniform   -64..15   size=8  count=8
#define ORE_REDSTONE_L   15  // idx 17: trapezoid -32..-24  size=8  count=8
#define ORE_DIAMOND_N    16  // idx 18: trapezoid -64..-4   size=8  count=7
#define ORE_DIAMOND_M    17  // idx 19: trapezoid -64..-4   size=8  count=2
#define ORE_DIAMOND_LG   18  // idx 20: trapezoid -64..-4   size=12 count=1
#define ORE_DIAMOND_BUR  19  // idx 21: trapezoid -64..-4   size=4  count=4
#define ORE_LAPIS_N      20  // idx 22: trapezoid -32..32   size=7  count=2
#define ORE_LAPIS_BUR    21  // idx 23: trapezoid -32..32   size=7  count=4
#define ORE_COPPER_LG    22  // idx 24: trapezoid -16..112  size=20 count=16
// All ores combined (everything idx 9-24):
#define ORE_ALL          23

static void generateChunkOres(ThreadState *ts, u64 *bitset, u64 worldSeed, int cx, int cz, int oreType,
                              int min_bx, int min_bz, int range_x, int range_z,
                              int core_min_x, int core_max_x, int core_min_z, int core_max_z) {
    int blockX=cx<<4, blockZ=cz<<4;
    u64 popSeed=populationSeed(worldSeed,blockX,blockZ);

#define GEN(idx, count, size, heightFn) \
    { Xoro xr; xSetSeed(&xr, popSeed+(idx)+10000ULL*6); \
      for (int i=0;i<(count);i++) { \
          int bx=blockX+xNextInt(&xr,16); \
          int bz=blockZ+xNextInt(&xr,16); \
          int by=(heightFn)(&xr); \
          generateVein(ts,bitset,&xr,bx,by,bz,(size),min_bx,min_bz,range_x,range_z, \
                       core_min_x,core_max_x,core_min_z,core_max_z); \
      } }

    switch (oreType) {
        // Combined groups
        case ORE_IRON:
            GEN(12, 10, 9, heightIronMiddle)
            GEN(13, 10, 4, heightIronSmall_)
            break;
        case ORE_COAL:
            GEN( 9, 20, 17, heightCoalUpper)
            GEN(10, 20, 17, heightCoalLower)
            break;
        case ORE_COPPER:
            GEN(24, 16, 20, heightCopper)
            break;
        case ORE_GOLD:
            GEN(14, 4, 9, heightGold)
            GEN(15, 4, 9, heightGoldLower)
            break;
        case ORE_REDSTONE:
            GEN(16, 8, 8, heightRedstone)
            GEN(17, 8, 8, heightRedstoneLower)
            break;
        case ORE_LAPIS:
            GEN(22, 2, 7, heightLapis)
            GEN(23, 4, 7, heightLapis)
            break;
        case ORE_DIAMOND:
            GEN(18, 7,  8, heightDiamond)
            GEN(19, 2,  8, heightDiamond)
            GEN(20, 1, 12, heightDiamond)
            GEN(21, 4,  4, heightDiamond)
            break;
        // Individual placements
        case ORE_IRON_UPPER:   GEN(11, 90,  9, heightIronUpper)    break;
        case ORE_IRON_MIDDLE:  GEN(12, 10,  9, heightIronMiddle)   break;
        case ORE_IRON_SMALL:   GEN(13, 10,  4, heightIronSmall_)   break;
        case ORE_COAL_UPPER:   GEN( 9, 20, 17, heightCoalUpper)    break;
        case ORE_COAL_LOWER:   GEN(10, 20, 17, heightCoalLower)    break;
        case ORE_GOLD_NORMAL:  GEN(14,  4,  9, heightGold)         break;
        case ORE_GOLD_LOWER:   GEN(15,  4,  9, heightGoldLower)    break;
        case ORE_REDSTONE_N:   GEN(16,  8,  8, heightRedstone)     break;
        case ORE_REDSTONE_L:   GEN(17,  8,  8, heightRedstoneLower) break;
        case ORE_DIAMOND_N:    GEN(18,  7,  8, heightDiamond)      break;
        case ORE_DIAMOND_M:    GEN(19,  2,  8, heightDiamond)      break;
        case ORE_DIAMOND_LG:   GEN(20,  1, 12, heightDiamond)      break;
        case ORE_DIAMOND_BUR:  GEN(21,  4,  4, heightDiamond)      break;
        case ORE_LAPIS_N:      GEN(22,  2,  7, heightLapis)        break;
        case ORE_LAPIS_BUR:    GEN(23,  4,  7, heightLapis)        break;
        case ORE_COPPER_LG:    GEN(24, 16, 20, heightCopper)       break;
        // Everything
        case ORE_ALL:
            GEN( 9, 20, 17, heightCoalUpper)
            GEN(10, 20, 17, heightCoalLower)
            GEN(11, 90,  9, heightIronUpper)
            GEN(12, 10,  9, heightIronMiddle)
            GEN(13, 10,  4, heightIronSmall_)
            GEN(14,  4,  9, heightGold)
            GEN(15,  4,  9, heightGoldLower)
            GEN(16,  8,  8, heightRedstone)
            GEN(17,  8,  8, heightRedstoneLower)
            GEN(18,  7,  8, heightDiamond)
            GEN(19,  2,  8, heightDiamond)
            GEN(20,  1, 12, heightDiamond)
            GEN(21,  4,  4, heightDiamond)
            GEN(22,  2,  7, heightLapis)
            GEN(23,  4,  7, heightLapis)
            GEN(24, 16, 20, heightCopper)
            break;
    }
#undef GEN
}

// ---- Pattern ----
#define MAX_PATTERN 64

static int PATTERN_COUNT = 0;
static int PATTERN[MAX_PATTERN][3];
static int PATTERN_ORE[MAX_PATTERN];       // ore type index (into ACTIVE_ORES) per offset
static int TRANSFORMED_PATTERN[8][MAX_PATTERN][3];

// Unique ore types actually used across all offsets
static int ACTIVE_ORES[MAX_ORE_TYPES];
static int ACTIVE_ORE_COUNT = 0;

// Approximate per-chunk generation cost (ns), measured via bench_gen.exe.
// Used to pick the cheapest ore type as the search anchor: anchoring on the
// rarest/cheapest ore shrinks the candidate set we iterate in Phase 2.
static double oreAnchorCost(int oreType) {
    switch (oreType) {
        case ORE_LAPIS:    return 2885;
        case ORE_GOLD:     return 5170;
        case ORE_REDSTONE: return 5200;
        case ORE_DIAMOND:  return 7225;
        case ORE_IRON:     return 8915;
        case ORE_COPPER:   return 30000;
        case ORE_COAL:     return 62290;
        default:           return 9000; // individual placements ~ iron-ish
    }
}

// Returns the index into ACTIVE_ORES for a given ore type, inserting if new
static int oreSlot(int oreType) {
    for (int i = 0; i < ACTIVE_ORE_COUNT; i++)
        if (ACTIVE_ORES[i] == oreType) return i;
    ACTIVE_ORES[ACTIVE_ORE_COUNT] = oreType;
    return ACTIVE_ORE_COUNT++;
}

static void applyTransform(int t, int dx, int dy, int dz,
                           int *ox, int *oy, int *oz) {
    *oy=dy;
    switch(t) {
        case 0: *ox= dx; *oz= dz; break;
        case 1: *ox= dx; *oz=-dz; break;
        case 2: *ox=-dx; *oz= dz; break;
        case 3: *ox=-dx; *oz=-dz; break;
        case 4: *ox= dz; *oz= dx; break;
        case 5: *ox= dz; *oz=-dx; break;
        case 6: *ox=-dz; *oz= dx; break;
        case 7: *ox=-dz; *oz=-dx; break;
    }
}

// Optimized pattern scoring with early-exit abortion based on maximum allowed misses
static int scorePattern(ThreadState *ts, int cx, int cy, int cz, int t, int min_score,
                        int min_bx, int min_bz, int range_x, int range_z) {
    int misses = 0;
    int max_allowed_misses = PATTERN_COUNT - min_score;
    int score = 0;

    for (int p = 0; p < PATTERN_COUNT; p++) {
        int ox = TRANSFORMED_PATTERN[t][p][0];
        int oy = TRANSFORMED_PATTERN[t][p][1];
        int oz = TRANSFORMED_PATTERN[t][p][2];
        u64 *bitset = ts->bitsets[PATTERN_ORE[p]];
        if (bitsetContainsFast(bitset, cx + ox, cy + oy, cz + oz, min_bx, min_bz, range_z)) {
            score++;
        } else {
            misses++;
            if (misses > max_allowed_misses) {
                return -1;
            }
        }
    }
    return score;
}

static int patternReachBlocks(void) {
    int maxr=0;
    for (int p=0;p<PATTERN_COUNT;p++) {
        int dx=abs(PATTERN[p][0]), dz=abs(PATTERN[p][2]);
        int r = dx>dz ? dx : dz;
        if (r>maxr) maxr=r;
    }
    return maxr;
}

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s <seed> <cx> <cy> <cz> <radiusBlocks> <orient> [--threads N] [--ore TYPE] <dx,dy,dz>...\n"
        "  orient: -1=all 8, or 0..7\n"
        "\n"
        "  Combined types (group all sub-placements for that mineral):\n"
        "    iron       = iron_middle+iron_small(12,13)     y=-24..56 and -64..72\n"
        "    coal       = coal_upper+coal_lower(9,10)       y=0..192\n"
        "    copper     = copper_large(24)                  y=-16..112\n"
        "    gold       = gold+gold_lower(14,15)            y=-64..32 and -64..-48\n"
        "    redstone   = redstone+redstone_lower(16,17)    y=-64..15 and -32..-24\n"
        "    lapis      = lapis+lapis_buried(22,23)         y=-32..32\n"
        "    diamond    = diamond all four(18-21)           y=-64..-4\n"
        "\n"
        "  Individual placements:\n"
        "    iron_upper   idx=11  trapezoid  80..384  size=9  count=90\n"
        "    iron_middle  idx=12  trapezoid -24..56   size=9  count=10\n"
        "    iron_small   idx=13  uniform   -64..72   size=4  count=10\n"
        "    coal_upper   idx=9   uniform     0..192  size=17 count=20\n"
        "    coal_lower   idx=10  trapezoid   0..192  size=17 count=20\n"
        "    gold_normal  idx=14  trapezoid -64..32   size=9  count=4\n"
        "    gold_lower   idx=15  trapezoid -64..-48  size=9  count=4\n"
        "    redstone_n   idx=16  uniform   -64..15   size=8  count=8\n"
        "    redstone_lower idx=17 trapezoid -32..-24 size=8  count=8\n"
        "    diamond_n    idx=18  trapezoid -64..-4   size=8  count=7\n"
        "    diamond_m    idx=19  trapezoid -64..-4   size=8  count=2\n"
        "    diamond_lg   idx=20  trapezoid -64..-4   size=12 count=1\n"
        "    diamond_bur  idx=21  trapezoid -64..-4   size=4  count=4\n"
        "    lapis_n      idx=22  trapezoid -32..32   size=7  count=2\n"
        "    lapis_bur    idx=23  trapezoid -32..32   size=7  count=4\n"
        "    copper_lg    idx=24  trapezoid -16..112  size=20 count=16\n"
        "    all          = every ore above (idx 9-24)\n"
        "\nExample:\n"
        "  %s -377264746167088810 205 17 -88 1000 -1 1,0,0 0,3,3 1,4,3 -1,4,3 5,1,6 9,4,4\n",
        prog, prog);
}

// Compare function for sorting tasks from center outward
int compareTasks(const void *a, const void *b) {
    double distA = ((SlabTask*)a)->distSq;
    double distB = ((SlabTask*)b)->distSq;
    if (distA < distB) return -1;
    if (distA > distB) return 1;
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 8) { usage(argv[0]); return 1; }

    i64 WORLD_SEED = (i64)strtoll(argv[1], NULL, 10);
    int centX      = atoi(argv[2]);
    int centY      = atoi(argv[3]);
    int centZ      = atoi(argv[4]);
    int radiusBlks = atoi(argv[5]);
    int orientArg  = atoi(argv[6]);

    g_centX = centX;
    g_centZ = centZ;

    int nthreads = omp_get_max_threads();
    int oreType = ORE_IRON;

    // Anchor override: -1 = auto (pick cheapest), -2 = legacy (first offset),
    // >=0 = force that ore type as anchor.
    int forcedAnchorOre = -1;

    int argOff = 7;
    while (argOff < argc) {
        if (strcmp(argv[argOff], "--threads") == 0) {
            if (argOff+1 >= argc) { fprintf(stderr,"--threads needs a value\n"); return 1; }
            nthreads = atoi(argv[argOff+1]);
            if (nthreads < 1) nthreads = 1;
            argOff += 2;
        } else if (strcmp(argv[argOff], "--anchor") == 0) {
            if (argOff+1 >= argc) { fprintf(stderr,"--anchor needs a value\n"); return 1; }
            const char *s = argv[argOff+1];
            if      (strcmp(s,"auto")   == 0) forcedAnchorOre = -1;
            else if (strcmp(s,"legacy") == 0) forcedAnchorOre = -2;
            else {
                #define A1(nm,v) else if(strcmp(s,nm)==0) forcedAnchorOre=v
                if(0){}
                A1("iron",ORE_IRON); A1("coal",ORE_COAL); A1("copper",ORE_COPPER);
                A1("gold",ORE_GOLD); A1("redstone",ORE_REDSTONE); A1("lapis",ORE_LAPIS);
                A1("diamond",ORE_DIAMOND);
                #undef A1
                else { fprintf(stderr,"--anchor: unknown ore '%s'\n",s); return 1; }
            }
            argOff += 2;
        } else if (strcmp(argv[argOff], "--ore") == 0) {
            if (argOff+1 >= argc) { fprintf(stderr,"--ore needs a value\n"); return 1; }
            const char *s = argv[argOff+1];
            if      (strcmp(s,"iron")          == 0) oreType = ORE_IRON;
            else if (strcmp(s,"coal")          == 0) oreType = ORE_COAL;
            else if (strcmp(s,"copper")        == 0) oreType = ORE_COPPER;
            else if (strcmp(s,"gold")          == 0) oreType = ORE_GOLD;
            else if (strcmp(s,"redstone")      == 0) oreType = ORE_REDSTONE;
            else if (strcmp(s,"lapis")         == 0) oreType = ORE_LAPIS;
            else if (strcmp(s,"diamond")       == 0) oreType = ORE_DIAMOND;
            else if (strcmp(s,"iron_upper")    == 0) oreType = ORE_IRON_UPPER;
            else if (strcmp(s,"iron_middle")   == 0) oreType = ORE_IRON_MIDDLE;
            else if (strcmp(s,"iron_small")    == 0) oreType = ORE_IRON_SMALL;
            else if (strcmp(s,"coal_upper")    == 0) oreType = ORE_COAL_UPPER;
            else if (strcmp(s,"coal_lower")    == 0) oreType = ORE_COAL_LOWER;
            else if (strcmp(s,"gold_normal")   == 0) oreType = ORE_GOLD_NORMAL;
            else if (strcmp(s,"gold_lower")    == 0) oreType = ORE_GOLD_LOWER;
            else if (strcmp(s,"redstone_n")    == 0) oreType = ORE_REDSTONE_N;
            else if (strcmp(s,"redstone_lower")== 0) oreType = ORE_REDSTONE_L;
            else if (strcmp(s,"diamond_n")     == 0) oreType = ORE_DIAMOND_N;
            else if (strcmp(s,"diamond_m")     == 0) oreType = ORE_DIAMOND_M;
            else if (strcmp(s,"diamond_lg")    == 0) oreType = ORE_DIAMOND_LG;
            else if (strcmp(s,"diamond_bur")   == 0) oreType = ORE_DIAMOND_BUR;
            else if (strcmp(s,"lapis_n")       == 0) oreType = ORE_LAPIS_N;
            else if (strcmp(s,"lapis_bur")     == 0) oreType = ORE_LAPIS_BUR;
            else if (strcmp(s,"copper_lg")     == 0) oreType = ORE_COPPER_LG;
            else if (strcmp(s,"all")           == 0) oreType = ORE_ALL;
            else {
                fprintf(stderr,"Unknown ore '%s'. Run with no args to see full list.\n", s);
                return 1;
            }
            argOff += 2;
        } else {
            break;
        }
    }

    omp_set_num_threads(nthreads);

    // Parse ore name string to ORE_* constant (-1 on failure)
    #define PARSE_ORE(s) ( \
        strcmp(s,"iron")==0          ? ORE_IRON       : \
        strcmp(s,"coal")==0          ? ORE_COAL       : \
        strcmp(s,"copper")==0        ? ORE_COPPER     : \
        strcmp(s,"gold")==0          ? ORE_GOLD       : \
        strcmp(s,"redstone")==0      ? ORE_REDSTONE   : \
        strcmp(s,"lapis")==0         ? ORE_LAPIS      : \
        strcmp(s,"diamond")==0       ? ORE_DIAMOND    : \
        strcmp(s,"iron_upper")==0    ? ORE_IRON_UPPER : \
        strcmp(s,"iron_middle")==0   ? ORE_IRON_MIDDLE: \
        strcmp(s,"iron_small")==0    ? ORE_IRON_SMALL : \
        strcmp(s,"coal_upper")==0    ? ORE_COAL_UPPER : \
        strcmp(s,"coal_lower")==0    ? ORE_COAL_LOWER : \
        strcmp(s,"gold_normal")==0   ? ORE_GOLD_NORMAL: \
        strcmp(s,"gold_lower")==0    ? ORE_GOLD_LOWER : \
        strcmp(s,"redstone_n")==0    ? ORE_REDSTONE_N : \
        strcmp(s,"redstone_lower")==0? ORE_REDSTONE_L : \
        strcmp(s,"diamond_n")==0     ? ORE_DIAMOND_N  : \
        strcmp(s,"diamond_m")==0     ? ORE_DIAMOND_M  : \
        strcmp(s,"diamond_lg")==0    ? ORE_DIAMOND_LG : \
        strcmp(s,"diamond_bur")==0   ? ORE_DIAMOND_BUR: \
        strcmp(s,"lapis_n")==0       ? ORE_LAPIS_N    : \
        strcmp(s,"lapis_bur")==0     ? ORE_LAPIS_BUR  : \
        strcmp(s,"copper_lg")==0     ? ORE_COPPER_LG  : \
        strcmp(s,"all")==0           ? ORE_ALL        : -1)

    PATTERN_COUNT = 0;
    ACTIVE_ORE_COUNT = 0;
    // Ensure the global ore type is always slot 0
    oreSlot(oreType);

    for (int i = argOff; i < argc && PATTERN_COUNT < MAX_PATTERN; i++) {
        int dx, dy, dz;
        char oreStr[32] = "";
        // Accept dx,dy,dz or dx,dy,dz,orename
        int parsed = sscanf(argv[i], "%d,%d,%d,%31s", &dx, &dy, &dz, oreStr);
        if (parsed < 3) { fprintf(stderr, "Bad offset '%s'\n", argv[i]); return 1; }

        int offOreType = oreType; // default to global
        if (parsed == 4) {
            int t2 = PARSE_ORE(oreStr);
            if (t2 < 0) { fprintf(stderr, "Unknown ore '%s' in offset '%s'\n", oreStr, argv[i]); return 1; }
            offOreType = t2;
        }
        PATTERN[PATTERN_COUNT][0] = dx;
        PATTERN[PATTERN_COUNT][1] = dy;
        PATTERN[PATTERN_COUNT][2] = dz;
        PATTERN_ORE[PATTERN_COUNT] = oreSlot(offOreType);
        PATTERN_COUNT++;
    }
    #undef PARSE_ORE
    if (PATTERN_COUNT == 0) { fprintf(stderr,"No offsets\n"); return 1; }

    // --- Pick the search anchor: the offset whose ore type is cheapest to
    //     generate. Anchoring on the rarest ore shrinks the Phase-2 candidate
    //     set. We then re-base every offset so the anchor sits at (0,0,0); the
    //     true match centre is reconstructed per-transform in Phase 2.
    int anchorP = 0;
    if (forcedAnchorOre == -2) {
        anchorP = 0; // legacy: first offset is the anchor
    } else if (forcedAnchorOre >= 0) {
        // force a specific ore type; pick the first offset using it, else first offset
        anchorP = 0;
        for (int p = 0; p < PATTERN_COUNT; p++)
            if (ACTIVE_ORES[PATTERN_ORE[p]] == forcedAnchorOre) { anchorP = p; break; }
    } else {
        double anchorBestCost = 1e18;
        for (int p = 0; p < PATTERN_COUNT; p++) {
            double c = oreAnchorCost(ACTIVE_ORES[PATTERN_ORE[p]]);
            if (c < anchorBestCost) { anchorBestCost = c; anchorP = p; }
        }
    }
    int ANCHOR_SLOT = PATTERN_ORE[anchorP];
    // The anchor's original (centre-relative) offset, and its 8 transforms,
    // used to reconstruct centre C = anchorBlock - T_t(anchorOff).
    int anchorOff[3] = { PATTERN[anchorP][0], PATTERN[anchorP][1], PATTERN[anchorP][2] };
    int ANCHOR_OFF_T[8][3];
    for (int t = 0; t < 8; t++)
        applyTransform(t, anchorOff[0], anchorOff[1], anchorOff[2],
                       &ANCHOR_OFF_T[t][0], &ANCHOR_OFF_T[t][1], &ANCHOR_OFF_T[t][2]);
    // Re-base offsets so the anchor is at the origin.
    for (int p = 0; p < PATTERN_COUNT; p++) {
        PATTERN[p][0] -= anchorOff[0];
        PATTERN[p][1] -= anchorOff[1];
        PATTERN[p][2] -= anchorOff[2];
    }

    // Precompute sin() lookups matching float precision
    initSinTable();

    // Precompute all 8 orientations of the (re-based) pattern
    for (int t = 0; t < 8; t++) {
        for (int p = 0; p < PATTERN_COUNT; p++) {
            int ox, oy, oz;
            applyTransform(t, PATTERN[p][0], PATTERN[p][1], PATTERN[p][2], &ox, &oy, &oz);
            TRANSFORMED_PATTERN[t][p][0] = ox;
            TRANSFORMED_PATTERN[t][p][1] = oy;
            TRANSFORMED_PATTERN[t][p][2] = oz;
        }
    }

    int tStart = (orientArg < 0) ? 0 : orientArg;
    int tEnd   = (orientArg < 0) ? 7 : orientArg;

    u64 wseed = (u64)WORLD_SEED;
    initPopSeedMults(wseed);

    int cx0 = centX >> 4, cz0 = centZ >> 4;
    int chunkRadius = (radiusBlks + 15) >> 4;

    int oreBleedChunks = 2;
    int patReach = patternReachBlocks();
    int patReachChunks = (patReach + 15) >> 4;
    int haloChunks = oreBleedChunks + patReachChunks;

    // Core slab size
    int slabHalf = 128;

    printf("Ore pattern matcher (C) -- grid mode\n");
    printf("Seed: %lld  Centre: (%d,%d,%d)  Radius: %d blocks\n",
           (long long)WORLD_SEED, centX, centY, centZ, radiusBlks);
    printf("Pattern: %d offsets, orientations %d..%d, reach %d blocks\n",
           PATTERN_COUNT, tStart, tEnd, patReach);
    static const char *ORE_NAMES[] = {
        "iron","coal","copper","gold","redstone","lapis","diamond",
        "iron_upper","iron_middle","iron_small",
        "coal_upper","coal_lower",
        "gold_normal","gold_lower",
        "redstone_n","redstone_lower",
        "diamond_n","diamond_m","diamond_lg","diamond_bur",
        "lapis_n","lapis_bur",
        "copper_lg",
        "all"
    };
    printf("Ore type: %s\n", ORE_NAMES[oreType]);
    printf("Halo: %d chunks (%d bleed + %d pattern reach)\n",
           haloChunks, oreBleedChunks, patReachChunks);
    printf("Slab size: %d x %d chunks, threads: %d\n\n",
           slabHalf, slabHalf, nthreads);

    // Compute memory dimensions of the flat bitset once at startup
    long long maxW = (long long)slabHalf + 2LL * haloChunks;
    long long range_blocks_max = maxW * 16;
    g_bitset_words = ((size_t)range_blocks_max * 384 * range_blocks_max + 63) / 64;

    // Exact matches only
    int MIN_SCORE = PATTERN_COUNT;
    int bestScore = 0;
    long long rSq = (long long)radiusBlks * radiusBlks;

    // Collect all intersecting slabs into a sequential array
    int totalSlabs_max = ((2 * chunkRadius) / slabHalf + 2) * ((2 * chunkRadius) / slabHalf + 2);
    SlabTask *tasks = malloc(sizeof(SlabTask) * totalSlabs_max);
    int totalSlabs = 0;

    for (int scz = -chunkRadius; scz <= chunkRadius; scz += slabHalf) {
        for (int scx = -chunkRadius; scx <= chunkRadius; scx += slabHalf) {
            int cxA = cx0 + scx;
            int cxB = cxA + slabHalf - 1;
            if (cxB > cx0 + chunkRadius) cxB = cx0 + chunkRadius;

            int czA = cz0 + scz;
            int czB = czA + slabHalf - 1;
            if (czB > cz0 + chunkRadius) czB = cz0 + chunkRadius;

            if (cxA > cxB || czA > czB) continue;

            long long minX = (long long)cxA * 16;
            long long maxX = (long long)(cxB + 1) * 16 - 1;
            long long minZ = (long long)czA * 16;
            long long maxZ = (long long)(czB + 1) * 16 - 1;

            long long dx = (centX < minX) ? (minX - centX) : ((centX > maxX) ? (centX - maxX) : 0);
            long long dz = (centZ < minZ) ? (minZ - centZ) : ((centZ > maxZ) ? (centZ - maxZ) : 0);
            if (dx*dx + dz*dz <= rSq) {
                // Calculate physical center-of-slab coordinates for sorting
                double midX = (cxA + cxB + 1.0) * 8.0;
                double midZ = (czA + czB + 1.0) * 8.0;
                double diffX = midX - centX;
                double diffZ = midZ - centZ;
                double distSq = diffX*diffX + diffZ*diffZ;

                tasks[totalSlabs++] = (SlabTask){ cxA, cxB, czA, czB, distSq };
            }
        }
    }

    // Sort the slabs so the search moves from center outward
    qsort(tasks, totalSlabs, sizeof(SlabTask), compareTasks);

    printf("Total valid slabs: %d\n", totalSlabs);
    printf("Ore types in pattern: %d, memory per thread: %.1f MB\n\n",
           ACTIVE_ORE_COUNT, (double)g_bitset_words * ACTIVE_ORE_COUNT * sizeof(u64) / 1024.0 / 1024.0);
    fflush(stdout);

    omp_init_lock(&g_matchLock);
    g_matchFile = fopen("matches.txt", "w");
    if (g_matchFile) {
        fprintf(g_matchFile, "--- MATCH LOG ---\n");
        fprintf(g_matchFile, "Seed: %lld\nCenter: (%d,%d,%d)\nRadius: %d\n\n",
                (long long)WORLD_SEED, centX, centY, centZ, radiusBlks);
        fflush(g_matchFile);
    } else {
        fprintf(stderr, "warning: could not open matches.txt\n");
    }

    int processedSlabs = 0;

    double total_gen_time = 0;
    double total_match_time = 0;

    double global_start_time = omp_get_wtime();

    #pragma omp parallel
    {
        // Allocate one bitset per active ore type
        ThreadState ts;
        for (int o = 0; o < ACTIVE_ORE_COUNT; o++)
            ts.bitsets[o] = malloc(g_bitset_words * sizeof(u64));
        ts.keys   = NULL;
        ts.nkeys  = 0;
        ts.kcap   = 0;
        ts.collectKeys = 1;

        double thread_gen_time = 0;
        double thread_match_time = 0;

        #pragma omp for schedule(dynamic, 1) reduction(max:bestScore)
        for (int s = 0; s < totalSlabs; s++) {
            SlabTask task = tasks[s];

            int H = haloChunks;
            int cxA_gen = task.cxA - H;
            int cxB_gen = task.cxB + H;
            int czA_gen = task.czA - H;
            int czB_gen = task.czB + H;

            int min_bx = cxA_gen * 16;
            int min_bz = czA_gen * 16;
            int range_x = (cxB_gen - cxA_gen + 1) * 16;
            int range_z = (czB_gen - czA_gen + 1) * 16;

            int core_min_x = task.cxA * 16;
            int core_max_x = (task.cxB + 1) * 16 - 1;
            int core_min_z = task.czA * 16;
            int core_max_z = (task.czB + 1) * 16 - 1;

            size_t active_words = ((size_t)range_x * 384 * range_z + 63) / 64;
            for (int o = 0; o < ACTIVE_ORE_COUNT; o++)
                memset(ts.bitsets[o], 0, active_words * sizeof(u64));
            ts.nkeys = 0;

            double t0 = omp_get_wtime();

            // Phase 1: Generate each active ore type into its own bitset
            for (int cz = czA_gen; cz <= czB_gen; cz++) {
                for (int cx = cxA_gen; cx <= cxB_gen; cx++) {
                    long long _nx = (long long)cx*16 - centX;
                    if (_nx<0) _nx=-_nx; _nx-=16; if (_nx<0) _nx=0;
                    long long _nz = (long long)cz*16 - centZ;
                    if (_nz<0) _nz=-_nz; _nz-=16; if (_nz<0) _nz=0;
                    long long _rg = (long long)radiusBlks+(long long)haloChunks*16;
                    if (_nx*_nx+_nz*_nz > _rg*_rg) continue;

                    for (int o = 0; o < ACTIVE_ORE_COUNT; o++) {
                        ts.collectKeys = (o == ANCHOR_SLOT); // only iterate anchor-ore blocks
                        generateChunkOres(&ts, ts.bitsets[o], wseed, cx, cz, ACTIVE_ORES[o],
                                          min_bx, min_bz, range_x, range_z,
                                          core_min_x, core_max_x, core_min_z, core_max_z);
                    }
                }
            }

            double t1 = omp_get_wtime();
            thread_gen_time += (t1 - t0);

            // Phase 2: Pattern match. Keys are anchor-ore blocks in the slab core.
            // For each anchor block B and transform t, the rebased pattern is
            // tested from B (scorePattern uses B as origin). The true match centre
            // is C = B - T_t(anchorOff); the radius filter applies to C.
            for (u32 ki = 0; ki < ts.nkeys; ki++) {
                u64 key = ts.keys[ki];
                int x = (int)((key >> 34) & 0xFFFFFFULL) - 8388608 + centX;
                int z = (int)((key >> 10) & 0xFFFFFFULL) - 8388608 + centZ;
                int y = (int)(key & 0x3FFULL) - 64;

                for (int t = tStart; t <= tEnd; t++) {
                    int cX = x - ANCHOR_OFF_T[t][0];
                    int cY = y - ANCHOR_OFF_T[t][1];
                    int cZ = z - ANCHOR_OFF_T[t][2];
                    long long dcx = cX - centX, dcz = cZ - centZ;
                    if (dcx*dcx + dcz*dcz > rSq) continue;

                    int score = scorePattern(&ts, x, y, z, t, MIN_SCORE, min_bx, min_bz, range_x, range_z);
                    if (score >= MIN_SCORE) {
                        // Write the full match to the file (locked); keep stdout for
                        // progress only so high match volume can't drop lines.
                        omp_set_lock(&g_matchLock);
                        g_matchCount++;
                        if (g_matchFile)
                            fprintf(g_matchFile,
                                    "MATCH score=%d/%d transform=%d centre=(%d,%d,%d)\n",
                                    score, PATTERN_COUNT, t, cX, cY, cZ);
                        omp_unset_lock(&g_matchLock);
                        if (score > bestScore) bestScore = score;
                    }
                }
            }

            double t3 = omp_get_wtime();
            thread_match_time += (t3 - t1);

            #pragma omp critical
            {
                processedSlabs++;
                printf("Slab %d/%d done  core=[%d,%d]x[%d,%d] chunk  best=%d/%d\n",
                       processedSlabs, totalSlabs, task.cxA, task.cxB, task.czA, task.czB, bestScore, PATTERN_COUNT);
                fflush(stdout);
            }
        }

        #pragma omp critical
        {
            total_gen_time   += thread_gen_time;
            total_match_time += thread_match_time;
        }

        for (int o = 0; o < ACTIVE_ORE_COUNT; o++)
            free(ts.bitsets[o]);
        free(ts.keys);
    }

    double global_end_time = omp_get_wtime();

    if (g_matchFile) fclose(g_matchFile);
    omp_destroy_lock(&g_matchLock);

    printf("\n--- Performance Breakdown (Average CPU-seconds per thread) ---\n");
    printf("Ore Generation: %.3f s\n", total_gen_time / nthreads);
    printf("Pattern Match:  %.3f s\n", total_match_time / nthreads);
    printf("Total Wall-Clock Time: %.3f s\n", global_end_time - global_start_time);
    printf("Matches found: %lld (written to matches.txt)\n", g_matchCount);
    printf("All slabs done. Best score: %d/%d\n", bestScore, PATTERN_COUNT);

    free(tasks);
    return 0;
}