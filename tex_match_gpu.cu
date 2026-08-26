/*
 * tex_match_gpu.cu  --  CUDA texture-rotation coordinate finder (26.1.2+)
 *
 * Two execution paths:
 *
 *  FIELD path (default, ~10x less hashing than brute force):
 *    The rotation value at an absolute position (x,y,z) is shared by every
 *    candidate/facing/observation that maps onto it.  So per tile we:
 *      Kernel A: compute a 4-bit "rotation nibble" (top 4 bits of the
 *                Java-Random next(31) output) once per absolute cell --
 *                this is the information-theoretic minimum of hashing.
 *      Kernel B: match candidates as shifted 64-bit AND sieves over the
 *                nibble field: 16 candidates per word, one add + two loads
 *                + a few logic ops per observation, with early exit.
 *    Supports mod in {2,4,8,16} and modeff (mirrored-variant matching).
 *
 *  NAIVE path (--naive, or auto-fallback for exotic mod values):
 *    The original per-candidate brute-force kernel, kept as a verification
 *    oracle.  Now also honours modeff (matching tex_match.c semantics).
 *
 *  Tiles are processed in expanding rings (Chebyshev distance from the
 *  search centre), so close matches are found first.  --stop N halts once
 *  N matches are confirmed closer than every unsearched cell.
 *
 * Usage:
 *   tex_match_gpu <cx> <cy> <cz> <radius> [--ymin N] [--ymax N]
 *                 [--facing 0,1,...|all4|all] [--stop N] [--naive]
 *                 dx,dy,dz,rot,mod[,modeff] ...
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <limits.h>
#include <math.h>
#include <algorithm>
#include <chrono>
#include <vector>

/* ---------- hash constants ---------- */
#define K2      116129781LL
#define A_SQ    42317861LL
#define LCG_M   0x5DEECE66DLL
#define MASK48  ((1LL<<48)-1)

#define CK(e) do{ cudaError_t _e=(e); if(_e){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(_e)); \
    exit(1); } }while(0)

/* ---------- device hash (naive path + reference) ---------- */
__device__ __forceinline__ int tex_rot(int x, int y, int z, int mod) {
    /* x multiply is 32-bit in Java (imul then i2l) */
    long long l = (long long)(int)(x * 3129871) ^ ((long long)z * K2) ^ (long long)y;
    l = l * l * A_SQ + l * 11LL;
    long long cr = l >> 16;  /* arithmetic shift, matches Java lshr */
    /* SingleThreadedRandomSource.setSeed then nextInt(mod), mod is power of 2 */
    long long stored = (cr ^ LCG_M) & MASK48;
    long long s1 = (stored * LCG_M + 0xBLL) & MASK48;
    int bits31 = (int)(s1 >> 17);
    return (int)(((long long)mod * (long long)bits31) >> 31);
}

/* ---------- constant memory ---------- */
#define MAX_OBS 64
#define MAX_FACINGS 8
#define MAX_FI (MAX_OBS * MAX_FACINGS)

/* naive path */
__constant__ int c_dx[MAX_OBS], c_dy[MAX_OBS], c_dz[MAX_OBS];
__constant__ int c_mod[MAX_OBS];
__constant__ unsigned c_amask[MAX_OBS];  /* bit v set = variant v accepted */

/* field path: per (enabled-facing, obs) pair, flattened fi = f*n_obs + oi */
__constant__ long long          c_K[MAX_FI];    /* word offset into field   */
__constant__ int                c_sh[MAX_FI];   /* sub-word shift (0..60)   */
__constant__ unsigned long long c_val[MAX_FI];  /* replicated compare value */
__constant__ unsigned long long c_msk[MAX_FI];  /* replicated nibble mask   */
__constant__ int                c_face_id[MAX_FACINGS];

/* multi-cube mask observations (obs index >= n_simple): the allowed value
 * set is an OR of subcube terms; a cell matches if ANY term matches.
 * Terms are packed unreplicated: low 8 bits = compare value (cell bits),
 * high 8 bits = fixed-bit mask; replication is one multiply in-kernel. */
#define MAX_TERMS 2048
__constant__ int            c_xoff[MAX_FI];
__constant__ int            c_xcnt[MAX_FI];
__constant__ unsigned short c_tvm[MAX_TERMS];

/* ---------- result buffer ---------- */
#define MAX_RESULTS 500000
struct MatchResult { int x, y, z, facing; };

/* =====================================================================
 * FIELD PATH
 * ===================================================================== */

/* Kernel A: build the rotation field for one tile.
 * B = bits stored per cell (top B bits of next(31); B=4 covers mod 16,
 * B=2 mod 4, B=1 mod 2).  One thread per 64-bit word (64/B consecutive
 * x cells) at fixed (z,y).
 * Layout: field[((y_idx*fz)+z_idx)*fxw + word_idx], group k = cell x0+k. */
template <int B>
__global__ void field_kernel(
    unsigned long long* __restrict__ field,
    long long fx0, int fz0, int fy0,
    int fxw, int fz, long long total)
{
    const int C = 64 / B;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int w  = (int)(idx % fxw);
    long long t = idx / fxw;
    int zi = (int)(t % fz);
    int y  = fy0 + (int)(t / fz);

    long long zq = (long long)(fz0 + zi) * K2;
    int x = (int)(fx0 + (long long)w * C);
    unsigned int X = (unsigned int)x * 3129871u;

    unsigned long long word = 0;
    /* Unroll the whole word, not a fixed 16: C is 64/B (32 cells at B=2), so a
     * hardcoded 16 left half the loop rolled and blocked the X strength
     * reduction across the boundary.  Full unroll measured ~11% faster. */
    #pragma unroll
    for (int k = 0; k < C; k++) {
        long long h = (long long)(int)X ^ zq ^ (long long)y;
        long long l = h * h * A_SQ + h * 11LL;
        long long cr = l >> 16;
        unsigned long long stored = ((unsigned long long)cr ^ (unsigned long long)LCG_M)
                                    & (unsigned long long)MASK48;
        unsigned long long s1 = (stored * (unsigned long long)LCG_M + 0xBULL)
                                    & (unsigned long long)MASK48;
        word |= ((s1 >> (48 - B)) & ((1ULL << B) - 1)) << (B * k);
        X += 3129871u;
    }
    field[idx] = word;
}

/* nibble/group-wise mismatch -> clear the group's flag bit (bit B*k) */
template <int B>
__device__ __forceinline__ unsigned long long group_mismatch(unsigned long long e) {
    if (B == 1) return e;
    unsigned long long t = e | (e >> 1);
    if (B == 4) t |= t >> 2;
    return t;
}

/* Kernel B: sieve candidates against the field.
 * One thread per (64/B-candidate word column, z row); loops y and facings.
 * acc holds one flag bit per group (bit B*k = candidate x0+k still alive). */
template <int B, bool HAS_COMPLEX>
__global__ void sieve_kernel(
    const unsigned long long* __restrict__ field,
    long long fzfxw, int fxw,
    int gx0, int gz0,
    int ix_n, int iz_n,
    int y0, int y1,
    int n_obs, int n_simple, int n_f,
    MatchResult* results, unsigned int* result_count, unsigned int max_results)
{
    const int C = 64 / B;
    const unsigned long long FLAGS =
        (B == 4) ? 0x1111111111111111ULL :
        (B == 2) ? 0x5555555555555555ULL : ~0ULL;

    int wxn = (ix_n + C - 1) / C;
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (long long)wxn * iz_n) return;
    int wx = (int)(tid % wxn);
    int zi = (int)(tid / wxn);
    int gx = gx0 + wx * C;

    /* validity mask: flag bits only for cells inside the tile interior */
    int rem = ix_n - wx * C;
    unsigned long long vmask = FLAGS;
    if (rem < C) vmask >>= B * (C - rem);

    for (int y = y0; y <= y1; y++) {
        long long ybase = (long long)(y - y0) * fzfxw + (long long)zi * fxw + wx;
        for (int f = 0; f < n_f; f++) {
            unsigned long long acc = vmask;
            int base_fi = f * n_obs;
            /* process obs in pairs: two independent load/logic chains in
             * flight per alive-check halves the exposed latency */
            int oi = 0;
            for (; oi + 1 < n_simple; oi += 2) {
                int fia = base_fi + oi, fib = base_fi + oi + 1;
                long long aa = ybase + c_K[fia], ab = ybase + c_K[fib];
                unsigned long long a0 = field[aa], a1 = field[aa + 1];
                unsigned long long b0 = field[ab], b1 = field[ab + 1];
                int sha = c_sh[fia], shb = c_sh[fib];
                unsigned long long wa = sha ? ((a0 >> sha) | (a1 << (64 - sha))) : a0;
                unsigned long long wb = shb ? ((b0 >> shb) | (b1 << (64 - shb))) : b0;
                unsigned long long ea = (wa ^ c_val[fia]) & c_msk[fia];
                unsigned long long eb = (wb ^ c_val[fib]) & c_msk[fib];
                acc &= ~group_mismatch<B>(ea) & ~group_mismatch<B>(eb);
                if (!acc) break;
            }
            if (acc && oi < n_simple) {
                int fi = base_fi + oi;
                long long a = ybase + c_K[fi];
                unsigned long long w0 = field[a];
                unsigned long long w1 = field[a + 1];
                int sh = c_sh[fi];
                unsigned long long w = sh ? ((w0 >> sh) | (w1 << (64 - sh))) : w0;
                unsigned long long e = (w ^ c_val[fi]) & c_msk[fi];
                acc &= ~group_mismatch<B>(e);
            }
            /* multi-cube mask observations: word loaded once, then one
             * xor/and/fold per subcube term; mismatch only if EVERY term
             * mismatches (terms are ORed alternatives) */
            if (HAS_COMPLEX)
            for (int ox = n_simple; ox < n_obs && acc; ox++) {
                int fi = base_fi + ox;
                long long a = ybase + c_K[fi];
                unsigned long long w0 = field[a];
                unsigned long long w1 = field[a + 1];
                int sh = c_sh[fi];
                unsigned long long w = sh ? ((w0 >> sh) | (w1 << (64 - sh))) : w0;
                int t0 = c_xoff[fi], tn = c_xcnt[fi];
                unsigned long long mm = ~0ULL;
                for (int t = 0; t < tn; t++) {
                    unsigned short tvm = c_tvm[t0 + t];
                    unsigned long long tv = FLAGS * (unsigned long long)(tvm & 0xFF);
                    unsigned long long tm = FLAGS * (unsigned long long)(tvm >> 8);
                    mm &= group_mismatch<B>((w ^ tv) & tm);
                }
                acc &= ~mm;
            }
            while (acc) {
                int b = __ffsll(acc) - 1;
                acc &= acc - 1;
                unsigned int slot = atomicAdd(result_count, 1u);
                if (slot < max_results)
                    results[slot] = { gx + b / B, y, gz0 + zi, c_face_id[f] };
            }
        }
    }
}

/* =====================================================================
 * NAIVE PATH (verification oracle / exotic-mod fallback)
 * ===================================================================== */
__global__ void tex_kernel(
    int x_min, int z_min,
    long long nx, long long nz,
    long long slab_offset,
    long long slab_size,
    int oy_min, int oy_max,
    int n_obs,
    int facing_mask,
    MatchResult* results,
    unsigned int* result_count,
    unsigned int max_results)
{
    long long local_idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (local_idx >= slab_size) return;

    long long global_idx = slab_offset + local_idx;
    int ox = x_min + (int)(global_idx % nx);
    int oz = z_min + (int)(global_idx / nx);

    for (int oy = oy_min; oy <= oy_max; oy++) {
        for (int f = 0; f < 8; f++) {
            if (!(facing_mask & (1 << f))) continue;
            int ok = 1;
            for (int i = 0; i < n_obs; i++) {
                int dx = c_dx[i], dz = c_dz[i];
                int rdx, rdz;
                switch (f) {
                    case 0: rdx =  dx; rdz =  dz; break;
                    case 1: rdx = -dz; rdz =  dx; break;
                    case 2: rdx = -dx; rdz = -dz; break;
                    case 3: rdx =  dz; rdz = -dx; break;
                    case 4: rdx = -dx; rdz =  dz; break;
                    case 5: rdx =  dz; rdz =  dx; break;
                    case 6: rdx =  dx; rdz = -dz; break;
                    default:rdx = -dz; rdz = -dx; break;
                }
                int got = tex_rot(ox + rdx, oy + c_dy[i], oz + rdz, c_mod[i]);
                if (!((c_amask[i] >> got) & 1u)) { ok = 0; break; }
            }
            if (ok) {
                unsigned int slot = atomicAdd(result_count, 1u);
                if (slot < max_results)
                    results[slot] = {ox, oy, oz, f};
            }
        }
    }
}

/* =====================================================================
 * Host
 * ===================================================================== */

typedef struct {
    int dx, dy, dz, rot, mod, modeff;
    int is_mask;          /* 1 = allowed-set mask observation */
    unsigned mask;        /* bit v set = variant value v accepted */
} Obs;
static Obs obs_raw[MAX_OBS];
static int n_obs = 0;
static int facing_mask = 0x01;

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s <cx> <cy> <cz> <radius> [--ymin N] [--ymax N]\n"
        "       [--facing 0,1,...|all4|all] [--stop N] [--naive] [--view-relative]\n"
        "       dx,dy,dz,rot,mod[,modeff] | dx,dy,dz,mHEX,mod ...\n"
        "  mHEX: allowed-set mask observation (bit v = variant v accepted),\n"
        "       e.g. m0807,16 accepts variants {0,1,2,11} of a mod-16 block\n"
        "       (netherrack reads from the screenshot extractor)\n"
        "  --view-relative: rot values were read relative to the camera (vanilla\n"
        "       textures, no pack; a +90 CW camera turn reads rot one higher).\n"
        "       Facings 0-3 = camera turned CW by f (game convention);\n"
        "       5-7 = opposite spin. Obs must be mod=4 (or mod=16 masks,\n"
        "       rotated via their orientation layout).\n", prog);
    exit(1);
}

static int popcount32(unsigned v) {
    int c = 0;
    while (v) { v &= v - 1; c++; }
    return c;
}

/* mask equivalent of the legacy (rot, modeff) check */
static unsigned legacy_mask(int rot, int mod, int modeff) {
    unsigned m = 0;
    for (int v = 0; v < mod; v++)
        if ((v & (modeff - 1)) == rot) m |= 1u << v;
    return m;
}

/* Permute a mask under a view-relative yaw of rf*90 degrees.
 * mod=4 : pure y-rotation variant lists (grass etc): v' = (v+rf)&3.
 * mod=16: full-orientation lists (netherrack): index v = 4*yi+xi
 *         (x rotations inner, y outer); yaw shifts yi only. */
static unsigned mask_yaw(unsigned mask, int mod, int rf) {
    unsigned out = 0;
    for (int v = 0; v < mod; v++) {
        if (!(mask & (1u << v))) continue;
        int vp = (mod == 16) ? ((((v >> 2) + rf) & 3) << 2) | (v & 3)
                             : (v + rf) & (mod - 1);
        out |= 1u << vp;
    }
    return out;
}

/* Decompose an allowed-value mask over `bits` bits into subcube terms
 * (msk, val): value v matches a term iff (v & msk) == val.  Greedy
 * largest-cube-first cover.  Returns term count, fills terms[][2]. */
static int cube_cover(unsigned mask, int bits, unsigned terms[][2], int max_terms) {
    unsigned full = (bits >= 32) ? ~0u : ((1u << (1 << bits)) - 1);
    (void)full;
    int n = 1 << bits;
    unsigned remaining = mask;
    int cnt = 0;
    while (remaining) {
        int best_size = -1; unsigned best_m = 0, best_v = 0;
        for (unsigned m = 0; m < (unsigned)n; m++) {       /* fixed-bit mask */
            unsigned freebits = (~m) & (n - 1);
            int size = 1 << popcount32(freebits);
            if (size <= best_size) continue;
            for (unsigned v = 0; v < (unsigned)n; v++) {
                if (v & freebits) continue;                /* canonical val */
                /* cube = {v | any subset of freebits}; must lie in mask and
                 * cover at least one remaining value */
                int inside = 1, covers_new = 0;
                for (unsigned f2 = freebits; ; f2 = (f2 - 1) & freebits) {
                    unsigned x = v | f2;
                    if (!(mask & (1u << x))) { inside = 0; break; }
                    if (remaining & (1u << x)) covers_new = 1;
                    if (f2 == 0) break;
                }
                if (inside && covers_new) {
                    best_size = size; best_m = m; best_v = v;
                    break;          /* any val at this size is fine */
                }
            }
        }
        if (best_size < 0) return -1;   /* cannot happen for nonzero mask */
        if (cnt >= max_terms) return -1;
        terms[cnt][0] = best_m; terms[cnt][1] = best_v; cnt++;
        for (unsigned f2 = (~best_m) & (n - 1); ; f2 = (f2 - 1) & ((~best_m) & (n - 1))) {
            remaining &= ~(1u << (best_v | f2));
            if (f2 == 0) break;
        }
    }
    return cnt;
}

static double wall_sec() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

/* Sort comparator: by Chebyshev distance from centre, then Y */
static int g_cx, g_cz;
static int n_simple_g = 0;   /* obs 0..n_simple-1 are single-subcube */
static bool cmp_dist(const MatchResult &a, const MatchResult &b) {
    int da = std::max(abs(a.x - g_cx), abs(a.z - g_cz));
    int db = std::max(abs(b.x - g_cx), abs(b.z - g_cz));
    if (da != db) return da < db;
    return a.y < b.y;
}

static void rotate_off(int dx, int dz, int f, int *rdx, int *rdz) {
    switch (f & 7) {
        case 0: *rdx =  dx; *rdz =  dz; break;
        case 1: *rdx = -dz; *rdz =  dx; break;
        case 2: *rdx = -dx; *rdz = -dz; break;
        case 3: *rdx =  dz; *rdz = -dx; break;
        case 4: *rdx = -dx; *rdz =  dz; break;
        case 5: *rdx =  dz; *rdz =  dx; break;
        case 6: *rdx =  dx; *rdz = -dz; break;
        default:*rdx = -dz; *rdz = -dx; break;
    }
}

static int ilog2_pow2(int v) {  /* returns -1 if not a power of two >= 1 */
    if (v < 1 || (v & (v - 1))) return -1;
    int b = 0; while (v > 1) { v >>= 1; b++; }
    return b;
}

/* result accumulator */
static MatchResult *all_results = NULL;
static size_t alloc_cap = 0, all_count = 0;
static long long dropped = 0;

static void append_results(const MatchResult *src, size_t n) {
    if (all_count + n > alloc_cap) {
        while (all_count + n > alloc_cap) alloc_cap = alloc_cap ? alloc_cap * 2 : 65536;
        all_results = (MatchResult*)realloc(all_results, alloc_cap * sizeof(MatchResult));
        if (!all_results) { fprintf(stderr, "out of host memory for results\n"); exit(1); }
    }
    memcpy(all_results + all_count, src, n * sizeof(MatchResult));
    all_count += n;
}

int main(int argc, char **argv) {
    if (argc < 6) usage(argv[0]);

    int cx     = atoi(argv[1]);
    int cy     = atoi(argv[2]);
    int cz     = atoi(argv[3]);
    int radius = atoi(argv[4]);

    int ymin_override = INT_MAX, ymax_override = INT_MAX;
    long long stop_n = 0;
    int force_naive = 0;
    int view_rel = 0;

    for (int i = 5; i < argc; i++) {
        if (!strcmp(argv[i],"--ymin") && i+1<argc)      { ymin_override=atoi(argv[++i]); }
        else if (!strcmp(argv[i],"--ymax") && i+1<argc) { ymax_override=atoi(argv[++i]); }
        else if (!strcmp(argv[i],"--stop") && i+1<argc) { stop_n=atoll(argv[++i]); }
        else if (!strcmp(argv[i],"--naive"))            { force_naive=1; }
        else if (!strcmp(argv[i],"--view-relative"))    { view_rel=1; }
        else if (!strcmp(argv[i],"--facing") && i+1<argc) {
            ++i;
            if (!strcmp(argv[i],"all"))       facing_mask=0xFF;
            else if (!strcmp(argv[i],"all4")) facing_mask=0x0F;
            else {
                facing_mask=0;
                char *tok=strtok(argv[i],",");
                while(tok){ int f=atoi(tok); if(f<0||f>7){fprintf(stderr,"--facing indices must be 0-7\n");exit(1);} facing_mask|=(1<<f); tok=strtok(NULL,","); }
                if(!facing_mask){fprintf(stderr,"--facing: no valid indices\n");exit(1);}
            }
        } else {
            if (n_obs>=MAX_OBS){fprintf(stderr,"Too many observations\n");exit(1);}
            int dx,dy,dz,rot,mod,modeff;
            unsigned mask;
            if (sscanf(argv[i],"%d,%d,%d,m%x,%d",&dx,&dy,&dz,&mask,&mod) == 5) {
                if (mod < 2 || mod > 32 || (mod & (mod-1)) ||
                    mask == 0 || (mod < 32 && mask >> mod)) {
                    fprintf(stderr,"Bad mask obs '%s': mod must be a power of 2 "
                            "(2..32) and mask must fit in mod bits\n",argv[i]);
                    exit(1);
                }
                obs_raw[n_obs++]={dx,dy,dz,0,mod,mod,1,mask};
                continue;
            }
            int nf = sscanf(argv[i],"%d,%d,%d,%d,%d,%d",&dx,&dy,&dz,&rot,&mod,&modeff);
            if (nf == 5) modeff = mod;
            else if (nf != 6){fprintf(stderr,"Bad obs '%s'\n",argv[i]);exit(1);}
            obs_raw[n_obs++]={dx,dy,dz,rot,mod,modeff,0,legacy_mask(rot,mod,modeff)};
        }
    }
    if (n_obs==0){fprintf(stderr,"No observations.\n");exit(1);}

    /* view-relative mode: rot values were read off vanilla textures relative
     * to the (unknown) camera direction.  Each facing hypothesis must shift
     * every rot by the camera rotation in lockstep with the offset rotation.
     * Facings 0-3 = game convention (camera CW by f, rot-f), 5-7 = opposite
     * spin (4 would duplicate 0).  Only for mod=4 (pure-rotation variants). */
    if (view_rel) {
        for (int i = 0; i < n_obs; i++)
            if (obs_raw[i].mod != 4 && !(obs_raw[i].is_mask && obs_raw[i].mod == 16)) {
                fprintf(stderr, "--view-relative requires mod=4 observations "
                        "(or mod=16 mask observations)\n");
                exit(1);
            }
        if (force_naive) {
            fprintf(stderr, "--view-relative is not supported with --naive (use tex_match.exe as oracle)\n");
            exit(1);
        }
        if (facing_mask & 0x10) {
            facing_mask &= ~0x10;
            fprintf(stderr, "note: facing 4 duplicates facing 0 in view-relative mode; dropped\n");
        }
        if (!facing_mask) { fprintf(stderr, "--view-relative: no facings left\n"); exit(1); }
    }

    /* sort observations most-selective first (lowest pass fraction rejects
     * more); ties keep input order */
    std::stable_sort(obs_raw, obs_raw + n_obs, [](const Obs &a, const Obs &b){
        return (double)popcount32(a.mask) * b.mod <
               (double)popcount32(b.mask) * a.mod;
    });
    /* partition: single-subcube observations first (the tuned sieve pair
     * loop), multi-cube mask observations after (term-loop tail).  The
     * effective set under view-relative yaw has the same cube structure
     * for legacy obs; for mask obs check every enabled facing. */
    {
        Obs simple[MAX_OBS], complexo[MAX_OBS];
        int ns = 0, nc = 0;
        for (int i = 0; i < n_obs; i++) {
            int lm = ilog2_pow2(obs_raw[i].mod);
            int is_simple = 1;
            if (lm >= 1) {
                for (int f = 0; f < 8 && is_simple; f++) {
                    if (!(facing_mask & (1 << f))) continue;
                    int rf = !view_rel ? 0 : (f < 4) ? ((4 - f) & 3) : (f - 4);
                    unsigned em = obs_raw[i].is_mask
                        ? (view_rel ? mask_yaw(obs_raw[i].mask, obs_raw[i].mod, rf)
                                    : obs_raw[i].mask)
                        : legacy_mask(view_rel ? ((obs_raw[i].rot + rf) & (obs_raw[i].modeff - 1))
                                               : obs_raw[i].rot,
                                      obs_raw[i].mod, obs_raw[i].modeff);
                    unsigned terms[32][2];
                    if (cube_cover(em, lm, terms, 32) != 1) is_simple = 0;
                }
            }
            if (is_simple) simple[ns++] = obs_raw[i];
            else           complexo[nc++] = obs_raw[i];
        }
        for (int i = 0; i < ns; i++) obs_raw[i] = simple[i];
        for (int i = 0; i < nc; i++) obs_raw[ns + i] = complexo[i];
        n_simple_g = ns;
    }

    int x_min = cx - radius, x_max = cx + radius;
    int z_min = cz - radius, z_max = cz + radius;
    int y_min = (ymin_override!=INT_MAX) ? ymin_override : cy;
    int y_max = (ymax_override!=INT_MAX) ? ymax_override : cy;
    if (y_min > y_max) { int t = y_min; y_min = y_max; y_max = t; }

    long long nx = (long long)(x_max - x_min + 1);
    long long nz = (long long)(z_max - z_min + 1);
    int ny = y_max - y_min + 1;
    long long total_xz = nx * nz;
    int n_f = 0; { int m = facing_mask; while(m){ n_f += m&1; m>>=1; } }
    long long total_cells = total_xz * ny * n_f;
    /* Cells actually scanned. Equals total_cells unless --stop exits early
     * (field path), in which case the final throughput must use this, not
     * total_cells, or it credits the whole radius to a partial scan. */
    long long cells_searched = total_cells;

    g_cx = cx; g_cz = cz;

    /* decide path: field needs mod in {2,4,8,16}, modeff pow2 <= mod */
    int use_field = !force_naive;
    int halo = 0;
    for (int i = 0; i < n_obs; i++) {
        int lm = ilog2_pow2(obs_raw[i].mod);
        int le = ilog2_pow2(obs_raw[i].modeff);
        if (lm < 1 || lm > 4 || le < 0 || le > lm) use_field = 0;
        int h = std::max(abs(obs_raw[i].dx), abs(obs_raw[i].dz));
        if (h > halo) halo = h;
    }
    if (halo > 512) use_field = 0;
    if (!use_field && !force_naive)
        fprintf(stderr, "note: observation set not field-eligible (mod must be 2/4/8/16, offsets <=512); using naive kernel\n");

    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop,0));
    printf("GPU: %s\n", prop.name);
    printf("Texture rotation search: centre (%d,%d,%d) radius %d  [%s path]%s\n",
           cx, cy, cz, radius, use_field ? "field" : "naive",
           view_rel ? "  [view-relative: facings 0-3 = game convention, 5-7 = opposite spin]" : "");
    char fstr[64]={0};
    for(int f=0;f<8;f++) if(facing_mask&(1<<f)){char tmp[4];snprintf(tmp,sizeof(tmp),"%d,",f);strcat(fstr,tmp);}
    if(fstr[0]) fstr[strlen(fstr)-1]='\0';
    printf("  XZ [%d..%d]x[%d..%d], Y [%d..%d], %d obs, facing={%s}\n",
           x_min, x_max, z_min, z_max, y_min, y_max, n_obs, fstr);
    printf("  Total cells: %lld\n", total_cells);
    fflush(stdout);

    MatchResult *d_results;
    unsigned int *d_count;
    CK(cudaMalloc(&d_results, MAX_RESULTS * sizeof(MatchResult)));
    CK(cudaMalloc(&d_count,   sizeof(unsigned int)));
    CK(cudaMemset(d_count, 0, sizeof(unsigned int)));

    MatchResult *h_batch = (MatchResult*)malloc(MAX_RESULTS * sizeof(MatchResult));
    FILE *match_file = fopen("matches.txt","w");

    double t0 = wall_sec();
    int stopped_early = 0;
    long long frontier_dist = -1;   /* all cells closer than this are searched */

    if (use_field) {
        /* ---------------- FIELD PATH ---------------- */
        /* B = bits stored per field cell = widest observation's bit count */
        int B = 1;
        for (int i = 0; i < n_obs; i++)
            B = std::max(B, ilog2_pow2(obs_raw[i].mod));
        if (B == 3) B = 4;          /* mod 8: store nibbles */
        int C = 64 / B;             /* candidates per 64-bit word */

        /* 4096 over 2048: the field carries a halo of `halo` z-rows and HX x-cells
         * on every side, so per-tile waste falls from ~5.9% to ~2.9% of the words
         * generated.  Field generation is ~78% of runtime, so that halo shrink is
         * worth ~4%.  Costs ~334 MB instead of ~86 MB; the loop below halves the
         * tile if the allocation does not fit. */
        /* Tile size trades two costs against each other:
         *   too large -> the per-tile halo is generated for a region we barely
         *                use (at radius 500 a 4096 tile builds ~25x the words
         *                it reads), and one giant tile cannot overlap with the
         *                next, so the launch tail is exposed;
         *   too small -> the halo is a larger fraction of every tile.
         * Measured optimum tracks ~half the search span, capped at 4096:
         *   r=500 -> 512/1024, r=1000 -> 1024, r=2000..4000 -> 2048, r>=8000 -> 4096.
         * That keeps at least ~4 tiles in flight while holding halo waste near
         * its floor. */
        int TILE_X = 4096, TILE_Z = 4096;
        { long long hx = (nx + 1) / 2, hz = (nz + 1) / 2;
          if (hx < TILE_X) TILE_X = (int)hx;
          if (hz < TILE_Z) TILE_Z = (int)hz; }
        { const char *e = getenv("TEX_TILE_X"); if (e) TILE_X = atoi(e);
          e = getenv("TEX_TILE_Z"); if (e) TILE_Z = atoi(e); }
        if (TILE_X < 64) TILE_X = 64;
        TILE_X = (TILE_X + 63) & ~63;
        if (TILE_Z < 1) TILE_Z = 1;

        int HX = (halo + C - 1) & ~(C - 1);
        int dy_min = obs_raw[0].dy, dy_max = obs_raw[0].dy;
        for (int i = 1; i < n_obs; i++) {
            dy_min = std::min(dy_min, obs_raw[i].dy);
            dy_max = std::max(dy_max, obs_raw[i].dy);
        }
        int fy0 = y_min + dy_min;
        int fy  = (y_max + dy_max) - fy0 + 1;

        /* allocate the field; shrink the tile if it does not fit */
        int fxw, fz;
        long long fzfxw, field_words;
        unsigned long long *d_field = NULL;
        for (;;) {
            fxw = (2 * HX + TILE_X) / C + 1;
            fz  = TILE_Z + 2 * halo;
            fzfxw = (long long)fz * fxw;
            field_words = fzfxw * fy;
            if (cudaMalloc(&d_field, field_words * sizeof(unsigned long long)) == cudaSuccess)
                break;
            cudaGetLastError();   /* clear the failed-alloc error */
            if (TILE_Z > 64)      TILE_Z /= 2;
            else if (TILE_X > 64) TILE_X /= 2;
            else { fprintf(stderr, "cannot allocate field buffer\n"); exit(1); }
        }

        /* per-(facing,obs) constants */
        int face_ids[MAX_FACINGS]; int nfi = 0;
        for (int f = 0; f < 8; f++) if (facing_mask & (1<<f)) face_ids[nfi++] = f;
        CK(cudaMemcpyToSymbol(c_face_id, face_ids, sizeof(int)*nfi));

        long long h_K[MAX_FI]; int h_sh[MAX_FI];
        unsigned long long h_val[MAX_FI], h_msk[MAX_FI];
        int h_xoff[MAX_FI], h_xcnt[MAX_FI];
        unsigned short h_tvm[MAX_TERMS];
        int n_terms = 0;
        for (int fo = 0; fo < nfi; fo++) {
            int fid = face_ids[fo];
            /* view-relative: a +90 (CW) camera turn reads rot one HIGHER, so
             * world rot = read rot - camera turn.  Facings 0-3 = transform f
             * with rot-f (the real-game convention); 5-7 = opposite spin. */
            int tf = (view_rel && fid >= 4) ? fid - 4 : fid;
            int rf = !view_rel ? 0 : (fid < 4) ? ((4 - fid) & 3) : (fid - 4);
            for (int oi = 0; oi < n_obs; oi++) {
                int rdx, rdz;
                rotate_off(obs_raw[oi].dx, obs_raw[oi].dz, tf, &rdx, &rdz);
                int fi = fo * n_obs + oi;
                h_K[fi]  = (long long)(obs_raw[oi].dy - dy_min) * fzfxw
                         + (long long)(halo + rdz) * fxw
                         + (HX + rdx) / C;
                h_sh[fi] = ((HX + rdx) % C) * B;
                int lm = ilog2_pow2(obs_raw[oi].mod);
                int s  = B - lm;    /* cell stores top B bits of next(31) */
                /* effective allowed-value mask under this facing */
                unsigned em = obs_raw[oi].is_mask
                    ? (view_rel ? mask_yaw(obs_raw[oi].mask, obs_raw[oi].mod, rf)
                                : obs_raw[oi].mask)
                    : legacy_mask(view_rel ? ((obs_raw[oi].rot + rf) & (obs_raw[oi].modeff - 1))
                                           : obs_raw[oi].rot,
                                  obs_raw[oi].mod, obs_raw[oi].modeff);
                unsigned terms[32][2];
                int tn = cube_cover(em, lm, terms, 32);
                if (tn < 1) { fprintf(stderr, "internal: cube cover failed\n"); exit(1); }
                h_val[fi] = 0; h_msk[fi] = 0;
                h_xoff[fi] = 0; h_xcnt[fi] = 0;
                if (oi < n_simple_g) {
                    /* single subcube: replicated compare (tuned pair loop) */
                    unsigned long long cm = (1ULL << B) - 1;
                    unsigned long long v = ((unsigned long long)terms[0][1] << s) & cm;
                    unsigned long long m = ((unsigned long long)terms[0][0] << s) & cm;
                    unsigned long long rep_v = 0, rep_m = 0;
                    for (int k = 0; k < C; k++) { rep_v |= v << (B*k); rep_m |= m << (B*k); }
                    h_val[fi] = rep_v; h_msk[fi] = rep_m;
                } else {
                    if (n_terms + tn > MAX_TERMS) {
                        fprintf(stderr, "too many mask observation terms (max %d)\n",
                                MAX_TERMS);
                        exit(1);
                    }
                    h_xoff[fi] = n_terms; h_xcnt[fi] = tn;
                    for (int t = 0; t < tn; t++)
                        h_tvm[n_terms++] = (unsigned short)
                            (((terms[t][0] << s) << 8) | (terms[t][1] << s));
                }
            }
        }
        CK(cudaMemcpyToSymbol(c_K,   h_K,   sizeof(long long)*nfi*n_obs));
        CK(cudaMemcpyToSymbol(c_sh,  h_sh,  sizeof(int)*nfi*n_obs));
        CK(cudaMemcpyToSymbol(c_val, h_val, sizeof(unsigned long long)*nfi*n_obs));
        CK(cudaMemcpyToSymbol(c_msk, h_msk, sizeof(unsigned long long)*nfi*n_obs));
        CK(cudaMemcpyToSymbol(c_xoff, h_xoff, sizeof(int)*nfi*n_obs));
        CK(cudaMemcpyToSymbol(c_xcnt, h_xcnt, sizeof(int)*nfi*n_obs));
        if (n_terms)
            CK(cudaMemcpyToSymbol(c_tvm, h_tvm, sizeof(unsigned short)*n_terms));

        /* tile grid; centre tile for ring enumeration */
        long long ntx = (nx + TILE_X - 1) / TILE_X;
        long long ntz = (nz + TILE_Z - 1) / TILE_Z;
        long long total_tiles = ntx * ntz;
        long long ti0 = std::min((long long)std::max(cx - x_min, 0) / TILE_X, ntx - 1);
        long long tj0 = std::min((long long)std::max(cz - z_min, 0) / TILE_Z, ntz - 1);
        long long max_ring = 0;
        max_ring = std::max(max_ring, std::max(ti0, ntx - 1 - ti0));
        max_ring = std::max(max_ring, std::max(tj0, ntz - 1 - tj0));

        long long tiles_done = 0, cells_done = 0;
        unsigned int last_cnt = 0;       /* matches already copied to host */
        double last_print = t0;
        const int DRAIN_TILES = 64;
        int since_drain = 0;
        int bs = 256;

        /* process one tile: enqueue kernels (no sync) */
        auto run_tile = [&](long long ti, long long tj) {
            int gx0 = (int)(x_min + ti * TILE_X);
            int gz0 = (int)(z_min + tj * TILE_Z);
            int ix_n = (int)std::min((long long)TILE_X, (long long)x_max - gx0 + 1);
            int iz_n = (int)std::min((long long)TILE_Z, (long long)z_max - gz0 + 1);
            long long fx0 = (long long)gx0 - HX;
            int fz0 = gz0 - halo;

            long long fthreads = field_words;
            int fblocks = (int)((fthreads + bs - 1) / bs);
            long long sthreads = (long long)((ix_n + C - 1) / C) * iz_n;
            int sblocks = (int)((sthreads + bs - 1) / bs);
            bool cx_obs = (n_simple_g < n_obs);
            switch (B) {
            case 1:
                field_kernel<1><<<fblocks, bs>>>(d_field, fx0, fz0, fy0, fxw, fz, fthreads);
                if (cx_obs) sieve_kernel<1, true><<<sblocks, bs>>>(d_field, fzfxw, fxw, gx0, gz0, ix_n, iz_n,
                    y_min, y_max, n_obs, n_simple_g, nfi, d_results, d_count, (unsigned int)MAX_RESULTS);
                else sieve_kernel<1, false><<<sblocks, bs>>>(d_field, fzfxw, fxw, gx0, gz0, ix_n, iz_n,
                    y_min, y_max, n_obs, n_simple_g, nfi, d_results, d_count, (unsigned int)MAX_RESULTS);
                break;
            case 2:
                field_kernel<2><<<fblocks, bs>>>(d_field, fx0, fz0, fy0, fxw, fz, fthreads);
                if (cx_obs) sieve_kernel<2, true><<<sblocks, bs>>>(d_field, fzfxw, fxw, gx0, gz0, ix_n, iz_n,
                    y_min, y_max, n_obs, n_simple_g, nfi, d_results, d_count, (unsigned int)MAX_RESULTS);
                else sieve_kernel<2, false><<<sblocks, bs>>>(d_field, fzfxw, fxw, gx0, gz0, ix_n, iz_n,
                    y_min, y_max, n_obs, n_simple_g, nfi, d_results, d_count, (unsigned int)MAX_RESULTS);
                break;
            default:
                field_kernel<4><<<fblocks, bs>>>(d_field, fx0, fz0, fy0, fxw, fz, fthreads);
                if (cx_obs) sieve_kernel<4, true><<<sblocks, bs>>>(d_field, fzfxw, fxw, gx0, gz0, ix_n, iz_n,
                    y_min, y_max, n_obs, n_simple_g, nfi, d_results, d_count, (unsigned int)MAX_RESULTS);
                else sieve_kernel<4, false><<<sblocks, bs>>>(d_field, fzfxw, fxw, gx0, gz0, ix_n, iz_n,
                    y_min, y_max, n_obs, n_simple_g, nfi, d_results, d_count, (unsigned int)MAX_RESULTS);
                break;
            }

            cells_done += (long long)ix_n * iz_n * ny * nfi;
            tiles_done++;
            since_drain++;
        };

        /* drain: sync, copy any new results to host */
        auto drain = [&](void) {
            CK(cudaDeviceSynchronize());
            unsigned int cnt = 0;
            CK(cudaMemcpy(&cnt, d_count, sizeof(unsigned int), cudaMemcpyDeviceToHost));
            unsigned int stored = std::min(cnt, (unsigned int)MAX_RESULTS);
            if (stored > last_cnt) {
                unsigned int n = stored - last_cnt;
                CK(cudaMemcpy(h_batch, d_results + last_cnt,
                              n * sizeof(MatchResult), cudaMemcpyDeviceToHost));
                append_results(h_batch, n);
            }
            if (cnt > MAX_RESULTS) {            /* overflow: reset buffer */
                dropped += (long long)(cnt - MAX_RESULTS);
                CK(cudaMemset(d_count, 0, sizeof(unsigned int)));
                last_cnt = 0;
            } else if (stored > MAX_RESULTS / 2) {  /* keep headroom */
                CK(cudaMemset(d_count, 0, sizeof(unsigned int)));
                last_cnt = 0;
            } else {
                last_cnt = stored;
            }
            since_drain = 0;
            double now = wall_sec();
            if (now - last_print >= 0.5 || tiles_done >= total_tiles) {
                double rate = cells_done / std::max(now - t0, 1e-3);
                long long rem = total_cells - cells_done;
                int eta = (int)(rem / std::max(rate, 1.0));
                printf("Progress: rows %lld/%lld  (%.1f G cells/s, ETA %ds)\n",
                       tiles_done, total_tiles, rate/1e9, eta);
                fflush(stdout);
                last_print = now;
            }
        };

        /* ring walk over the tile grid, expanding from the centre tile */
        for (long long k = 0; k <= max_ring && !stopped_early; k++) {
            long long i_lo = std::max(ti0 - k, 0LL), i_hi = std::min(ti0 + k, ntx - 1);
            long long j_lo = std::max(tj0 - k, 0LL), j_hi = std::min(tj0 + k, ntz - 1);
            if (k == 0) {
                run_tile(ti0, tj0);
            } else {
                if (tj0 - k >= 0)  for (long long i = i_lo; i <= i_hi; i++) { run_tile(i, tj0 - k); if (since_drain >= DRAIN_TILES) drain(); }
                if (tj0 + k < ntz) for (long long i = i_lo; i <= i_hi; i++) { run_tile(i, tj0 + k); if (since_drain >= DRAIN_TILES) drain(); }
                long long jc_lo = std::max(tj0 - k + 1, 0LL), jc_hi = std::min(tj0 + k - 1, ntz - 1);
                if (ti0 - k >= 0)  for (long long j = jc_lo; j <= jc_hi; j++) { run_tile(ti0 - k, j); if (since_drain >= DRAIN_TILES) drain(); }
                if (ti0 + k < ntx) for (long long j = jc_lo; j <= jc_hi; j++) { run_tile(ti0 + k, j); if (since_drain >= DRAIN_TILES) drain(); }
            }
            /* at ring boundaries drain when --stop is active so the frontier
             * check below sees every match found so far */
            if (stop_n > 0 || since_drain >= DRAIN_TILES) drain();
            if (stop_n > 0 && k < max_ring) {
                /* nearest unsearched cell: nearest edge of ring k+1 (axis tiles) */
                long long fd = LLONG_MAX;
                int ax0 = (int)(x_min + ti0 * TILE_X);          /* centre tile x start */
                int az0 = (int)(z_min + tj0 * TILE_Z);
                if (ti0 - (k+1) >= 0)  fd = std::min(fd, (long long)cx - (ax0 - (long long)(k+1)*TILE_X + TILE_X - 1));
                if (ti0 + (k+1) < ntx) fd = std::min(fd, (ax0 + (long long)(k+1)*TILE_X) - cx);
                if (tj0 - (k+1) >= 0)  fd = std::min(fd, (long long)cz - (az0 - (long long)(k+1)*TILE_Z + TILE_Z - 1));
                if (tj0 + (k+1) < ntz) fd = std::min(fd, (az0 + (long long)(k+1)*TILE_Z) - cz);
                if (fd != LLONG_MAX) {
                    long long confirmed = 0;
                    for (size_t i = 0; i < all_count; i++) {
                        long long d = std::max(llabs((long long)all_results[i].x - cx),
                                               llabs((long long)all_results[i].z - cz));
                        if (d < fd) confirmed++;
                    }
                    if (confirmed >= stop_n) {
                        frontier_dist = fd;
                        stopped_early = 1;
                    }
                }
            }
        }
        drain();
        cells_searched = cells_done;   /* < total_cells when --stop exits early */
        CK(cudaFree(d_field));
    } else {
        /* ---------------- NAIVE PATH ---------------- */
        if (stop_n > 0)
            printf("note: --stop is only supported on the field path; searching the whole radius\n");
        int h_dx[MAX_OBS],h_dy[MAX_OBS],h_dz[MAX_OBS],h_mod[MAX_OBS];
        unsigned h_am[MAX_OBS];
        for(int i=0;i<n_obs;i++){
            h_dx[i]=obs_raw[i].dx; h_dy[i]=obs_raw[i].dy; h_dz[i]=obs_raw[i].dz;
            h_mod[i]=obs_raw[i].mod; h_am[i]=obs_raw[i].mask;
        }
        CK(cudaMemcpyToSymbol(c_dx,  h_dx,  sizeof(int)*n_obs));
        CK(cudaMemcpyToSymbol(c_dy,  h_dy,  sizeof(int)*n_obs));
        CK(cudaMemcpyToSymbol(c_dz,  h_dz,  sizeof(int)*n_obs));
        CK(cudaMemcpyToSymbol(c_mod, h_mod, sizeof(int)*n_obs));
        CK(cudaMemcpyToSymbol(c_amask, h_am, sizeof(unsigned)*n_obs));

        long long SLAB = 500000000LL;
        int bs = 256;
        for (long long slab_start = 0; slab_start < total_xz; slab_start += SLAB) {
            long long slab_size = std::min(SLAB, total_xz - slab_start);
            CK(cudaMemset(d_count, 0, sizeof(unsigned int)));
            long long sblocks = (slab_size + bs - 1) / bs;
            tex_kernel<<<(int)sblocks, bs>>>(
                x_min, z_min, nx, nz, slab_start, slab_size,
                y_min, y_max, n_obs, facing_mask,
                d_results, d_count, (unsigned int)MAX_RESULTS);
            CK(cudaDeviceSynchronize());
            unsigned int cnt = 0;
            CK(cudaMemcpy(&cnt, d_count, sizeof(unsigned int), cudaMemcpyDeviceToHost));
            if (cnt > MAX_RESULTS) { dropped += cnt - MAX_RESULTS; cnt = MAX_RESULTS; }
            if (cnt > 0) {
                CK(cudaMemcpy(h_batch, d_results, cnt * sizeof(MatchResult), cudaMemcpyDeviceToHost));
                append_results(h_batch, cnt);
            }
            long long xz_done = slab_start + slab_size;
            double elapsed = wall_sec() - t0;
            double rate = (double)(xz_done * ny * n_f) / std::max(elapsed, 1e-3);
            long long cells_rem = (total_xz - xz_done) * ny * n_f;
            int eta = (int)((double)cells_rem / std::max(rate, 1.0));
            printf("Progress: rows %lld/%lld  (%.1f G cells/s, ETA %ds)\n",
                   xz_done / nz, nx, rate/1e9, eta);
            fflush(stdout);
        }
    }

    /* Sort all results globally by distance and emit */
    std::sort(all_results, all_results + all_count, cmp_dist);
    for (size_t i = 0; i < all_count; i++) {
        char line[160];
        if (n_f > 1)
            snprintf(line, sizeof(line), "MATCH x=%d y=%d z=%d facing=%d\n",
                     all_results[i].x, all_results[i].y, all_results[i].z, all_results[i].facing);
        else
            snprintf(line, sizeof(line), "MATCH x=%d y=%d z=%d\n",
                     all_results[i].x, all_results[i].y, all_results[i].z);
        printf("%s", line);
        if (match_file) fputs(line, match_file);
    }
    fflush(stdout);
    if (match_file) fflush(match_file);

    double elapsed = wall_sec() - t0;
    if (dropped > 0)
        printf("WARNING: result buffer overflowed; %lld match(es) dropped. Add observations or reduce radius.\n", dropped);
    if (stopped_early)
        printf("Stopped early: %lld match(es) confirmed within Chebyshev distance %lld of centre.\n",
               (long long)stop_n, frontier_dist);
    printf("Done. %lld match(es) found. (%.2f s, %.2f G cells/s)\n",
           (long long)all_count + dropped, elapsed,
           (double)cells_searched / std::max(elapsed, 1e-3) / 1e9);

    if (match_file) fclose(match_file);
    CK(cudaFree(d_results));
    CK(cudaFree(d_count));
    free(all_results);
    free(h_batch);
    return 0;
}
