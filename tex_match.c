/*
 * tex_match.c  --  Texture-rotation coordinate finder for Minecraft 26.1.2+
 *
 * Blocks with multiple equal-weight blockstate variants (grass, dirt, stone ...)
 * show a variant determined solely by (x,y,z) via Mth.getSeed + one Java-Random
 * nextInt call.  Top/bottom blocks typically have 4 variants (0-3 = 0/90/180/270
 * deg rotation); side-only blocks have 2 variants (mod 2).
 *
 * Formula (derived from 26.1.2 SingleThreadedRandomSource via jar analysis):
 *   coordRand(x,y,z):                    // Mth.getSeed -- x multiply is 32-bit!
 *     l = ((int32_t)(x*3129871) as int64) ^ (z*116129781LL) ^ y
 *     l = l*l*42317861LL + l*11LL
 *     return l >> 16                      // arithmetic (signed) shift
 *   getTexture(x,y,z,mod):               // setSeed then nextInt(mod), mod power-of-2
 *     stored = (coordRand ^ 0x5DEECE66DLL) & ((1LL<<48)-1)
 *     s1     = (stored * 0x5DEECE66DLL + 0xBLL) & ((1LL<<48)-1)
 *     return (int)((mod * (int64_t)(s1 >> 17)) >> 31)
 *
 * Performance: field+sieve inner loop.  For each (z,y) row we:
 *   1. Build a packed nibble field for the full X range (plus obs dx offsets).
 *      16 cells per uint64_t word, 4 bits each = top 4 bits of next(31).
 *   2. For each candidate word (16 consecutive x values), apply observations as
 *      shift+nibble-lookup sieves, tracking a 16-bit survivor mask.  Exit early
 *      as soon as no candidates survive.
 *   Observations sorted by mask popcount (most constraining first).
 *   Obs with dy!=0 or dz!=0 are checked scalar per survivor after the sieve.
 *
 * Usage:
 *   tex_match <cx> <cy> <cz> <radius> <threads> [--ymin N] [--ymax N]
 *             [--facing 0|1|2|3|all] dx,dy,dz,rot,mod ...
 *
 *   cx/cy/cz  : search centre (block coords)
 *   radius    : half-side of XZ box to search (Y searched +-radius too)
 *   threads   : worker thread count
 *   --ymin/--ymax : override Y search range
 *   --facing  : which cardinal rotation of the offset pattern to try
 *                 0 = as entered (default)
 *                 1 = rotated 90 deg CCW  (dx,dz) -> (-dz, dx)
 *                 2 = rotated 180 deg     (dx,dz) -> (-dx,-dz)
 *                 3 = rotated 270 deg CCW (dx,dz) -> ( dz,-dx)
 *                 all = try all 4, label each match with facing=N
 *   dx,dy,dz  : offset from candidate origin (in the direction you were facing)
 *   rot       : expected texture rotation value (0-3 for tops, 0-1 for sides)
 *   mod       : 4 for top/bottom face, 2 for side face
 *
 * Prints progress to stdout and MATCH lines to both stdout and matches.txt.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <limits.h>
#include <math.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
typedef HANDLE  thread_t;
typedef CRITICAL_SECTION mutex_t;
#define mutex_init(m)   InitializeCriticalSection(m)
#define mutex_lock(m)   EnterCriticalSection(m)
#define mutex_unlock(m) LeaveCriticalSection(m)
#define mutex_destroy(m) DeleteCriticalSection(m)
static DWORD WINAPI thread_func(LPVOID arg);
#define CREATE_THREAD(t,f,a) ((t)=CreateThread(NULL,0,thread_func,(a),0,NULL))
#define JOIN_THREAD(t)       WaitForSingleObject((t),INFINITE)
#else
#include <pthread.h>
typedef pthread_t thread_t;
typedef pthread_mutex_t mutex_t;
#define mutex_init(m)    pthread_mutex_init((m),NULL)
#define mutex_lock(m)    pthread_mutex_lock(m)
#define mutex_unlock(m)  pthread_mutex_unlock(m)
#define mutex_destroy(m) pthread_mutex_destroy(m)
static void* thread_func(void* arg);
#define CREATE_THREAD(t,f,a) pthread_create(&(t),NULL,thread_func,(a))
#define JOIN_THREAD(t)       pthread_join((t),NULL)
#endif

/* ---------- hash ---------- */

static inline int64_t coord_rand(int x, int y, int z) {
    /* x multiply is 32-bit in Java (imul), must truncate before widening */
    int64_t l = (int64_t)(int32_t)(x * 3129871) ^ ((int64_t)z * 116129781LL) ^ (int64_t)y;
    l = l * l * 42317861LL + l * 11LL;
    return l >> 16;  /* arithmetic shift, matches Java lshr */
}

static inline int get_texture(int x, int y, int z, int mod) {
    /* SingleThreadedRandomSource.setSeed(coordRand) then nextInt(mod), mod is power of 2 */
    int64_t cr = coord_rand(x, y, z);
    int64_t stored = (cr ^ (int64_t)0x5DEECE66DLL) & (((int64_t)1 << 48) - 1);
    int64_t s1 = (stored * (int64_t)0x5DEECE66DLL + 0xBLL) & (((int64_t)1 << 48) - 1);
    int bits31 = (int)(s1 >> 17);  /* next(31) */
    return (int)(((int64_t)mod * (int64_t)bits31) >> 31);
}

/* ---------- observation ---------- */

#define MAX_OBS 64

typedef struct {
    int dx, dy, dz, rot, mod, modeff;
    int is_mask;          /* 1 = allowed-set mask observation */
    unsigned mask;        /* bit v set = variant value v accepted */
} Obs;

static Obs obs_raw[MAX_OBS];   /* as entered by user */
static int n_obs = 0;

/* mask equivalent of the legacy (rot, modeff) check */
static unsigned legacy_mask(int rot, int mod, int modeff) {
    unsigned m = 0;
    for (int v = 0; v < mod; v++)
        if ((v & (modeff - 1)) == rot) m |= 1u << v;
    return m;
}

/* Permute a mask under a view-relative yaw of rf*90 degrees.
 * mod=4 : pure y-rotation variant lists (grass etc): v' = (v+rf)&3.
 * mod=16: full-orientation lists (netherrack): index v = 4*yi+xi with
 *         yi = v>>2, xi = v&3 (blockstate order: x rotations inner, y
 *         outer); yaw shifts yi only: v' = 4*((yi+rf)&3) + xi. */
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

/* view-relative mode: rot values were read off vanilla textures relative to
 * the unknown camera direction; each facing hypothesis shifts rot in lockstep
 * with the offset rotation.  Facings 0-3 = game convention (camera CW by f,
 * rot-f), 5-7 = opposite spin. */
static int view_rel = 0;

/* Apply one of 8 transforms to (dx,dz).
 * 0-3: rotations only (CCW).  4-7: mirror-X then rotate.
 *   0: ( dx,  dz)   1: (-dz,  dx)   2: (-dx, -dz)   3: ( dz, -dx)
 *   4: (-dx,  dz)   5: ( dz,  dx)   6: ( dx, -dz)   7: (-dz, -dx)
 * In view-relative mode 4-7 are instead rotations 1-3 with the opposite
 * spin convention (facings 0-3 use the game convention, rot-f). */
static void rotate_obs(const Obs *src, Obs *dst, int n, int facing) {
    int tf = (view_rel && facing >= 4) ? facing - 4 : facing;
    int rf = !view_rel ? 0 : (facing < 4) ? ((4 - facing) & 3) : (facing - 4);
    for (int i = 0; i < n; i++) {
        dst[i] = src[i];
        if (view_rel) {
            if (src[i].is_mask) {
                dst[i].mask = mask_yaw(src[i].mask, src[i].mod, rf);
            } else {
                dst[i].rot = (src[i].rot + rf) & (src[i].modeff - 1);
                dst[i].mask = legacy_mask(dst[i].rot, src[i].mod, src[i].modeff);
            }
        }
        int dx = src[i].dx, dz = src[i].dz;
        switch (tf & 7) {
            case 0: dst[i].dx =  dx; dst[i].dz =  dz; break;
            case 1: dst[i].dx = -dz; dst[i].dz =  dx; break;
            case 2: dst[i].dx = -dx; dst[i].dz = -dz; break;
            case 3: dst[i].dx =  dz; dst[i].dz = -dx; break;
            case 4: dst[i].dx = -dx; dst[i].dz =  dz; break;
            case 5: dst[i].dx =  dz; dst[i].dz =  dx; break;
            case 6: dst[i].dx =  dx; dst[i].dz = -dz; break;
            case 7: dst[i].dx = -dz; dst[i].dz = -dx; break;
        }
    }
}

/* ---------- packed-field sieve ----------
 *
 * We pack 16 x-consecutive cells into one uint64_t (4 bits per cell).
 * Each nibble = top 4 bits of next(31) for that cell.
 *
 * mod-2  result = nibble >> 3
 * mod-4  result = nibble >> 2
 * mod-16 result = nibble
 *
 * For each observation (same dy,dz as the row being built) we:
 *   1. Find the field word covering the obs x position (candidate x + dx).
 *   2. Stitch two adjacent words if obs_dx shifts us across a word boundary.
 *   3. For each lane k check: is nibble[k] in the obs allowed set?
 *      Encode the result as a 16-bit survivor mask (bit k = lane k still alive).
 *
 * Observations with dy!=0 or dz!=0 cannot use the same field row;
 * surviving candidates after the flat-obs sieve are checked scalar for those.
 */

#define CELLS_PER_WORD 16   /* 64-bit / 4-bit nibbles */

/* Convert an obs allowed-variant mask into a nibble-accept bitmask.
 * Bit n set means raw nibble n produces an accepted variant.
 *   mod=16: variant = nibble      (shift 0)
 *   mod=4:  variant = nibble >> 2 (shift 2)
 *   mod=2:  variant = nibble >> 3 (shift 3) */
static uint16_t obs_nibble_accept(const Obs *o) {
    uint16_t acc = 0;
    int shift = (o->mod == 16) ? 0 : (o->mod == 4) ? 2 : 3;
    for (int n = 0; n < 16; n++) {
        int variant = n >> shift;
        if (variant < o->mod && ((o->mask >> variant) & 1u))
            acc |= (uint16_t)1 << n;
    }
    return acc;
}

/* Build a packed nibble row for absolute x in [field_x0, field_x0 + n_words*16),
 * at fixed absolute (y, z).  The top 4 bits of next(31) are stored per cell. */
static void build_field_row(uint64_t *field, int field_x0, int n_words, int y, int z) {
    int64_t zq = (int64_t)z * 116129781LL;
    unsigned Xbase = (unsigned)field_x0 * 3129871u;
    for (int w = 0; w < n_words; w++) {
        uint64_t word = 0;
        unsigned X = Xbase + (unsigned)(w * CELLS_PER_WORD) * 3129871u;
        for (int k = 0; k < CELLS_PER_WORD; k++) {
            int64_t h = (int64_t)(int32_t)X ^ zq ^ (int64_t)y;
            int64_t l = h * h * 42317861LL + h * 11LL;
            uint64_t stored = ((uint64_t)(l >> 16) ^ (uint64_t)0x5DEECE66DLL)
                              & (((uint64_t)1 << 48) - 1);
            uint64_t s1 = (stored * (uint64_t)0x5DEECE66DLL + 0xBULL)
                          & (((uint64_t)1 << 48) - 1);
            word |= (uint64_t)((s1 >> 44) & 0xF) << (4 * k);
            X += 3129871u;
        }
        field[w] = word;
    }
}

/* Precomputed lookup: for a given nibble_accept mask and a packed byte (two
 * nibbles), returns a 2-bit accept result (bit 0 = lo nibble ok, bit 1 = hi).
 * Indexed by [nibble_accept_index][byte_value].  We store one table per unique
 * nibble_accept value encountered; populated lazily at search start. */
#define MAX_ACCEPT_TABLES 64
static uint8_t accept_table[MAX_ACCEPT_TABLES][256];
static uint16_t accept_table_key[MAX_ACCEPT_TABLES];
static int n_accept_tables = 0;

static int get_accept_table(uint16_t nibble_accept) {
    for (int i = 0; i < n_accept_tables; i++)
        if (accept_table_key[i] == nibble_accept) return i;
    int idx = n_accept_tables++;
    accept_table_key[idx] = nibble_accept;
    for (int b = 0; b < 256; b++) {
        int lo = b & 0xF, hi = b >> 4;
        accept_table[idx][b] = (uint8_t)(
            (((nibble_accept >> lo) & 1u)) |
            (((nibble_accept >> hi) & 1u) << 1));
    }
    return idx;
}

/* Check one 16-lane word from the field against a nibble_accept mask.
 * Returns a 16-bit mask: bit k set iff lane k's nibble is accepted.
 * Uses a byte lookup table: each byte holds two nibbles, giving 2 result bits. */
static inline uint16_t sieve_word(uint64_t field_word, const uint8_t *tbl) {
    uint16_t result = 0;
    /* 8 bytes per word, 2 lanes per byte */
    for (int b = 0; b < 8; b++) {
        uint8_t byte = (uint8_t)(field_word >> (8 * b));
        result |= (uint16_t)tbl[byte] << (2 * b);
    }
    return result;
}

/* ---------- search state ---------- */

static int cx, cy, cz, radius;
static int n_threads;
static int x_min, x_max;
static int z_min, z_max;
static int y_min, y_max;

/* facing_mask: bitmask of which of the 8 transforms to try. */
static int facing_mask = 0x01;

static mutex_t file_mutex;
static FILE   *match_file = NULL;
static volatile long match_count = 0;
static volatile long rows_done   = 0;

/* ---------- per-facing precomputed data ---------- */

#define MAX_FACINGS 8

/* Rotated + rarity-sorted observations per facing. */
static Obs obs_rot[MAX_FACINGS][MAX_OBS];

/* Nibble-accept bitmask for each (facing, obs) -- kept for table construction. */
static uint16_t obs_nibble[MAX_FACINGS][MAX_OBS];
/* Accept table index for each (facing, obs). */
static int obs_tbl[MAX_FACINGS][MAX_OBS];

/* For each facing: how many obs share the same (dy,dz) as obs[0] and
 * can be sieves against one field row (n_flat), and how many need a
 * separate scalar check (n_scalar = n_obs - n_flat).
 * The flat obs are stored first in obs_rot[f][0..n_flat-1]. */
static int facing_n_flat[MAX_FACINGS];   /* obs that share same (dy,dz) */
static int facing_use_sieve[MAX_FACINGS];/* 1 if all flat obs have mod in {2,4,16} */

static int cmp_obs_by_rarity(const void *a, const void *b) {
    const Obs *oa = (const Obs *)a;
    const Obs *ob = (const Obs *)b;
    return __builtin_popcount(oa->mask) - __builtin_popcount(ob->mask);
}

/* ---------- worker ---------- */

typedef struct { int z0, z1; } Range;

#define MAX_FIELD_WORDS 131072  /* covers X radius up to ~1M with obs offsets */

#ifdef _WIN32
static DWORD WINAPI thread_func(LPVOID arg)
#else
static void* thread_func(void* arg)
#endif
{
    Range *r = (Range*)arg;
    int multi = __builtin_popcount(facing_mask) > 1;

    /* Compute field X extent: must cover [x_min + min_dx, x_max + max_dx]
     * for all enabled facings, rounded outward to CELLS_PER_WORD. */
    int global_min_dx = 0, global_max_dx = 0;
    for (int f = 0; f < MAX_FACINGS; f++) {
        if (!(facing_mask & (1 << f))) continue;
        for (int i = 0; i < facing_n_flat[f]; i++) {
            if (obs_rot[f][i].dx < global_min_dx) global_min_dx = obs_rot[f][i].dx;
            if (obs_rot[f][i].dx > global_max_dx) global_max_dx = obs_rot[f][i].dx;
        }
    }
    /* Align field_x0 down to CELLS_PER_WORD boundary */
    int field_x0 = (x_min + global_min_dx) & ~(CELLS_PER_WORD - 1);
    int field_x1_incl = x_max + global_max_dx;
    int field_n_words = (field_x1_incl - field_x0) / CELLS_PER_WORD + 1;
    if (field_n_words > MAX_FIELD_WORDS) field_n_words = MAX_FIELD_WORDS;

    /* Per-thread field buffer (one row at a time). */
    uint64_t *field = (uint64_t *)malloc(field_n_words * sizeof(uint64_t));
    char line[160];

    /* Outer loops: (z, y), then per-facing sieve over the full X range.
     * This amortizes field construction across all x candidates. */
    for (int oz = r->z0; oz <= r->z1; oz++) {
        for (int oy = y_min; oy <= y_max; oy++) {

            for (int f = 0; f < MAX_FACINGS; f++) {
                if (!(facing_mask & (1 << f))) continue;
                int nf = facing_n_flat[f];
                int ns = n_obs - nf;  /* scalar-only obs count */

                if (facing_use_sieve[f] && nf > 0) {
                    /* Flat obs all share (dy0, dz0). */
                    int dy0 = obs_rot[f][0].dy;
                    int dz0 = obs_rot[f][0].dz;
                    int abs_y = oy + dy0;
                    int abs_z = oz + dz0;

                    /* Build the field row for this (abs_y, abs_z). */
                    build_field_row(field, field_x0, field_n_words, abs_y, abs_z);

                    /* Process 16 candidates at a time across X. */
                    int x_word0 = x_min & ~(CELLS_PER_WORD - 1);
                    int x_word1 = x_max;
                    for (int wx = x_word0; wx <= x_word1; wx += CELLS_PER_WORD) {
                        /* 16-bit survivor mask for lanes 0..15 */
                        uint16_t surv = 0xFFFF;

                        /* Sieve each flat obs */
                        for (int i = 0; i < nf && surv; i++) {
                            /* Obs x for lane 0 of this word: wx + obs_rot[f][i].dx */
                            int obs_ax0 = wx + obs_rot[f][i].dx;
                            int obs_w = (obs_ax0 - field_x0) / CELLS_PER_WORD;
                            int obs_sh = (obs_ax0 - field_x0) % CELLS_PER_WORD;
                            /* Handle negative modulo */
                            if (obs_sh < 0) { obs_sh += CELLS_PER_WORD; obs_w--; }

                            if (obs_w < 0 || obs_w >= field_n_words) {
                                /* Entire word outside field -- scalar per live lane */
                                uint16_t new_surv = 0;
                                uint16_t s = surv;
                                while (s) {
                                    int k = __builtin_ctz(s);
                                    s &= s - 1;
                                    int ox = wx + k;
                                    if (ox >= x_min && ox <= x_max) {
                                        int got = get_texture(ox + obs_rot[f][i].dx,
                                                              abs_y, abs_z,
                                                              obs_rot[f][i].mod);
                                        if ((obs_rot[f][i].mask >> got) & 1u)
                                            new_surv |= (uint16_t)1 << k;
                                    }
                                }
                                surv = new_surv;
                                continue;
                            }

                            uint64_t fw;
                            if (obs_sh == 0) {
                                fw = field[obs_w];
                            } else {
                                uint64_t lo = field[obs_w];
                                uint64_t hi = (obs_w + 1 < field_n_words) ? field[obs_w + 1] : 0ULL;
                                int bsh = obs_sh * 4;
                                fw = (lo >> bsh) | (hi << (64 - bsh));
                            }
                            surv &= sieve_word(fw, accept_table[obs_tbl[f][i]]);
                        }

                        if (!surv) continue;

                        /* Mask off lanes that fall outside [x_min, x_max] */
                        if (wx < x_min) {
                            int skip = x_min - wx;
                            surv &= (uint16_t)(0xFFFF << skip);
                        }
                        if (wx + CELLS_PER_WORD - 1 > x_max) {
                            int keep = x_max - wx + 1;
                            surv &= (uint16_t)((1u << keep) - 1u);
                        }
                        if (!surv) continue;

                        /* For each survivor, run scalar check for off-plane obs */
                        uint16_t s = surv;
                        while (s) {
                            int k = __builtin_ctz(s);
                            s &= s - 1;
                            int ox = wx + k;

                            /* Scalar obs (different dy/dz) */
                            int ok = 1;
                            for (int i = nf; i < n_obs && ok; i++) {
                                int got = get_texture(ox + obs_rot[f][i].dx,
                                                      oy + obs_rot[f][i].dy,
                                                      oz + obs_rot[f][i].dz,
                                                      obs_rot[f][i].mod);
                                if (!((obs_rot[f][i].mask >> got) & 1u)) ok = 0;
                            }
                            if (!ok) continue;

                            /* Emit match */
                            if (multi)
                                snprintf(line, sizeof(line), "MATCH x=%d y=%d z=%d facing=%d\n", ox, oy, oz, f);
                            else
                                snprintf(line, sizeof(line), "MATCH x=%d y=%d z=%d\n", ox, oy, oz);
                            mutex_lock(&file_mutex);
                            match_count++;
                            printf("%s", line);
                            fflush(stdout);
                            if (match_file) { fputs(line, match_file); fflush(match_file); }
                            mutex_unlock(&file_mutex);
                        }
                    }

                } else {
                    /* Scalar fallback: exotic mod or no flat obs */
                    for (int ox = x_min; ox <= x_max; ox++) {
                        int ok = 1;
                        for (int i = 0; i < n_obs && ok; i++) {
                            int got = get_texture(ox + obs_rot[f][i].dx,
                                                  oy + obs_rot[f][i].dy,
                                                  oz + obs_rot[f][i].dz,
                                                  obs_rot[f][i].mod);
                            if (!((obs_rot[f][i].mask >> got) & 1u)) ok = 0;
                        }
                        if (!ok) continue;
                        if (multi)
                            snprintf(line, sizeof(line), "MATCH x=%d y=%d z=%d facing=%d\n", ox, oy, oz, f);
                        else
                            snprintf(line, sizeof(line), "MATCH x=%d y=%d z=%d\n", ox, oy, oz);
                        mutex_lock(&file_mutex);
                        match_count++;
                        printf("%s", line);
                        fflush(stdout);
                        if (match_file) { fputs(line, match_file); fflush(match_file); }
                        mutex_unlock(&file_mutex);
                    }
                }
            }
        }
        mutex_lock(&file_mutex);
        rows_done++;
        mutex_unlock(&file_mutex);
    }

    free(field);
#ifdef _WIN32
    return 0;
#else
    return NULL;
#endif
}

/* ---------- main ---------- */

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s <cx> <cy> <cz> <radius> <threads>\n"
        "       [--ymin N] [--ymax N] [--facing 0,1,...|all4|all]\n"
        "       dx,dy,dz,rot,mod[,modeff] | dx,dy,dz,mHEX,mod ...\n"
        "  mHEX: allowed-set mask observation (bit v = variant v accepted),\n"
        "        e.g. m0807,16 accepts variants {0,1,2,11} of a mod-16 block\n"
        "        (netherrack top-face reads from the screenshot extractor)\n"
        "  --facing 0,1,2,3   comma-separated transform indices (0-7)\n"
        "  --facing all4      all 4 rotations (no mirrors)\n"
        "  --facing all       all 8 transforms (4 rotations + 4 mirrored)\n"
        "  Transforms: 0=(dx,dz) 1=(-dz,dx) 2=(-dx,-dz) 3=(dz,-dx)\n"
        "              4=(-dx,dz) 5=(dz,dx) 6=(dx,-dz) 7=(-dz,-dx)\n"
        "  --view-relative    rot values were read relative to the camera\n"
        "       (vanilla textures, no pack; +90 CW turn reads rot one\n"
        "       higher); facings 0-3 = camera turned CW by f (game\n"
        "       convention); 5-7 = opposite spin; obs must be mod=4\n"
        "       (or mod=16 masks, rotated via their orientation layout)\n",
        prog);
    exit(1);
}

int main(int argc, char **argv) {
    if (argc < 7) usage(argv[0]);

    cx      = atoi(argv[1]);
    cy      = atoi(argv[2]);
    cz      = atoi(argv[3]);
    radius  = atoi(argv[4]);
    n_threads = atoi(argv[5]);
    if (n_threads < 1) n_threads = 1;

    int ymin_override = INT32_MAX, ymax_override = INT32_MAX;

    for (int i = 6; i < argc; i++) {
        if (strcmp(argv[i], "--ymin") == 0 && i+1 < argc) {
            ymin_override = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--ymax") == 0 && i+1 < argc) {
            ymax_override = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--view-relative") == 0) {
            view_rel = 1;
        } else if (strcmp(argv[i], "--facing") == 0 && i+1 < argc) {
            ++i;
            if (strcmp(argv[i], "all") == 0) {
                facing_mask = 0xFF;
            } else if (strcmp(argv[i], "all4") == 0) {
                facing_mask = 0x0F;
            } else {
                facing_mask = 0;
                char *tok = strtok(argv[i], ",");
                while (tok) {
                    int f = atoi(tok);
                    if (f < 0 || f > 7) {
                        fprintf(stderr, "--facing indices must be 0-7\n"); exit(1);
                    }
                    facing_mask |= (1 << f);
                    tok = strtok(NULL, ",");
                }
                if (!facing_mask) { fprintf(stderr, "--facing: no valid indices\n"); exit(1); }
            }
        } else {
            if (n_obs >= MAX_OBS) { fprintf(stderr, "Too many observations (max %d)\n", MAX_OBS); exit(1); }
            int dx,dy,dz,rot,mod,modeff;
            unsigned mask;
            if (sscanf(argv[i], "%d,%d,%d,m%x,%d", &dx,&dy,&dz,&mask,&mod) == 5) {
                if (mod < 2 || mod > 32 || (mod & (mod-1)) ||
                    mask == 0 || (mod < 32 && mask >> mod)) {
                    fprintf(stderr, "Bad mask observation '%s': mod must be a "
                            "power of 2 (2..32) and mask must fit in mod bits\n",
                            argv[i]);
                    exit(1);
                }
                obs_raw[n_obs++] = (Obs){dx,dy,dz,0,mod,mod,1,mask};
                continue;
            }
            int nf = sscanf(argv[i], "%d,%d,%d,%d,%d,%d", &dx,&dy,&dz,&rot,&mod,&modeff);
            if (nf == 5) modeff = mod;
            else if (nf != 6) {
                fprintf(stderr, "Bad observation '%s': expected "
                        "dx,dy,dz,rot,mod[,modeff] or dx,dy,dz,mHEX,mod\n", argv[i]);
                exit(1);
            }
            obs_raw[n_obs++] = (Obs){dx,dy,dz,rot,mod,modeff,0,
                                     legacy_mask(rot,mod,modeff)};
        }
    }
    if (n_obs == 0) { fprintf(stderr, "No observations given.\n"); exit(1); }

    if (view_rel) {
        for (int i = 0; i < n_obs; i++)
            if (obs_raw[i].mod != 4 && !(obs_raw[i].is_mask && obs_raw[i].mod == 16)) {
                fprintf(stderr, "--view-relative requires mod=4 observations "
                        "(or mod=16 mask observations)\n");
                exit(1);
            }
        if (facing_mask & 0x10) {
            facing_mask &= ~0x10;
            fprintf(stderr, "note: facing 4 duplicates facing 0 in view-relative mode; dropped\n");
        }
        if (!facing_mask) { fprintf(stderr, "--view-relative: no facings left\n"); exit(1); }
    }

    x_min = cx - radius;  x_max = cx + radius;
    z_min = cz - radius;  z_max = cz + radius;
    y_min = (ymin_override != INT32_MAX) ? ymin_override : cy;
    y_max = (ymax_override != INT32_MAX) ? ymax_override : cy;

    int total_x = x_max - x_min + 1;
    int total_z = z_max - z_min + 1;
    int total_y = y_max - y_min + 1;
    long long total_cells = (long long)total_x * total_z * total_y;

    /* Precompute per-facing data. */
    for (int f = 0; f < MAX_FACINGS; f++) {
        rotate_obs(obs_raw, obs_rot[f], n_obs, f);

        /* Sort by rarity (most constraining first) -- helps early exit in both
         * scalar path and the obs loop inside the sieve. */
        qsort(obs_rot[f], n_obs, sizeof(Obs), cmp_obs_by_rarity);

        /* Partition: flat obs (all sharing the same dy,dz as obs[0]) first,
         * then obs with different dy/dz that need scalar checks. */
        if (n_obs > 0) {
            int ref_dy = obs_rot[f][0].dy, ref_dz = obs_rot[f][0].dz;
            /* Stable partition: move non-flat obs to the end. */
            Obs tmp[MAX_OBS];
            int nflat = 0, nscalar = 0;
            Obs flat_buf[MAX_OBS], scalar_buf[MAX_OBS];
            for (int i = 0; i < n_obs; i++) {
                if (obs_rot[f][i].dy == ref_dy && obs_rot[f][i].dz == ref_dz)
                    flat_buf[nflat++] = obs_rot[f][i];
                else
                    scalar_buf[nscalar++] = obs_rot[f][i];
            }
            (void)tmp;
            for (int i = 0; i < nflat; i++) obs_rot[f][i] = flat_buf[i];
            for (int i = 0; i < nscalar; i++) obs_rot[f][nflat + i] = scalar_buf[i];
            facing_n_flat[f] = nflat;
        } else {
            facing_n_flat[f] = 0;
        }

        /* Build nibble-accept tables for flat obs. */
        for (int i = 0; i < facing_n_flat[f]; i++) {
            obs_nibble[f][i] = obs_nibble_accept(&obs_rot[f][i]);
            obs_tbl[f][i] = get_accept_table(obs_nibble[f][i]);
        }

        /* Can all flat obs use the sieve? Requires mod in {2,4,16}. */
        facing_use_sieve[f] = 1;
        for (int i = 0; i < facing_n_flat[f]; i++) {
            int m = obs_rot[f][i].mod;
            if (m != 2 && m != 4 && m != 16) { facing_use_sieve[f] = 0; break; }
        }
    }

    char facing_str[64] = {0};
    for (int f = 0; f < MAX_FACINGS; f++)
        if (facing_mask & (1 << f)) { char tmp[4]; snprintf(tmp,sizeof(tmp),"%d,",f); strcat(facing_str,tmp); }
    if (facing_str[0]) facing_str[strlen(facing_str)-1] = '\0';
    printf("Texture rotation search: centre (%d,%d,%d) radius %d\n", cx, cy, cz, radius);
    printf("  X [%d..%d], Z [%d..%d], Y [%d..%d], %d obs, facing={%s}, %d thread(s)\n",
           x_min, x_max, z_min, z_max, y_min, y_max, n_obs, facing_str, n_threads);
    printf("  Total cells: %lld\n", total_cells);
    fflush(stdout);

    match_file = fopen("matches.txt", "w");
    mutex_init(&file_mutex);

    /* Split Z range across threads (outer loop is Z now). */
    Range *ranges = (Range*)malloc(n_threads * sizeof(Range));
    thread_t *threads = (thread_t*)malloc(n_threads * sizeof(thread_t));

    int per = total_z / n_threads;
    int rem = total_z % n_threads;
    int cur = z_min;
    for (int i = 0; i < n_threads; i++) {
        ranges[i].z0 = cur;
        int len = per + (i < rem ? 1 : 0);
        ranges[i].z1 = cur + len - 1;
        cur += len;
        CREATE_THREAD(threads[i], thread_func, &ranges[i]);
    }

    /* Progress reporter in main thread */
    long prev = 0;
    while (1) {
#ifdef _WIN32
        Sleep(1000);
#else
        struct timespec ts = {1, 0};
        nanosleep(&ts, NULL);
#endif
        mutex_lock(&file_mutex);
        long rd = rows_done;
        mutex_unlock(&file_mutex);
        if (rd != prev) {
            prev = rd;
            printf("Progress: rows %ld/%d\n", rd, total_z);
            fflush(stdout);
        }
        if (rd >= total_z) break;
    }

    for (int i = 0; i < n_threads; i++) JOIN_THREAD(threads[i]);

    if (match_file) fclose(match_file);
    mutex_destroy(&file_mutex);
    free(ranges);
    free(threads);

    printf("Done. %ld match(es) found.\n", match_count);
    return 0;
}
