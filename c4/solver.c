/*
 * Exact Connect Four solver (7 columns x 6 rows).
 *
 * Negamax with alpha-beta, a two-bound transposition table, threat-based
 * move ordering, "non-losing move" pruning and an iterative null-window
 * driver. Positions use the column-major bitboard layout shared with
 * c4/game.py: bit (col*7 + row), row 0 = bottom, bit 6 of each column is a
 * sentinel that is always empty.
 *
 * Score convention (from the side to move):
 *   > 0  : side to move wins; larger means it wins sooner
 *   = 0  : draw with perfect play
 *   < 0  : side to move loses; more negative means it loses sooner
 * A win with your k-th stone (counting only your own stones) scores 22 - k.
 */
#include <stdint.h>
#ifndef __wasm__
#include <stdlib.h>
#include <string.h>
#endif

#define W 7
#define H 6
#define H1 (H + 1)
#define CELLS (W * H)

typedef uint64_t u64;

#define BOTTOM_MASK 0x0040810204081ULL /* bit 0 of every column */
#define BOARD_MASK (BOTTOM_MASK * ((1ULL << H) - 1))

static inline u64 col_mask(int c) { return ((1ULL << H) - 1) << (c * H1); }

static u64 winning_spots(u64 p, u64 mask) {
    u64 r = (p << 1) & (p << 2) & (p << 3); /* vertical */
    u64 t;
    static const int dirs[3] = {H1, H, H + 2}; /* horizontal, diag, anti-diag */
    for (int i = 0; i < 3; i++) {
        int d = dirs[i];
        t = (p << d) & (p << (2 * d));
        r |= t & (p << (3 * d));
        r |= t & (p >> d);
        t = (p >> d) & (p >> (2 * d));
        r |= t & (p << d);
        r |= t & (p >> (3 * d));
    }
    return r & (BOARD_MASK ^ mask);
}

static inline u64 possible(u64 mask) { return (mask + BOTTOM_MASK) & BOARD_MASK; }

static inline int popcount(u64 x) { return __builtin_popcountll(x); }

/* ---------------- transposition table ---------------- */
typedef struct { u64 key; int8_t lo, hi; } Entry;
static Entry *tt = 0;
static u64 tt_mask = 0;
static unsigned long long node_count = 0;
static unsigned long long node_limit = 0; /* 0 = unlimited */
static int aborted = 0;

#ifdef __wasm__
/* browser build: fixed 1M-entry (16 MB) table, no libc */
#define TT_BITS 20
static Entry tt_static[1u << TT_BITS];
static void tt_init(void) { tt = tt_static; tt_mask = (1u << TT_BITS) - 1; }
void c4_reset(void) { tt_init(); for (u64 i = 0; i <= tt_mask; i++) { tt[i].key = 0; } }
#else
static void tt_init(void) {
    if (tt) return;
    u64 size = 1ULL << 23; /* 8M entries, 128 MB */
    tt = (Entry *)calloc(size, sizeof(Entry));
    tt_mask = size - 1;
}
void c4_reset(void) { if (tt) memset(tt, 0, (tt_mask + 1) * sizeof(Entry)); }
#endif
unsigned long long c4_nodes(void) { return node_count; }

/* horizontal mirror; positions and their mirror images share TT entries */
static inline u64 mirror(u64 k) {
    const u64 m = (1ULL << H1) - 1;
    return ((k & m) << 6 * H1) | (((k >> H1) & m) << 5 * H1) | (((k >> 2 * H1) & m) << 4 * H1)
         | (k & (m << 3 * H1)) | (((k >> 4 * H1) & m) << 2 * H1) | (((k >> 5 * H1) & m) << H1)
         | ((k >> 6 * H1) & m);
}

static inline u64 hash(u64 k) {
    k ^= k >> 33; k *= 0xff51afd7ed558ccdULL; k ^= k >> 33;
    return k;
}

/* ---------------- search ---------------- */
static const int ORDER[W] = {3, 2, 4, 1, 5, 0, 6};

static int negamax(u64 cur, u64 mask, int moves, int alpha, int beta) {
    node_count++;
    if (node_limit && node_count > node_limit) { aborted = 1; return 0; }
    u64 opp = cur ^ mask;
    u64 poss = possible(mask);
    u64 opp_win = winning_spots(opp, mask);
    u64 forced = poss & opp_win;
    if (forced) {
        if (forced & (forced - 1)) return -(CELLS - moves) / 2; /* two threats */
        poss = forced;
    }
    u64 next = poss & ~(opp_win >> 1);
    if (!next) return -(CELLS - moves) / 2;
    if (moves >= CELLS - 2) return 0;

    int lo = -(CELLS - 2 - moves) / 2;
    if (alpha < lo) { alpha = lo; if (alpha >= beta) return alpha; }
    int hi = (CELLS - 1 - moves) / 2;
    if (beta > hi) { beta = hi; if (alpha >= beta) return beta; }

    u64 key = cur + mask;
    if (moves < 24) { u64 mk = mirror(key); if (mk < key) key = mk; }
    Entry *e = &tt[hash(key) & tt_mask];
    if (e->key == key) {
        if (e->lo > alpha) { alpha = e->lo; if (alpha >= beta) return alpha; }
        if (e->hi < beta) { beta = e->hi; if (alpha >= beta) return beta; }
    }

    /* order moves: most new threats first, ties broken centre-first */
    u64 mv[W]; int sc[W]; int n = 0;
    for (int i = 0; i < W; i++) {
        u64 m = next & col_mask(ORDER[i]);
        if (!m) continue;
        int s = popcount(winning_spots(cur | m, mask));
        int j = n++;
        while (j > 0 && sc[j - 1] < s) { mv[j] = mv[j - 1]; sc[j] = sc[j - 1]; j--; }
        mv[j] = m; sc[j] = s;
    }

    int alpha0 = alpha;
    for (int i = 0; i < n; i++) {
        u64 nm = mask | mv[i];
        int s = -negamax(cur ^ mask, nm, moves + 1, -beta, -alpha);
        if (aborted) return 0; /* never store results of an interrupted search */
        if (s >= beta) {
            if (e->key != key) { e->key = key; e->lo = -64; e->hi = 64; }
            if (s > e->lo) e->lo = (int8_t)s;
            return s;
        }
        if (s > alpha) alpha = s;
    }
    if (e->key != key) { e->key = key; e->lo = -64; e->hi = 64; }
    if (alpha > alpha0) { e->lo = e->hi = (int8_t)alpha; }
    else if (alpha < e->hi) e->hi = (int8_t)alpha;
    return alpha;
}

/* Exact (weak=0) or win/draw/loss-only (weak=1, result in {-1,0,1}) value. */
int c4_solve(u64 cur, u64 mask, int moves, int weak) {
    tt_init();
    if (winning_spots(cur, mask) & possible(mask)) {
        int s = (CELLS + 1 - moves) / 2;
        return weak ? 1 : s;
    }
    int lo = -(CELLS - moves) / 2, hi = (CELLS + 1 - moves) / 2;
    if (weak) { lo = -1; hi = 1; }
    while (lo < hi) {
        int med = lo + (hi - lo) / 2;
        if (med <= 0 && lo / 2 < med) med = lo / 2;
        else if (med >= 0 && hi / 2 > med) med = hi / 2;
        int r = negamax(cur, mask, moves, med, med + 1);
        if (r <= med) hi = r; else lo = r;
    }
    if (weak) return lo > 0 ? 1 : (lo < 0 ? -1 : 0);
    return lo;
}

/* Value of every column from the mover's perspective; -1000 if illegal. */
void c4_analyze(u64 cur, u64 mask, int moves, int weak, int *out) {
    tt_init();
    for (int c = 0; c < W; c++) {
        if (mask & (1ULL << (H - 1 + c * H1))) { out[c] = -1000; continue; }
        u64 m = (mask + (1ULL << (c * H1))) & col_mask(c);
        if (winning_spots(cur, mask) & m) { out[c] = weak ? 1 : (CELLS + 1 - moves) / 2; continue; }
        if (moves + 1 == CELLS) { out[c] = 0; continue; }
        out[c] = -c4_solve(cur ^ mask, mask | m, moves + 1, weak);
    }
}

/* Like c4_analyze, but gives up after `budget` search nodes in total.
   Returns 1 if every column was solved, 0 if the budget ran out. */
int c4_analyze_budget(u64 cur, u64 mask, int moves, int weak, int *out, unsigned long long budget) {
    node_limit = node_count + budget;
    aborted = 0;
    c4_analyze(cur, mask, moves, weak, out);
    int ok = !aborted;
    node_limit = 0;
    aborted = 0;
    return ok;
}
