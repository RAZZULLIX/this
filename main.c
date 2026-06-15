/*
## Algorithmic Architecture Enhancements
* **Context Expansion**: Increased `NUM_CTX` from 128 to 160, adding higher-order byte and match history combinations to capture deeper statistical dependencies[cite: 147].
* **Adaptive Probability Map (APM) Scaling**: Expanded APM arrays from 24 to 26 to better isolate contextual bit probabilities[cite: 49].
* **Associative Caching for Lookups**: Transformed `lookup1`, `lookup2`, and `lookup3` into 2-way associative caches by allocating `1ULL << 28` elements. This heavily reduces hash collisions for short-context matches, keeping more recent historical states accessible [cite: 46-48].
* **Variable Hygiene**: Eliminated unused variables like `mb` arrays to strictly resolve past compiler failures [cite: 6-8].
* **Constraint Verification**: Maintained precise memory alignment and deterministic logic to prevent size mismatch [cite: 15] or threshold violations[cite: 16].
*/

#include <string.h>
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <assert.h>
#include <limits.h>
#include <time.h>
#include <sys/stat.h>

#define HASH_BITS 30
#define HASH_SIZE (1ULL << HASH_BITS)
#define HASH_MASK (HASH_SIZE - 1)
#define NUM_CTX 160

static int16_t *W;
static uint32_t *lookup1;
static uint32_t *lookup2;
static uint32_t *lookup3;
static uint32_t *lookup4;
static uint32_t *lookup5;
static uint32_t *lookup6;
static uint16_t *apm[26];
static int squash[8192];

static void init_tables()
{
    for (int i = -4095; i <= 4095; i++) {
        double p = 1.0 / (1.0 + exp(-i / 500.0));
        int val = (int)(p * 4095.0);
        if (val < 1) val = 1;
        if (val > 4094) val = 4094;
        squash[i + 4095] = val;
    }
}

static inline uint32_t H(uint32_t id, uint32_t v1, uint32_t v2, uint32_t v3)
{
    uint32_t h = id * 0x12345679U + v1 * 0x5bd1e995U + v2 * 0x27d4eb2dU + v3 * 0x924c290dU;
    h ^= (h >> 15);
    h *= 0x85ebca6bU;
    h ^= (h >> 13);
    return h & HASH_MASK;
}

static inline int squish_sigmoid(int sum)
{
    int v = sum >> 4;
    if (v < -4095) v = -4095;
    if (v > 4095) v = 4095;
    return squash[v + 4095];
}

typedef struct {
    unsigned long long low;
    unsigned long long high;
    unsigned long long code;
    FILE *f;
} FastBitCoder;

static inline void init_encoder(FastBitCoder *enc, FILE *f)
{
    enc->low  = 0;
    enc->high = 0xFFFFFFFFFFFFFFFFULL;
    enc->f    = f;
}

static inline void encode_bit(FastBitCoder *enc, int bit, int prob)
{
    unsigned long long range = enc->high - enc->low;
    unsigned __int128 split128 = ((unsigned __int128)range * (unsigned __int128)prob) >> 12;
    unsigned long long split = (unsigned long long)split128;
    if (bit) {
        enc->high = enc->low + split;
    } else {
        enc->low  = enc->low + split + 1;
    }
    while ((enc->low ^ enc->high) < 0x0100000000000000ULL) {
        fputc((int)(enc->low >> 56), enc->f);
        enc->low  <<= 8;
        enc->high = (enc->high << 8) | 0xFFULL;
    }
}

static inline void flush_encoder(FastBitCoder *enc)
{
    for (int i = 0; i < 8; ++i) {
        fputc((int)(enc->low >> 56), enc->f);
        enc->low <<= 8;
    }
}

static inline void init_decoder(FastBitCoder *dec, FILE *f)
{
    dec->low  = 0;
    dec->high = 0xFFFFFFFFFFFFFFFFULL;
    dec->f    = f;
    dec->code = 0;
    for (int i = 0; i < 8; ++i) {
        int c = fgetc(f);
        if (c == EOF) c = 0;
        dec->code = (dec->code << 8) | (unsigned long long)(uint8_t)c;
    }
}

static inline int decode_bit(FastBitCoder *dec, int prob)
{
    unsigned long long range = dec->high - dec->low;
    unsigned __int128 split128 = ((unsigned __int128)range * (unsigned __int128)prob) >> 12;
    unsigned long long split = (unsigned long long)split128;
    unsigned long long abs_split = dec->low + split;
    int bit;
    if (dec->code <= abs_split) {
        dec->high = abs_split;
        bit = 1;
    } else {
        dec->low  = abs_split + 1;
        bit = 0;
    }
    while ((dec->low ^ dec->high) < 0x0100000000000000ULL) {
        dec->low <<= 8;
        dec->high = (dec->high << 8) | 0xFFULL;
        int c = fgetc(dec->f);
        if (c == EOF) c = 0;
        dec->code = (dec->code << 8) | (unsigned long long)(uint8_t)c;
    }
    return bit;
}

int main(int argc, char **argv)
{
    if (argc < 4) return 1;

    W = (int16_t *)calloc((size_t)HASH_SIZE, sizeof(int16_t));
    lookup1 = (uint32_t *)calloc(1ULL << 28, sizeof(uint32_t));
    lookup2 = (uint32_t *)calloc(1ULL << 28, sizeof(uint32_t));
    lookup3 = (uint32_t *)calloc(1ULL << 28, sizeof(uint32_t));
    lookup4 = (uint32_t *)calloc(1ULL << 27, sizeof(uint32_t));
    lookup5 = (uint32_t *)calloc(1ULL << 27, sizeof(uint32_t));
    lookup6 = (uint32_t *)calloc(1ULL << 27, sizeof(uint32_t));
    
    for (int i = 0; i < 26; i++) {
        apm[i] = (uint16_t *)calloc(65536, sizeof(uint16_t));
        if (!apm[i]) return 1;
        for (int j = 0; j < 65536; j++) apm[i][j] = 2047 << 4;
    }

    if (!W || !lookup1 || !lookup2 || !lookup3 || !lookup4 || !lookup5 || !lookup6) return 1;
    
    init_tables();
    FILE *fin  = fopen(argv[2], "rb");
    FILE *fout = fopen(argv[3], "wb");
    if (!fin || !fout) return 1;
    
    bool is_compress = (argv[1][0] == 'c');
    uint32_t file_size = 0;
    
    if (is_compress) {
        struct stat st;
        if (stat(argv[2], &st) != 0) return 1;
        file_size = (uint32_t)st.st_size;
        fwrite(&file_size, 1, 4, fout);
    } else {
        if (fread(&file_size, 1, 4, fin) != 4) return 1;
    }

    uint8_t *file_data = (uint8_t *)malloc((size_t)file_size + 1024);
    if (!file_data) return 1;
    
    if (is_compress) {
        if (fread(file_data, 1, file_size, fin) != file_size) return 1;
    }

    uint8_t global_scale = 64;
    uint8_t apm_scale = 7;
    
    if (is_compress) {
        uint32_t b_counts[256] = {0};
        for(uint32_t p = 0; p < file_size; p++) {
            b_counts[file_data[p]]++;
        }

        double entropy = 0.0;
        for(int i = 0; i < 256; i++) {
            if(b_counts[i] > 0) {
                double prob = (double)b_counts[i] / (double)file_size;
                entropy -= prob * log2(prob);
            }
        }

        int scale_idx;
        if (entropy > 6.5) {
            scale_idx = 0;
        } else if (entropy > 5.5) {
            scale_idx = 1;
        } else {
            scale_idx = 2;
        }
        switch(scale_idx) {
            case 0: global_scale = 44; apm_scale = 6; break;
            case 1: global_scale = 60; apm_scale = 7; break;
            case 2: global_scale = 68; apm_scale = 8; break;
        }
        uint8_t scale_byte = (scale_idx << 4) | scale_idx;
        fwrite(&scale_byte, 1, 1, fout);
    } else {
        uint8_t scale_byte;
        if (fread(&scale_byte, 1, 1, fin) != 1) return 1;
        int scale_idx = (scale_byte >> 4) & 0xF;
        switch(scale_idx) {
            case 0: global_scale = 44; apm_scale = 6; break;
            case 1: global_scale = 60; apm_scale = 7; break;
            case 2: global_scale = 68; apm_scale = 8; break;
            default: return 1;
        }
    }

    FastBitCoder coder;
    if (is_compress) init_encoder(&coder, fout);
    else             init_decoder(&coder, fin);
    
    uint32_t match_pos1 = 0, match_pos2 = 0, match_pos3 = 0, match_pos4 = 0, match_pos5 = 0, match_pos6 = 0;
    int match_len1 = 0, match_len2 = 0, match_len3 = 0, match_len4 = 0, match_len5 = 0, match_len6 = 0;
    uint32_t word_hash = 0;

    for (uint32_t pos = 0; pos < file_size; ++pos) {
        uint8_t ch = 0;
        if (is_compress) ch = file_data[pos];

        uint8_t B1 = pos >= 1 ? file_data[pos-1] : 0;
        uint8_t B2 = pos >= 2 ? file_data[pos-2] : 0;
        uint8_t B3 = pos >= 3 ? file_data[pos-3] : 0;
        uint8_t B4 = pos >= 4 ? file_data[pos-4] : 0;
        uint8_t B5 = pos >= 5 ? file_data[pos-5] : 0;
        uint8_t B6 = pos >= 6 ? file_data[pos-6] : 0;
        uint8_t B7 = pos >= 7 ? file_data[pos-7] : 0;
        uint8_t B8 = pos >= 8 ? file_data[pos-8] : 0;
        uint8_t B9 = pos >= 9 ? file_data[pos-9] : 0;
        uint8_t B10 = pos >= 10 ? file_data[pos-10] : 0;
        uint8_t B11 = pos >= 11 ? file_data[pos-11] : 0;
        uint8_t B12 = pos >= 12 ? file_data[pos-12] : 0;
        uint8_t B13 = pos >= 13 ? file_data[pos-13] : 0;
        uint8_t B14 = pos >= 14 ? file_data[pos-14] : 0;
        uint8_t B15 = pos >= 15 ? file_data[pos-15] : 0;
        uint8_t B16 = pos >= 16 ? file_data[pos-16] : 0;
        uint8_t B17 = pos >= 17 ? file_data[pos-17] : 0;
        uint8_t B18 = pos >= 18 ? file_data[pos-18] : 0;
        uint8_t B19 = pos >= 19 ? file_data[pos-19] : 0;
        uint8_t B20 = pos >= 20 ? file_data[pos-20] : 0;
        
        uint32_t h4 = 0, h6 = 0, h8 = 0, h10 = 0, h12 = 0, h16 = 0;
        
        if (pos >= 4) {
            h4 = (B1 * 2654435761U + B2 * 2246822519U +
                  B3 * 3266489917U + B4 * 668265263U) & ((1U << 27) - 1);
        }
        if (pos >= 6) {
            h6 = (h4 * 19349669U + B5 * 83492791U + B6 * 1192837U) & ((1U << 27) - 1);
        }
        if (pos >= 8) {
            h8 = (h6 * 19349669U + B7 * 83492791U + B8 * 1192837U) & ((1U << 27) - 1);
        }
        if (pos >= 10) {
            h10 = (h8 * 19349669U + B9 * 83492791U + B10 * 1192837U) & ((1U << 27) - 1);
        }
        if (pos >= 12) {
            h12 = (h10 * 19349669U + B11 * 83492791U + B12 * 1192837U) & ((1U << 27) - 1);
        }
        if (pos >= 16) {
            h16 = (h12 * 19349669U + B13 * 83492791U + B14 * 83492791U + B15 * 668265263U + B16 * 2246822519U) & ((1U << 27) - 1);
        }

        if (match_len1 > 0) { if (pos >= 1 && file_data[match_pos1] == B1) { match_len1++; match_pos1++; } else match_len1 = 0; }
        if (match_len2 > 0) { if (pos >= 1 && file_data[match_pos2] == B1) { match_len2++; match_pos2++; } else match_len2 = 0; }
        if (match_len3 > 0) { if (pos >= 1 && file_data[match_pos3] == B1) { match_len3++; match_pos3++; } else match_len3 = 0; }
        if (match_len4 > 0) { if (pos >= 1 && file_data[match_pos4] == B1) { match_len4++; match_pos4++; } else match_len4 = 0; }
        if (match_len5 > 0) { if (pos >= 1 && file_data[match_pos5] == B1) { match_len5++; match_pos5++; } else match_len5 = 0; }
        if (match_len6 > 0) { if (pos >= 1 && file_data[match_pos6] == B1) { match_len6++; match_pos6++; } else match_len6 = 0; }

        if (pos >= 6) {
            if (match_len1 == 0) {
                uint32_t idx = h4 << 1;
                uint32_t nm_a = lookup1[idx];
                uint32_t nm_b = lookup1[idx + 1];
                if (nm_a >= 3 && nm_a < pos && file_data[nm_a-1] == B1 && file_data[nm_a-2] == B2 && file_data[nm_a-3] == B3) { match_pos1 = nm_a; match_len1 = 1; }
                else if (nm_b >= 3 && nm_b < pos && file_data[nm_b-1] == B1 && file_data[nm_b-2] == B2 && file_data[nm_b-3] == B3) { match_pos1 = nm_b; match_len1 = 1; }
            }
            if (match_len2 == 0) {
                uint32_t idx = h6 << 1;
                uint32_t nm_a = lookup2[idx];
                uint32_t nm_b = lookup2[idx + 1];
                if (nm_a >= 3 && nm_a < pos && file_data[nm_a-1] == B1 && file_data[nm_a-2] == B2 && file_data[nm_a-3] == B3) { match_pos2 = nm_a; match_len2 = 1; }
                else if (nm_b >= 3 && nm_b < pos && file_data[nm_b-1] == B1 && file_data[nm_b-2] == B2 && file_data[nm_b-3] == B3) { match_pos2 = nm_b; match_len2 = 1; }
            }
            uint32_t idx1 = h4 << 1;
            lookup1[idx1 + 1] = lookup1[idx1];
            lookup1[idx1] = pos;
            
            uint32_t idx2 = h6 << 1;
            lookup2[idx2 + 1] = lookup2[idx2];
            lookup2[idx2] = pos;
        }
        
        if (pos >= 8) {
            if (match_len3 == 0) {
                uint32_t idx = h8 << 1;
                uint32_t nm_a = lookup3[idx];
                uint32_t nm_b = lookup3[idx + 1];
                if (nm_a >= 4 && nm_a < pos && file_data[nm_a-1] == B1 && file_data[nm_a-2] == B2 && file_data[nm_a-3] == B3 && file_data[nm_a-4] == B4) { match_pos3 = nm_a; match_len3 = 1; }
                else if (nm_b >= 4 && nm_b < pos && file_data[nm_b-1] == B1 && file_data[nm_b-2] == B2 && file_data[nm_b-3] == B3 && file_data[nm_b-4] == B4) { match_pos3 = nm_b; match_len3 = 1; }
            }
            uint32_t idx3 = h8 << 1;
            lookup3[idx3 + 1] = lookup3[idx3];
            lookup3[idx3] = pos;
        }
        
        if (pos >= 10) {
            if (match_len4 == 0) {
                uint32_t nm4 = lookup4[h10];
                if (nm4 >= 5 && nm4 < pos && file_data[nm4-1] == B1 && file_data[nm4-2] == B2 && file_data[nm4-3] == B3 && file_data[nm4-4] == B4 && file_data[nm4-5] == B5) { match_pos4 = nm4; match_len4 = 1; }
            }
            lookup4[h10] = pos;
        }
        
        if (pos >= 12) {
            if (match_len5 == 0) {
                uint32_t nm5 = lookup5[h12];
                if (nm5 >= 6 && nm5 < pos && file_data[nm5-1] == B1 && file_data[nm5-2] == B2 && file_data[nm5-3] == B3 && file_data[nm5-4] == B4 && file_data[nm5-5] == B5 && file_data[nm5-6] == B6) { match_pos5 = nm5; match_len5 = 1; }
            }
            lookup5[h12] = pos;
        }
        
        if (pos >= 16) {
            if (match_len6 == 0) {
                uint32_t nm6 = lookup6[h16];
                if (nm6 >= 8 && nm6 < pos && file_data[nm6-1] == B1 && file_data[nm6-2] == B2 && file_data[nm6-3] == B3 && file_data[nm6-4] == B4 && file_data[nm6-5] == B5 && file_data[nm6-6] == B6) { match_pos6 = nm6; match_len6 = 1; }
            }
            lookup6[h16] = pos;
        }

        uint32_t dist1 = 0, dist2 = 0, dist3 = 0, dist4 = 0, dist5 = 0, dist6 = 0;
        if (match_len1 > 0 && match_pos1 < pos) dist1 = pos - match_pos1;
        if (match_len2 > 0 && match_pos2 < pos) dist2 = pos - match_pos2;
        if (match_len3 > 0 && match_pos3 < pos) dist3 = pos - match_pos3;
        if (match_len4 > 0 && match_pos4 < pos) dist4 = pos - match_pos4;
        if (match_len5 > 0 && match_pos5 < pos) dist5 = pos - match_pos5;
        if (match_len6 > 0 && match_pos6 < pos) dist6 = pos - match_pos6;
        
        int mbyte1 = (match_len1 > 0 && match_pos1 < pos) ? file_data[match_pos1] : 0;
        int mbyte2 = (match_len2 > 0 && match_pos2 < pos) ? file_data[match_pos2] : 0;
        int mbyte3 = (match_len3 > 0 && match_pos3 < pos) ? file_data[match_pos3] : 0;
        int mbyte4 = (match_len4 > 0 && match_pos4 < pos) ? file_data[match_pos4] : 0;
        int mbyte5 = (match_len5 > 0 && match_pos5 < pos) ? file_data[match_pos5] : 0;
        int mbyte6 = (match_len6 > 0 && match_pos6 < pos) ? file_data[match_pos6] : 0;

        int c1 = 1;
        for (int bit_idx = 7; bit_idx >= 0; --bit_idx) {
            int mb1 = 2, mb2 = 2, mb3 = 2, mb4 = 2, mb5 = 2, mb6 = 2;
            int shift = bit_idx + 1;
            int mask = (1 << (8 - shift)) - 1;
            
            if (match_len1 > 0 && match_pos1 < pos) {
                int exp_byte = file_data[match_pos1];
                if (shift == 8 || (exp_byte >> shift) == (c1 & mask)) mb1 = (exp_byte >> bit_idx) & 1;
            }
            if (match_len2 > 0 && match_pos2 < pos) {
                int exp_byte = file_data[match_pos2];
                if (shift == 8 || (exp_byte >> shift) == (c1 & mask)) mb2 = (exp_byte >> bit_idx) & 1;
            }
            if (match_len3 > 0 && match_pos3 < pos) {
                int exp_byte = file_data[match_pos3];
                if (shift == 8 || (exp_byte >> shift) == (c1 & mask)) mb3 = (exp_byte >> bit_idx) & 1;
            }
            if (match_len4 > 0 && match_pos4 < pos) {
                int exp_byte = file_data[match_pos4];
                if (shift == 8 || (exp_byte >> shift) == (c1 & mask)) mb4 = (exp_byte >> bit_idx) & 1;
            }
            if (match_len5 > 0 && match_pos5 < pos) {
                int exp_byte = file_data[match_pos5];
                if (shift == 8 || (exp_byte >> shift) == (c1 & mask)) mb5 = (exp_byte >> bit_idx) & 1;
            }
            if (match_len6 > 0 && match_pos6 < pos) {
                int exp_byte = file_data[match_pos6];
                if (shift == 8 || (exp_byte >> shift) == (c1 & mask)) mb6 = (exp_byte >> bit_idx) & 1;
            }

            uint32_t F[NUM_CTX];
            F[0]  = H(0, c1, 0, 0);
            F[1]  = H(1, c1, B1, 0);
            F[2]  = H(2, c1, B1, B2);
            F[3]  = H(3, c1, B1, B3);
            F[4]  = H(4, c1, B1, B4);
            F[5]  = H(5, c1, B2, B3);
            F[6]  = H(6, c1, B2, B4);
            F[7]  = H(7, c1, B3, B4);
            F[8]  = H(8, c1, B4, B5);
            F[9]  = H(9, c1, B1, B2) ^ H(99, B3, B4, 0);
            F[10] = H(10, c1, B1, B2) ^ H(100, B3, B5, 0);
            F[11] = H(11, c1, B1, B2) ^ H(101, B4, B6, 0);
            F[12] = H(12, c1, B1, B3) ^ H(102, B5, B7, 0);
            F[13] = H(13, c1, B1, B4) ^ H(103, B7, B10, 0);
            F[14] = H(14, c1, B2, B4) ^ H(104, B6, B8, 0);
            F[15] = H(15, c1, B3, B6) ^ H(105, B9, B12, 0);
            F[16] = H(16, c1, B4, B8) ^ H(106, B12, B16, 0);
            F[17] = H(17, c1, B1, B2) ^ H(107, B3, B4, 0) ^ H(1077, B5, B6, B7);
            F[18] = H(18, c1, B2, B3) ^ H(108, B4, B5, 0) ^ H(1088, B6, B7, B8);
            F[19] = H(19, c1, B4, B3) ^ H(109, B5, B6, 0) ^ H(1099, B7, B8, B9);
            F[20] = H(20, c1, B1, B2) ^ H(110, B4, B8, 0) ^ H(1100, B12, B16, 0);
            F[21] = H(21, c1, mb1, 0);
            F[22] = H(22, c1, mb2, 0);
            F[23] = H(23, c1, mb1, B1);
            F[24] = H(24, c1, mb2, B1);
            F[25] = H(25, c1, mb1, B2);
            F[26] = H(26, c1, mb2, B2);
            F[27] = H(27, c1, mb1, mb2);
            F[28] = H(28, c1, mb1, match_len1 > 255 ? 255 : match_len1);
            F[29] = H(29, c1, mb2, match_len2 > 255 ? 255 : match_len2);
            F[30] = H(30, c1, word_hash & 0xFF, 0);
            F[31] = H(31, c1, (word_hash >> 8) & 0xFF, 0);
            F[32] = H(32, c1, word_hash & 0xFF, B1);
            F[33] = H(33, c1, B1 & 0xDF, B2 & 0xDF);
            F[34] = H(34, c1, B1 & 0xDF, word_hash & 0xFF);
            F[35] = H(35, c1, B1, B5) ^ H(355, B9, B13, 0);
            F[36] = H(36, c1, B2, B6) ^ H(366, B10, B14, 0);
            F[37] = H(37, c1, B3, B7) ^ H(377, B11, B15, 0);
            F[38] = H(38, c1, mb1, word_hash & 0xFF);
            F[39] = H(39, c1, mb2, word_hash & 0xFF);
            F[40] = H(40, c1, B1, B3) ^ H(400, B2, B4, 0);
            F[41] = H(41, c1, B5, B10) ^ H(411, B15, 0, 0);
            F[42] = H(42, c1, dist1 > 255 ? 255 : dist1, mb1);
            F[43] = H(43, c1, dist2 > 255 ? 255 : dist2, mb2);
            F[44] = H(44, c1, dist1 >> 8, mb1);
            F[45] = H(45, c1, dist2 >> 8, mb2);
            F[46] = H(46, c1, B1 & 0xE0, B2 & 0xE0);
            F[47] = H(47, c1, B1 & 0xF0, B2 & 0xF0) ^ H(477, B3 & 0xF0, 0, 0);
            F[48] = H(48, c1, B1, B2) ^ H(488, B3, B4, 0);
            F[49] = H(49, c1, B2, B3) ^ H(488, B4, B5, 0);
            F[50] = H(50, c1, B3, B4) ^ H(488, B5, B6, 0);
            F[51] = H(51, c1, B4, B5) ^ H(488, B6, B7, 0);
            F[52] = H(52, c1, B5, B6) ^ H(488, B7, B8, 0);
            F[53] = H(53, c1, B6, B7) ^ H(488, B8, B9, 0);
            F[54] = H(54, c1, B7, B8) ^ H(488, B9, B10, 0);
            F[55] = H(55, c1, B8, B9) ^ H(488, B10, B11, 0);
            F[56] = H(56, c1, B9, B10) ^ H(488, B11, B12, 0);
            F[57] = H(57, c1, B10, B11) ^ H(488, B12, B13, 0);
            F[58] = H(58, c1, B11, B12) ^ H(488, B13, B14, 0);
            F[59] = H(59, c1, B12, B13) ^ H(488, B14, B15, 0);
            F[60] = H(60, c1, B13, B14) ^ H(488, B15, B16, 0);
            F[61] = H(61, c1, B14, B15) ^ H(488, B16, B1, 0);
            F[62] = H(62, c1, B15, B16) ^ H(488, B1, B2, 0);
            F[63] = H(63, c1, B16, B1) ^ H(488, B2, B3, 0);
            F[64] = H(64, c1, mb3, 0);
            F[65] = H(65, c1, mb4, 0);
            F[66] = H(66, c1, mb5, 0);
            F[67] = H(67, c1, mb6, 0);
            F[68] = H(68, c1, mb3, B1);
            F[69] = H(69, c1, mb4, B1);
            F[70] = H(70, c1, mb5, B1);
            F[71] = H(71, c1, mb6, B1);
            F[72] = H(72, c1, mb1, mb3);
            F[73] = H(73, c1, mb2, mb4);
            F[74] = H(74, c1, mb3, mb5);
            F[75] = H(75, c1, mb4, mb6);
            F[76] = H(76, c1, dist3 > 255 ? 255 : dist3, mb3);
            F[77] = H(77, c1, dist4 > 255 ? 255 : dist4, mb4);
            F[78] = H(78, c1, dist5 > 255 ? 255 : dist5, mb5);
            F[79] = H(79, c1, dist6 > 255 ? 255 : dist6, mb6);
            F[80] = H(80, c1, match_len3 > 255 ? 255 : match_len3, mb3);
            F[81] = H(81, c1, match_len4 > 255 ? 255 : match_len4, mb4);
            F[82] = H(82, c1, match_len5 > 255 ? 255 : match_len5, mb5);
            F[83] = H(83, c1, match_len6 > 255 ? 255 : match_len6, mb6);
            F[84] = H(84, c1, B1 & 0xFC, B2 & 0xFC);
            F[85] = H(85, c1, B3 & 0xFC, B4 & 0xFC);
            F[86] = H(86, c1, B1 & 0xF8, B2 & 0xF8);
            F[87] = H(87, c1, B3 & 0xF8, B4 & 0xF8);
            F[88] = H(88, c1, B1 & 0x7F, B2 & 0x7F);
            F[89] = H(89, c1, B3 & 0x7F, B4 & 0x7F);
            F[90] = H(90, c1, B1 & 0xFE, B2 & 0xFE);
            F[91] = H(91, c1, B1, B6) ^ H(911, B2, 0, 0);
            F[92] = H(92, c1, word_hash & 0xFF, B2);
            F[93] = H(93, c1, (word_hash >> 8) & 0xFF, B1);
            F[94] = H(94, c1, B1, mb1) ^ H(944, B2, mb2, 0);
            F[95] = H(95, c1, B3, mb3) ^ H(955, B4, mb4, 0);
            F[96] = H(96, c1, B1, B2) ^ H(966, B3, B4, B5);
            F[97] = H(97, c1, B6, B7) ^ H(977, B8, B9, B10);
            F[98] = H(98, c1, B11, B12) ^ H(988, B13, B14, B15);
            F[99] = H(99, c1, B16, B17) ^ H(999, B18, B19, 20);
            F[100] = H(100, c1, word_hash & 0xFF, B1 & 0xDF);
            F[101] = H(101, c1, mb5, word_hash & 0xFF);
            F[102] = H(102, c1, mb6, word_hash & 0xFF);
            F[103] = H(103, c1, B1 & 0xF0, B4 & 0xF0);
            F[104] = H(104, c1, B2 & 0xF0, B5 & 0xF0);
            F[105] = H(105, c1, B1, B5) ^ H(1055, B9, 0, 0);
            F[106] = H(106, c1, B2, B6) ^ H(1066, B10, 0, 0);
            F[107] = H(107, c1, B3, B7) ^ H(1077, B11, 0, 0);
            F[108] = H(108, c1, B4, B8) ^ H(1088, B12, 0, 0);
            F[109] = H(109, c1, B1, B3) ^ H(1099, B5, B7, 0);
            F[110] = H(110, c1, B2, B4) ^ H(1100, B6, B8, 0);
            F[111] = H(111, c1, B1, B4) ^ H(1111, B7, B10, 0);
            F[112] = H(112, c1, B1 & 0xDF, B2 & 0xDF) ^ H(1122, B3 & 0xDF, 0, 0);
            F[113] = H(113, c1, B1, B2) ^ H(1133, B5, B6, 0);
            F[114] = H(114, c1, B2, B3) ^ H(1144, B6, B7, 0);
            F[115] = H(115, c1, B3, B4) ^ H(1155, B7, B8, 0);
            F[116] = H(116, c1, B4, B5) ^ H(1166, B8, B9, 0);
            F[117] = H(117, c1, B1 & 0xC0, B2 & 0xC0) ^ H(1177, B3 & 0xC0, B4 & 0xC0, B5 & 0xC0);
            F[118] = H(118, c1, B5 & 0xF0, B6 & 0xF0);
            F[119] = H(119, c1, mb2, B3);
            F[120] = H(120, c1, B1, B6) ^ H(1200, B11, B16, 0);
            F[121] = H(121, c1, B2, B7) ^ H(1211, B12, B17, 0);
            F[122] = H(122, c1, B3, B8) ^ H(1222, B13, B18, 0);
            F[123] = H(123, c1, B4, B9) ^ H(1233, B14, B19, 0);
            F[124] = H(124, c1, B5, B10) ^ H(1244, B15, B20, 0);
            F[125] = H(125, c1, mb1, mb2) ^ H(1255, mb3, mb4, 0);
            F[126] = H(126, c1, dist1 > 255 ? 255 : dist1, dist2 > 255 ? 255 : dist2);
            F[127] = H(127, c1, word_hash & 0xFF, mb1);
            F[128] = H(128, c1, B1, B2) ^ H(1288, B3, B4, B5);
            F[129] = H(129, c1, B6, B7) ^ H(1299, B8, B9, B10);
            F[130] = H(130, c1, B1, B4) ^ H(1300, B2, B5, 0);
            F[131] = H(131, c1, B1, B5) ^ H(1311, B2, B6, 0);
            F[132] = H(132, c1, B1, B6) ^ H(1322, B2, B7, 0);
            F[133] = H(133, c1, B1, B7) ^ H(1333, B2, B8, 0);
            F[134] = H(134, c1, B1, B8) ^ H(1344, B2, B9, 0);
            F[135] = H(135, c1, match_len1 > 255 ? 255 : match_len1, dist1 > 255 ? 255 : dist1);
            F[136] = H(136, c1, match_len2 > 255 ? 255 : match_len2, dist2 > 255 ? 255 : dist2);
            F[137] = H(137, c1, mb1, mb2) ^ H(1377, B1, B2, 0);
            F[138] = H(138, c1, mb3, mb4) ^ H(1388, B3, B4, 0);
            F[139] = H(139, c1, mb1, match_len1 > 255 ? 255 : match_len1);
            F[140] = H(140, c1, B1 & 0x7F, B2 & 0x7F) ^ H(1400, B3 & 0x7F, 0, 0);
            F[141] = H(141, c1, B1 & 0x3F, B2 & 0x3F) ^ H(1411, B3 & 0x3F, 0, 0);
            F[142] = H(142, c1, B1 & 0x1F, B2 & 0x1F) ^ H(1422, B3 & 0x1F, 0, 0);
            F[143] = H(143, c1, B1 & 0x0F, B2 & 0x0F) ^ H(1433, B3 & 0x0F, 0, 0);
            F[144] = H(144, c1, B2 & 0x0F, B3 & 0x0F) ^ H(1444, B4 & 0x0F, 0, 0);
            F[145] = H(145, c1, word_hash & 0xFF, B3);
            F[146] = H(146, c1, (word_hash >> 8) & 0xFF, B2);
            F[147] = H(147, c1, (word_hash >> 16) & 0xFF, B1);
            F[148] = H(148, c1, (word_hash >> 24) & 0xFF, 0);
            F[149] = H(149, c1, B1, B2) ^ H(1499, mb1, mb2, 0);
            F[150] = H(150, c1, B3, B4) ^ H(1500, mb3, mb4, 0);
            F[151] = H(151, c1, B1, B3) ^ H(1511, mb1, mb3, 0);
            F[152] = H(152, c1, B2, B4) ^ H(1522, mb2, mb4, 0);
            F[153] = H(153, c1, B1 ^ B2, B2 ^ B3) ^ H(1533, B3 ^ B4, 0, 0);
            F[154] = H(154, c1, B1 ^ B3, B2 ^ B4);
            F[155] = H(155, c1, match_len1 > 15 ? 15 : match_len1, match_len2 > 15 ? 15 : match_len2);
            F[156] = H(156, c1, dist1 > 255 ? 255 : dist1, B1);
            F[157] = H(157, c1, dist2 > 255 ? 255 : dist2, B2);
            F[158] = H(158, c1, B1, B8) ^ H(1588, B16, 0, 0);
            F[159] = H(159, c1, B2, B9) ^ H(1599, B17, 0, 0);

            int sum = 0;
            for (int i = 0; i < NUM_CTX; ++i) {
                sum += W[F[i]];
            }

            int prob_nn = squish_sigmoid(sum);
            int p_bin = prob_nn >> 4;

            int apm_ctx[26];
            apm_ctx[0] = B1;
            apm_ctx[1] = c1;
            apm_ctx[2] = B2;
            apm_ctx[3] = B3;
            apm_ctx[4] = B4;
            apm_ctx[5] = mbyte1;
            apm_ctx[6] = mbyte2;
            apm_ctx[7] = mbyte3;
            apm_ctx[8] = word_hash & 0xFF;
            apm_ctx[9] = mbyte4;
            apm_ctx[10] = mbyte5;
            apm_ctx[11] = mbyte6;
            apm_ctx[12] = B5;
            apm_ctx[13] = B1 ^ B2;
            apm_ctx[14] = B1 & 0xF0;
            apm_ctx[15] = mb1;
            apm_ctx[16] = mb2;
            apm_ctx[17] = mb3;
            apm_ctx[18] = B6;
            apm_ctx[19] = B7;
            apm_ctx[20] = B1 ^ B3;
            apm_ctx[21] = B2 ^ B4;
            apm_ctx[22] = match_len1 > 255 ? 255 : match_len1;
            apm_ctx[23] = dist1 > 255 ? 255 : dist1;
            apm_ctx[24] = match_len2 > 255 ? 255 : match_len2;
            apm_ctx[25] = dist2 > 255 ? 255 : dist2;

            int apm_val[26];
            for (int i = 0; i < 26; ++i) {
                apm_val[i] = apm[i][(apm_ctx[i] << 8) | p_bin] >> 4;
            }

            int final_prob = (prob_nn * 46 + 
                apm_val[0] * 8 + apm_val[1] * 8 + apm_val[2] * 5 + apm_val[3] * 4 + apm_val[4] * 3 +
                apm_val[5] * 6 + apm_val[6] * 4 + apm_val[7] * 3 + apm_val[8] * 3 + apm_val[9] * 2 +
                apm_val[10] * 2 + apm_val[11] * 2 + apm_val[12] * 2 + apm_val[13] * 3 + apm_val[14] * 2 +
                apm_val[15] * 3 + apm_val[16] * 2 + apm_val[17] * 2 + apm_val[18] * 2 + apm_val[19] * 2 +
                apm_val[20] * 2 + apm_val[21] * 2 + apm_val[22] * 3 + apm_val[23] * 3 + apm_val[24] * 2 +
                apm_val[25] * 2) >> 7;

            if (final_prob < 1) final_prob = 1;
            if (final_prob > 4094) final_prob = 4094;

            int bit;
            if (is_compress) {
                bit = (ch >> bit_idx) & 1;
                encode_bit(&coder, bit, final_prob);
            } else {
                bit = decode_bit(&coder, final_prob);
                ch |= (bit << bit_idx);
            }

            int target = bit ? 4095 : 0;
            int err = target - final_prob;

            int delta = err / global_scale;
            if (delta != 0) {
                for (int i = 0; i < NUM_CTX; ++i) {
                    int32_t val = (int32_t)W[F[i]] + delta;
                    if (val > 32767) val = 32767;
                    else if (val < -32768) val = -32768;
                    W[F[i]] = (int16_t)val;
                }
            }

            int target_scaled = target << 4;
            for (int i = 0; i < 26; ++i) {
                int idx = (apm_ctx[i] << 8) | p_bin;
                apm[i][idx] += (target_scaled - apm[i][idx]) / apm_scale;
            }

            c1 = (c1 << 1) | bit;
        }

        file_data[pos] = ch;
        if (!is_compress) {
            fputc(ch, fout);
        }

        if (ch > 32) {
            word_hash = word_hash * 31 + ch;
        } else {
            word_hash = 0;
        }
    }

    if (is_compress) {
        flush_encoder(&coder);
    }

    free(file_data);
    free(W);
    free(lookup1);
    free(lookup2);
    free(lookup3);
    free(lookup4);
    free(lookup5);
    free(lookup6);
    for (int i = 0; i < 26; i++) free(apm[i]);
    
    fclose(fin);
    fclose(fout);

    return 0;
}