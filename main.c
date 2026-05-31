#include <immintrin.h>

#include <stdbool.h>
#include <math.h>
#include <assert.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>

#define HASH_BITS 29
#define HASH_SIZE (1U << HASH_BITS)
#define HASH_MASK (HASH_SIZE - 1)
#define MIN(a, b) ((a) < (b) ? (a) : (b))

static int16_t *W;
static uint32_t *lookup1;
static uint32_t *lookup2;

static uint16_t apm1[256][32];
static uint16_t apm2[256][32];
static uint16_t apm3[65536][32];
static uint16_t apm4[65536][32];
static uint16_t apm_m1[3][32];
static uint16_t apm_m2[3][32];

static int squash[8192];

static void init_tables()
{
    for (int i = 0; i < 256; i++) {
        for (int j = 0; j < 32; j++) {
            uint16_t v = (j << 7) + 64;
            apm1[i][j] = v;
            apm2[i][j] = v;
        }
    }
    for (int i = 0; i < 65536; i++) {
        for (int j = 0; j < 32; j++) {
            uint16_t v = (j << 7) + 64;
            apm3[i][j] = v;
            apm4[i][j] = v;
        }
    }
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 32; j++) {
            uint16_t v = (j << 7) + 64;
            apm_m1[i][j] = v;
            apm_m2[i][j] = v;
        }
    }
    for (int i = -4095; i <= 4095; i++) {
        double p = 1.0 / (1.0 + exp(-i / 300.0));
        int val = (int)(p * 4095.0);
        if (val < 1) val = 1;
        if (val > 4094) val = 4094;
        squash[i + 4095] = val;
    }
}

static inline uint32_t H(uint32_t id, uint32_t v1, uint32_t v2, uint32_t v3)
{
    uint32_t h = id * 0x12345679 + v1 * 0x5bd1e995 + v2 * 0x27d4eb2d + v3 * 0x924c290d;
    h ^= (h >> 15);
    h *= 0x85ebca6b;
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
    uint32_t low;
    uint32_t high;
    uint32_t code;
    FILE *f;
} FastBitCoder;

static inline void init_encoder(FastBitCoder *enc, FILE *f)
{
    enc->low  = 0;
    enc->high = 0xFFFFFFFFU;
    enc->f    = f;
}

static inline void encode_bit(FastBitCoder *enc, int bit, int prob)
{
    uint32_t range = enc->high - enc->low;
    uint32_t split = (uint32_t)(((uint64_t)range * prob) >> 16);
    if (bit) {
        enc->high = enc->low + split;
    } else {
        enc->low  = enc->low + split + 1;
    }
    while ((enc->low ^ enc->high) < 0x01000000U) {
        fputc((int)(enc->low >> 24), enc->f);
        enc->low <<= 8;
        enc->high = (enc->high << 8) | 0xFFU;
    }
}

static inline void flush_encoder(FastBitCoder *enc)
{
    for (int i = 0; i < 4; ++i) {
        fputc((int)(enc->low >> 24), enc->f);
        enc->low <<= 8;
    }
}

static inline void init_decoder(FastBitCoder *dec, FILE *f)
{
    dec->low  = 0;
    dec->high = 0xFFFFFFFFU;
    dec->f    = f;
    dec->code = 0;
    for (int i = 0; i < 4; ++i) {
        int c = fgetc(f);
        if (c == EOF) c = 0;
        dec->code = (dec->code << 8) | (uint8_t)c;
    }
}

static inline int decode_bit(FastBitCoder *dec, int prob)
{
    uint32_t range = dec->high - dec->low;
    uint32_t split = (uint32_t)(((uint64_t)range * prob) >> 16);
    uint32_t abs_split = dec->low + split;
    int bit;
    if (dec->code <= abs_split) {
        dec->high = abs_split;
        bit = 1;
    } else {
        dec->low  = abs_split + 1;
        bit = 0;
    }
    while ((dec->low ^ dec->high) < 0x01000000U) {
        dec->low  <<= 8;
        dec->high = (dec->high << 8) | 0xFFU;
        int c = fgetc(dec->f);
        if (c == EOF) c = 0;
        dec->code = (dec->code << 8) | (uint8_t)c;
    }
    return bit;
}

int main(int argc, char **argv)
{
    if (argc < 4) return 1;
    W = (int16_t *)calloc(HASH_SIZE, sizeof(int16_t));
    lookup1 = (uint32_t *)calloc(1 << 25, sizeof(uint32_t));
    lookup2 = (uint32_t *)calloc(1 << 25, sizeof(uint32_t));
    if (!W || !lookup1 || !lookup2) return 1;
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
    FastBitCoder coder;
    if (is_compress) init_encoder(&coder, fout);
    else             init_decoder(&coder, fin);
    uint32_t match_pos1 = 0, match_pos2 = 0;
    int match_len1 = 0, match_len2 = 0;
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
        uint32_t h5 = 0;
        if (pos >= 5) {
            h5 = (B1 * 2654435761U + B2 * 2246822519U + 
                  B3 * 3266489917U + B4 * 668265263U + B5 * 19349669U) & 0x1FFFFFF;
        }
        if (match_len1 > 0) {
            if (pos >= 1 && file_data[match_pos1] == B1) {
                match_len1++;
                match_pos1++;
            } else match_len1 = 0;
        }
        if (match_len2 > 0) {
            if (pos >= 1 && file_data[match_pos2] == B1) {
                match_len2++;
                match_pos2++;
            } else match_len2 = 0;
        }
        if (pos >= 5) {
            if (match_len1 == 0) {
                uint32_t nm1 = lookup1[h5];
                if (nm1 > 0 && nm1 < pos) {
                    if (file_data[nm1-1] == B1 && file_data[nm1-2] == B2 && file_data[nm1-3] == B3) {
                        match_pos1 = nm1;
                        match_len1 = 1;
                    }
                }
            }
            if (match_len2 == 0) {
                uint32_t nm2 = lookup2[h5];
                if (nm2 > 0 && nm2 < pos) {
                    if (file_data[nm2-1] == B1 && file_data[nm2-2] == B2 && file_data[nm2-3] == B3) {
                        match_pos2 = nm2;
                        match_len2 = 1;
                    }
                }
            }
            lookup2[h5] = lookup1[h5];
            lookup1[h5] = pos;
        }
        int c1 = 1;
        for (int bit_idx = 7; bit_idx >= 0; --bit_idx) {
            int mb1 = 2, mb2 = 2;
            if (match_len1 > 0 && match_pos1 < pos) mb1 = (file_data[match_pos1] >> bit_idx) & 1;
            if (match_len2 > 0 && match_pos2 < pos) mb2 = (file_data[match_pos2] >> bit_idx) & 1;
            uint32_t F[28];
            F[0] = H(0, c1, 0, 0);
            F[1] = H(1, c1, B1, 0);
            F[2] = H(2, c1, B1, B2);
            F[3] = H(3, c1, B1, B3) ^ H(33, B2, B4, 0);
            F[4] = H(4, c1, B1, B2) ^ H(44, B3, B4, B5);
            F[5] = H(5, c1, B1, B2) ^ H(55, B3, B4, B5) ^ H(555, B6, B7, B8);
            F[6] = H(6, c1, mb1, MIN(match_len1, 255));
            F[7] = H(7, c1, mb2, MIN(match_len2, 255));
            F[8] = H(8, c1, mb1, B1);
            F[9] = H(9, c1, mb2, B2);
            F[10] = H(10, c1, word_hash & 0xFF, (word_hash >> 8) & 0xFF);
            F[11] = H(11, c1, B1, word_hash & 0xFF);
            F[12] = H(12, c1, B2, B5);
            F[13] = H(13, c1, B3, B6);
            F[14] = H(14, c1, B4, B8);
            F[15] = H(15, c1, B1, mb1);
            F[16] = H(16, c1, mb1, mb2);
            F[17] = H(17, c1, B1, B3) ^ H(177, B5, B7, 0);
            F[18] = H(18, c1, B2, B4) ^ H(188, B6, B8, 0);
            F[19] = H(19, c1, B1, B4) ^ H(199, B2, B5, 0);
            F[20] = H(20, c1, mb1, word_hash & 0xFF);
            F[21] = H(21, c1, B2, word_hash & 0xFF);
            F[22] = H(22, c1, B3, word_hash & 0xFF);
            F[23] = H(23, c1, B1, B2) ^ H(233, B4, B5, B6);
            F[24] = H(24, c1, B2, B3) ^ H(244, B5, B6, B7);
            F[25] = H(25, c1, mb2, word_hash & 0xFF);
            F[26] = H(26, c1, mb1, B3);
            F[27] = H(27, c1, mb2, B4);
            int sum = 0;
            for (int i = 0; i < 28; i++) {
                sum += W[F[i]];
            }
            int prob_nn = squish_sigmoid(sum);
            int q = prob_nn >> 7;
            int p1 = apm1[c1][q];
            int p2 = apm2[B1][q];
            int p3 = apm3[(B1 << 8) | c1][q];
            int p4 = apm4[(B2 << 8) | c1][q];
            int pm1 = apm_m1[mb1][q];
            int pm2 = apm_m2[mb2][q];
            int final_prob = (prob_nn * 10 + p1 + p2 + p3 * 2 + p4 * 2 + pm1 * 2 + pm2) / 19;
            if (final_prob < 1) final_prob = 1;
            if (final_prob > 4094) final_prob = 4094;
            int prob16 = final_prob << 4;
            if (prob16 < 1) prob16 = 1;
            if (prob16 > 65534) prob16 = 65534;
            int bit;
            if (is_compress) {
                bit = (ch >> bit_idx) & 1;
                encode_bit(&coder, bit, prob16);
            } else {
                bit = decode_bit(&coder, prob16);
                ch |= (bit << bit_idx);
            }
            int err = (bit << 12) - final_prob;
            int delta = err >> 5;
            for (int i = 0; i < 28; i++) {
                int32_t val = W[F[i]] + delta;
                if (val > 32767) val = 32767;
                else if (val < -32768) val = -32768;
                W[F[i]] = (int16_t)val;
            }
            int bit_scaled = bit << 12;
            apm1[c1][q] += (bit_scaled - apm1[c1][q]) >> 4;
            apm2[B1][q] += (bit_scaled - apm2[B1][q]) >> 4;
            apm3[(B1 << 8) | c1][q] += (bit_scaled - apm3[(B1 << 8) | c1][q]) >> 4;
            apm4[(B2 << 8) | c1][q] += (bit_scaled - apm4[(B2 << 8) | c1][q]) >> 4;
            apm_m1[mb1][q] += (bit_scaled - apm_m1[mb1][q]) >> 4;
            apm_m2[mb2][q] += (bit_scaled - apm_m2[mb2][q]) >> 4;
            c1 = (c1 << 1) | bit;
        }
        file_data[pos] = ch;
        if (!is_compress) {
            fputc(ch, fout);
        }
        if ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9')) {
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
    fclose(fin);
    fclose(fout);
    return 0;
}