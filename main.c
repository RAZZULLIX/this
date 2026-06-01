#include <immintrin.h>
#include <limits.h>

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <math.h>
#include <assert.h>

#define HASH_BITS 30
#define HASH_SIZE (1U << HASH_BITS)
#define HASH_MASK (HASH_SIZE - 1)
#define NUM_CTX 48

static int16_t *W;
static uint32_t *lookup1;
static uint32_t *lookup2;

static uint16_t *apm1;
static uint16_t *apm2;
static uint16_t *apm3;
static uint16_t *apm4;

static int squash[8192];

static void init_tables(void)
{
    for (int i = -4095; i <= 4095; ++i) {
        double p = 1.0 / (1.0 + exp(-i / 300.0));
        int val = (int)(p * 4095.0);
        if (val < 1) val = 1;
        if (val > 4094) val = 4094;
        squash[i + 4095] = val;
    }
    for (int i = 0; i < 65536; ++i) {
        apm1[i] = 2048;
        apm2[i] = 2048;
        apm3[i] = 2048;
        apm4[i] = 2048;
    }
}

static inline uint32_t H(uint32_t id, uint32_t v1, uint32_t v2, uint32_t v3)
{
    uint32_t h = id * 0x12345679U + v1 * 0x5bd1e995U + v2 * 0x27d4eb2dU + v3 * 0x924c290dU;
    h ^= h >> 15;
    h *= 0x85ebca6bU;
    h ^= h >> 13;
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
    uint32_t split = (range >> 12) * (uint32_t)prob;

    if (bit) {
        enc->high = enc->low + split;
    } else {
        enc->low  = enc->low + split + 1;
    }

    while ((enc->low ^ enc->high) < 0x01000000U) {
        fputc((int)(enc->low >> 24), enc->f);
        enc->low  <<= 8;
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
    uint32_t split = (range >> 12) * (uint32_t)prob;

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
    lookup1 = (uint32_t *)calloc(1 << 28, sizeof(uint32_t));
    lookup2 = (uint32_t *)calloc(1 << 28, sizeof(uint32_t));

    apm1 = (uint16_t *)calloc(65536, sizeof(uint16_t));
    apm2 = (uint16_t *)calloc(65536, sizeof(uint16_t));
    apm3 = (uint16_t *)calloc(65536, sizeof(uint16_t));
    apm4 = (uint16_t *)calloc(65536, sizeof(uint16_t));

    if (!W || !lookup1 || !lookup2 || !apm1 || !apm2 || !apm3 || !apm4) return 1;

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
        uint8_t B9 = pos >= 9 ? file_data[pos-9] : 0;
        uint8_t B10 = pos >= 10 ? file_data[pos-10] : 0;
        uint8_t B11 = pos >= 11 ? file_data[pos-11] : 0;
        uint8_t B12 = pos >= 12 ? file_data[pos-12] : 0;
        uint8_t B13 = pos >= 13 ? file_data[pos-13] : 0;
        uint8_t B14 = pos >= 14 ? file_data[pos-14] : 0;
        uint8_t B15 = pos >= 15 ? file_data[pos-15] : 0;
        uint8_t B16 = pos >= 16 ? file_data[pos-16] : 0;

        uint32_t h4 = 0, h6 = 0;
        if (pos >= 4) {
            h4 = (B1 * 2654435761U + B2 * 2246822519U +
                  B3 * 3266489917U + B4 * 668265263U) & ((1U << 28) - 1);
        }
        if (pos >= 6) {
            h6 = (h4 * 19349669U + B5 * 83492791U + B6 * 1192837U) & ((1U << 28) - 1);
        }

        if (match_len1 > 0) {
            if (pos >= 1 && file_data[match_pos1] == B1) {
                match_len1++;
                match_pos1++;
            } else {
                match_len1 = 0;
            }
        }
        if (match_len2 > 0) {
            if (pos >= 1 && file_data[match_pos2] == B1) {
                match_len2++;
                match_pos2++;
            } else {
                match_len2 = 0;
            }
        }

        if (pos >= 6) {
            if (match_len1 == 0) {
                uint32_t nm1 = lookup1[h4];
                if (nm1 > 0 && nm1 < pos) {
                    if (file_data[nm1-1] == B1 && file_data[nm1-2] == B2 && file_data[nm1-3] == B3) {
                        match_pos1 = nm1;
                        match_len1 = 1;
                    }
                }
            }
            if (match_len2 == 0) {
                uint32_t nm2 = lookup2[h6];
                if (nm2 > 0 && nm2 < pos) {
                    if (file_data[nm2-1] == B1 && file_data[nm2-2] == B2 && file_data[nm2-3] == B3) {
                        match_pos2 = nm2;
                        match_len2 = 1;
                    }
                }
            }
            lookup1[h4] = pos;
            lookup2[h6] = pos;
        }

        uint32_t dist1 = 0, dist2 = 0;
        if (match_len1 > 0 && match_pos1 < pos) dist1 = pos - match_pos1;
        if (match_len2 > 0 && match_pos2 < pos) dist2 = pos - match_pos2;

        int c1 = 1;
        for (int bit_idx = 7; bit_idx >= 0; --bit_idx) {
            int mb1 = 2, mb2 = 2;
            if (match_len1 > 0 && match_pos1 < pos) mb1 = (file_data[match_pos1] >> bit_idx) & 1;
            if (match_len2 > 0 && match_pos2 < pos) mb2 = (file_data[match_pos2] >> bit_idx) & 1;

            int F[NUM_CTX];
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
            F[19] = H(19, c1, B3, B4) ^ H(109, B5, B6, 0) ^ H(1099, B7, B8, B9);
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
            F[30] = H(30, c1, (uint32_t)(word_hash & 0xFF), 0);
            F[31] = H(31, c1, (uint32_t)((word_hash >> 8) & 0xFF), 0);
            F[32] = H(32, c1, (uint32_t)(word_hash & 0xFF), B1);
            F[33] = H(33, c1, B1 & 0xDF, B2 & 0xDF);
            F[34] = H(34, c1, B1 & 0xDF, (uint32_t)(word_hash & 0xFF));
            F[35] = H(35, c1, B1, B5) ^ H(355, B9, B13, 0);
            F[36] = H(36, c1, B2, B6) ^ H(366, B10, B14, 0);
            F[37] = H(37, c1, B3, B7) ^ H(377, B11, B15, 0);
            F[38] = H(38, c1, mb1, (uint32_t)(word_hash & 0xFF));
            F[39] = H(39, c1, mb2, (uint32_t)(word_hash & 0xFF));
            F[40] = H(40, c1, B1, B3) ^ H(400, B2, B4, 0);
            F[41] = H(41, c1, B5, B10) ^ H(411, B15, 0, 0);
            F[42] = H(42, c1, dist1, mb1);
            F[43] = H(43, c1, dist2, mb2);
            F[44] = H(44, c1, dist1 >> 8, mb1);
            F[45] = H(45, c1, dist2 >> 8, mb2);
            F[46] = H(46, c1, B1 & 0xE0, B2 & 0xE0);
            F[47] = H(47, c1, B1 & 0xF0, B2 & 0xF0) ^ H(477, B3 & 0xF0, 0, 0);

            int sum = 0;
            for (int i = 0; i < NUM_CTX; ++i) {
                sum += W[F[i]];
            }

            int prob_nn = squish_sigmoid(sum);
            int p_bin = prob_nn >> 4;

            int apm1_val = apm1[(B1 << 8) | p_bin];
            int apm2_val = apm2[(c1 << 8) | p_bin];
            int apm3_val = apm3[(B2 << 8) | p_bin];
            int apm4_val = apm4[(B3 << 8) | p_bin];

            int final_prob = (prob_nn * 4 + apm1_val * 2 + apm2_val * 2 + apm3_val + apm4_val) / 10;
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
            int delta = err / 32;

            if (delta != 0) {
                for (int i = 0; i < NUM_CTX; ++i) {
                    int32_t val = W[F[i]] + delta;
                    if (val > 32767) val = 32767;
                    else if (val < -32768) val = -32768;
                    W[F[i]] = (int16_t)val;
                }
            }

            apm1[(B1 << 8) | p_bin] += (target - apm1_val) / 16;
            apm2[(c1 << 8) | p_bin] += (target - apm2_val) / 16;
            apm3[(B2 << 8) | p_bin] += (target - apm3_val) / 16;
            apm4[(B3 << 8) | p_bin] += (target - apm4_val) / 16;

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
    free(apm1);
    free(apm2);
    free(apm3);
    free(apm4);
    fclose(fin);
    fclose(fout);

    return 0;
}