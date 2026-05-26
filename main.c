#include <stdbool.h>
#include <math.h>
#include <immintrin.h>
#include <assert.h>
#include <limits.h>

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>

/* ---------- Configuration ---------- */
#define TABLE_BITS 28
#define TABLE_SIZE  (1U << TABLE_BITS)
#define TABLE_MASK  (TABLE_SIZE - 1)
#define NUM_MIXERS 9                 /* increased from 8 to 9 mixers */
#define BUF_SIZE   (1 << 20)         /* 1 MiB buffers */

/* ---------- Sigmoid for probability estimation ---------- */
static inline int squish_sigmoid(int prediction)
{
    /* shift right by 5 for finer resolution */
    prediction >>= 5;                   /* scale to [-2048, 2047] */
    if (prediction < -2047) return 1;    /* 1/4096 */
    if (prediction > 2047)  return 4094;  /* 4094/4096 */
    return prediction + 2048;            /* 1..4094 */
}

/* ---------- Fast Bit Coder (Range Coding) ---------- */
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
    uint32_t range  = enc->high - enc->low;
    uint32_t split  = (range >> 12) * (uint32_t)prob;

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
    uint32_t range  = dec->high - dec->low;
    uint32_t split  = (range >> 12) * (uint32_t)prob;

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

/* ---------- Weight Tables ---------- */
static int32_t *weights[NUM_MIXERS];

static int32_t weights_context[256][NUM_MIXERS];
static int32_t weights_context2[256][NUM_MIXERS];
static int32_t weights_context3[256][2][NUM_MIXERS];
static int32_t weights_context4[256][NUM_MIXERS];
static int32_t weights_context5[65536][NUM_MIXERS];

/* ---------- Global State ---------- */
static uint8_t c_last_byte = 0;
static uint8_t c_prev_byte = 0;
static int last_bit = 0;

/* ---------- Helper: Clip weight updates to avoid overflow ---------- */
static inline void upd_weight(int32_t *w, int delta)
{
    int32_t val = *w + delta;
    if (val > (1 << 28)) val = (1 << 28);
    else if (val < -(1 << 28)) val = -(1 << 28);
    *w = val;
}

/* ---------- Bit Processing Macros ---------- */
#define PROCESS_BIT_COMPRESS(bit_idx, ch, c1, c2, c3, c4, c5, c6, enc)           \
    do {                                                                      \
        int bit = (ch >> (bit_idx)) & 1;                                      \
        uint32_t h1 = (c1 * 0x12345678UL) & TABLE_MASK;                      \
        uint32_t h2 = (c1 ^ (c2 * 0x23456789UL)) & TABLE_MASK;               \
        uint32_t h3 = (c1 ^ (c3 * 0x34567890UL) ^ (c2 << 8)) & TABLE_MASK;   \
        uint32_t h4 = (c1 ^ (c4 * 0x45678901UL) ^ (c3 << 8) ^ (c2 << 16)) & TABLE_MASK; \
        uint32_t h5 = (c1 ^ (c5 * 0x56789012UL) ^ (c4 << 8) ^ (c3 << 16) ^ (c2 << 24)) & TABLE_MASK; \
        uint32_t h6 = (c1 ^ ((bit_idx << 16) * 0x67890123UL) ^ (c2 << 8) ^ (c3 << 16)) & TABLE_MASK; \
        uint32_t h7 = (c1 ^ ((bit_idx << 8) * 0x78901234UL) ^ (c4 << 8) ^ (c5 << 16)) & TABLE_MASK; \
        uint32_t h8 = (c1 ^ (c2 * 0x89012345UL) ^ (c6 * 0x90123456UL) ^ (bit_idx << 24)) & TABLE_MASK; \
        uint32_t h9 = (c1 ^ (c2 * 0x23456789UL) ^ (c3 * 0x34567890UL) ^ (c4 * 0x45678901UL) ^ (c5 * 0x56789012UL) ^ (c6 * 0x67890123UL)) & TABLE_MASK; \
        int ctx_idx = ((c3 << 8) | c2) & 0xFFFF;                              \
        int64_t pred64 = (int64_t)weights[0][h1] + \
                         (int64_t)weights[1][h2] * 3 + \
                         (int64_t)weights[2][h3] * 5 + \
                         (int64_t)weights[3][h4] * 7 + \
                         (int64_t)weights[4][h5] * 9 + \
                         (int64_t)weights[5][h6] * 11 + \
                         (int64_t)weights[6][h7] * 13 + \
                         (int64_t)weights[7][h8] * 15 + \
                         (int64_t)weights[8][h9] * 17 + \
                         (int64_t)weights_context[c_last_byte][bit_idx] + \
                         (int64_t)weights_context2[c2][bit_idx] + \
                         (int64_t)weights_context3[c_last_byte][last_bit][bit_idx] + \
                         (int64_t)weights_context4[c3][bit_idx] + \
                         (int64_t)weights_context5[ctx_idx][bit_idx]; \
        int pred; \
        if (pred64 > INT_MAX) pred = INT_MAX; \
        else if (pred64 < INT_MIN) pred = INT_MIN; \
        else pred = (int)pred64; \
        int prob = squish_sigmoid(pred);                                      \
        encode_bit(enc, bit, prob);                                           \
        int error = (bit << 12) - prob;                                       \
        int delta = error / 32;                                               \
        upd_weight(&weights[0][h1], delta);                                   \
        upd_weight(&weights[1][h2], delta);                                   \
        upd_weight(&weights[2][h3], delta);                                   \
        upd_weight(&weights[3][h4], delta);                                   \
        upd_weight(&weights[4][h5], delta);                                   \
        upd_weight(&weights[5][h6], delta);                                   \
        upd_weight(&weights[6][h7], delta);                                   \
        upd_weight(&weights[7][h8], delta);                                   \
        upd_weight(&weights[8][h9], delta);                                   \
        upd_weight(&weights_context[c_last_byte][bit_idx], delta);             \
        upd_weight(&weights_context2[c2][bit_idx], delta);                    \
        upd_weight(&weights_context3[c_last_byte][last_bit][bit_idx], delta);  \
        upd_weight(&weights_context4[c3][bit_idx], delta);                    \
        upd_weight(&weights_context5[ctx_idx][bit_idx], delta);                \
        last_bit = bit;                                                      \
        c1 = (c1 << 1) | bit;                                                \
    } while (0)

#define PROCESS_BIT_DECOMPRESS(bit_idx, ch, c1, c2, c3, c4, c5, c6, dec)       \
    do {                                                                      \
        uint32_t h1 = (c1 * 0x12345678UL) & TABLE_MASK;                      \
        uint32_t h2 = (c1 ^ (c2 * 0x23456789UL)) & TABLE_MASK;               \
        uint32_t h3 = (c1 ^ (c3 * 0x34567890UL) ^ (c2 << 8)) & TABLE_MASK;   \
        uint32_t h4 = (c1 ^ (c4 * 0x45678901UL) ^ (c3 << 8) ^ (c2 << 16)) & TABLE_MASK; \
        uint32_t h5 = (c1 ^ (c5 * 0x56789012UL) ^ (c4 << 8) ^ (c3 << 16) ^ (c2 << 24)) & TABLE_MASK; \
        uint32_t h6 = (c1 ^ ((bit_idx << 16) * 0x67890123UL) ^ (c2 << 8) ^ (c3 << 16)) & TABLE_MASK; \
        uint32_t h7 = (c1 ^ ((bit_idx << 8) * 0x78901234UL) ^ (c4 << 8) ^ (c5 << 16)) & TABLE_MASK; \
        uint32_t h8 = (c1 ^ (c2 * 0x89012345UL) ^ (c6 * 0x90123456UL) ^ (bit_idx << 24)) & TABLE_MASK; \
        uint32_t h9 = (c1 ^ (c2 * 0x23456789UL) ^ (c3 * 0x34567890UL) ^ (c4 * 0x45678901UL) ^ (c5 * 0x56789012UL) ^ (c6 * 0x67890123UL)) & TABLE_MASK; \
        int ctx_idx = ((c3 << 8) | c2) & 0xFFFF;                              \
        int64_t pred64 = (int64_t)weights[0][h1] + \
                         (int64_t)weights[1][h2] * 3 + \
                         (int64_t)weights[2][h3] * 5 + \
                         (int64_t)weights[3][h4] * 7 + \
                         (int64_t)weights[4][h5] * 9 + \
                         (int64_t)weights[5][h6] * 11 + \
                         (int64_t)weights[6][h7] * 13 + \
                         (int64_t)weights[7][h8] * 15 + \
                         (int64_t)weights[8][h9] * 17 + \
                         (int64_t)weights_context[c_last_byte][bit_idx] + \
                         (int64_t)weights_context2[c2][bit_idx] + \
                         (int64_t)weights_context3[c_last_byte][last_bit][bit_idx] + \
                         (int64_t)weights_context4[c3][bit_idx] + \
                         (int64_t)weights_context5[ctx_idx][bit_idx]; \
        int pred; \
        if (pred64 > INT_MAX) pred = INT_MAX; \
        else if (pred64 < INT_MIN) pred = INT_MIN; \
        else pred = (int)pred64; \
        int prob = squish_sigmoid(pred);                                      \
        int bit = decode_bit(dec, prob);                                      \
        ch |= (bit << (bit_idx));                                            \
        int error = (bit << 12) - prob;                                       \
        int delta = error / 32;                                               \
        upd_weight(&weights[0][h1], delta);                                   \
        upd_weight(&weights[1][h2], delta);                                   \
        upd_weight(&weights[2][h3], delta);                                   \
        upd_weight(&weights[3][h4], delta);                                   \
        upd_weight(&weights[4][h5], delta);                                   \
        upd_weight(&weights[5][h6], delta);                                   \
        upd_weight(&weights[6][h7], delta);                                   \
        upd_weight(&weights[7][h8], delta);                                   \
        upd_weight(&weights[8][h9], delta);                                   \
        upd_weight(&weights_context[c_last_byte][bit_idx], delta);             \
        upd_weight(&weights_context2[c2][bit_idx], delta);                    \
        upd_weight(&weights_context3[c_last_byte][last_bit][bit_idx], delta);  \
        upd_weight(&weights_context4[c3][bit_idx], delta);                    \
        upd_weight(&weights_context5[ctx_idx][bit_idx], delta);                \
        last_bit = bit;                                                      \
        c1 = (c1 << 1) | bit;                                                \
    } while (0)

/* ---------- Main ---------- */
int main(int argc, char **argv)
{
    if (argc < 4) {
        printf("Usage: %s [c|d] input output\n", argv[0]);
        return 1;
    }

    /* Allocate weight tables */
    for (int i = 0; i < NUM_MIXERS; ++i) {
        weights[i] = (int32_t *)malloc(TABLE_SIZE * sizeof(int32_t));
        if (!weights[i]) {
            fprintf(stderr, "Memory allocation failed for weights[%d]\n", i);
            return 1;
        }
        memset(weights[i], 0, TABLE_SIZE * sizeof(int32_t));
    }

    clock_t start = clock();

    FILE *fin  = fopen(argv[2], "rb");
    FILE *fout = fopen(argv[3], "wb");
    if (!fin || !fout) {
        fprintf(stderr, "File I/O error\n");
        return 1;
    }

    /* Buffer for I/O */
    unsigned char inbuf[BUF_SIZE];
    unsigned char outbuf[BUF_SIZE];
    setvbuf(fin,  inbuf, _IOFBF, sizeof(inbuf));
    setvbuf(fout, outbuf, _IOFBF, sizeof(outbuf));

    struct stat st;
    if (stat(argv[2], &st) != 0) {
        perror("stat");
        return 1;
    }
    long long file_size = st.st_size;

    if (argv[1][0] == 'c') {  /* compression */
        uint32_t size32 = (uint32_t)file_size;
        fwrite(&size32, 1, 4, fout);

        FastBitCoder encoder;
        init_encoder(&encoder, fout);

        uint32_t c1 = 0, c2 = 0, c3 = 0, c4 = 0, c5 = 0, c6 = 0;
        int ch;

        c_last_byte = 0;
        c_prev_byte = 0;
        last_bit = 0;

        size_t bytes_read;
        while ((bytes_read = fread(inbuf, 1, sizeof(inbuf), fin)) > 0) {
            for (size_t i = 0; i < bytes_read; ++i) {
                ch = inbuf[i];

                PROCESS_BIT_COMPRESS(7, ch, c1, c2, c3, c4, c5, c6, &encoder);
                PROCESS_BIT_COMPRESS(6, ch, c1, c2, c3, c4, c5, c6, &encoder);
                PROCESS_BIT_COMPRESS(5, ch, c1, c2, c3, c4, c5, c6, &encoder);
                PROCESS_BIT_COMPRESS(4, ch, c1, c2, c3, c4, c5, c6, &encoder);
                PROCESS_BIT_COMPRESS(3, ch, c1, c2, c3, c4, c5, c6, &encoder);
                PROCESS_BIT_COMPRESS(2, ch, c1, c2, c3, c4, c5, c6, &encoder);
                PROCESS_BIT_COMPRESS(1, ch, c1, c2, c3, c4, c5, c6, &encoder);
                PROCESS_BIT_COMPRESS(0, ch, c1, c2, c3, c4, c5, c6, &encoder);

                if (ch == c_last_byte) {
                    /* run length increases */
                } else {
                    /* run length reset */
                }
                c_prev_byte = c_last_byte;
                c_last_byte = (uint8_t)ch;

                /* shift context history */
                c6 = c5; c5 = c4; c4 = c3; c3 = c2; c2 = c1 & 0xFF;
            }
        }
        flush_encoder(&encoder);

        long long comp_size = ftell(fout);
        printf("Compressed: %lld -> %lld bytes in %.3f sec\n",
               file_size, comp_size, (double)(clock() - start) / CLOCKS_PER_SEC);
    } else if (argv[1][0] == 'd') {  /* decompression */
        uint32_t orig_size;
        fread(&orig_size, 1, 4, fin);

        FastBitCoder decoder;
        init_decoder(&decoder, fin);

        uint32_t c1 = 0, c2 = 0, c3 = 0, c4 = 0, c5 = 0, c6 = 0;
        int ch;

        c_last_byte = 0;
        c_prev_byte = 0;
        last_bit = 0;

        for (long long bytes = 0; bytes < orig_size; ++bytes) {
            ch = 0;

            PROCESS_BIT_DECOMPRESS(7, ch, c1, c2, c3, c4, c5, c6, &decoder);
            PROCESS_BIT_DECOMPRESS(6, ch, c1, c2, c3, c4, c5, c6, &decoder);
            PROCESS_BIT_DECOMPRESS(5, ch, c1, c2, c3, c4, c5, c6, &decoder);
            PROCESS_BIT_DECOMPRESS(4, ch, c1, c2, c3, c4, c5, c6, &decoder);
            PROCESS_BIT_DECOMPRESS(3, ch, c1, c2, c3, c4, c5, c6, &decoder);
            PROCESS_BIT_DECOMPRESS(2, ch, c1, c2, c3, c4, c5, c6, &decoder);
            PROCESS_BIT_DECOMPRESS(1, ch, c1, c2, c3, c4, c5, c6, &decoder);
            PROCESS_BIT_DECOMPRESS(0, ch, c1, c2, c3, c4, c5, c6, &decoder);

            fputc(ch, fout);

            if (ch == c_last_byte) {
                /* run length increases */
            } else {
                /* run length reset */
            }
            c_prev_byte = c_last_byte;
            c_last_byte = (uint8_t)ch;

            c6 = c5; c5 = c4; c4 = c3; c3 = c2; c2 = c1 & 0xFF;
        }

        printf("Decompressed: %u bytes restored in %.3f sec\n",
               orig_size, (double)(clock() - start) / CLOCKS_PER_SEC);
    } else {
        fprintf(stderr, "Unknown mode '%c'\n", argv[1][0]);
        return 1;
    }

    fclose(fin);
    fclose(fout);

    for (int i = 0; i < NUM_MIXERS; ++i) {
        free(weights[i]);
    }

    return 0;
}