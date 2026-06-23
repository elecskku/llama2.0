#pragma once
#include <stdint.h>
#include <ap_int.h>
#include <hls_math.h>
#include "config.h"

// W(D×N) · x(N×1) -> y(D×1)
// xq: [N], xs: [N/GS], wq: [D*N], ws: [D*(N/GS)]
template <int N, int D>
void matmul(float * __restrict xout,
            const int8_t * __restrict xq,
            const float  * __restrict xs,
            const int8_t * __restrict wq,
            const float  * __restrict ws)
{
    constexpr int GROUPS = N / GS;

    // x / xs 프리로드 - 모든 행에서 재사용되므로 레지스터화
    int8_t xbuf[N];
    #pragma HLS ARRAY_PARTITION variable=xbuf cyclic factor=GS dim=1
    float xsbuf[GROUPS];
    #pragma HLS ARRAY_PARTITION variable=xsbuf complete

load_x:
    for (int j = 0; j < N; j++) {
        #pragma HLS PIPELINE II=1
        xbuf[j] = xq[j];
    }
load_xs:
    for (int g = 0; g < GROUPS; g++) {
        #pragma HLS PIPELINE II=1
        xsbuf[g] = xs[g];
    }

RowLoop:
    for (int i = 0; i < D; i++) {
        #pragma HLS loop_tripcount min=256 max=4096 avg=768
        float val = 0.f;

    GroupLoop:
        for (int g = 0; g < GROUPS; g++) {
            #pragma HLS PIPELINE II=GS
            #pragma HLS loop_tripcount min=4 max=64 avg=12
            int32_t acc = 0;

        KLoop:
            for (int k = 0; k < GS; k++) {
                #pragma HLS UNROLL
                acc += (int32_t)xbuf[g * GS + k] * (int32_t)wq[i * N + g * GS + k];
            }
            val += (float)acc * ws[i * GROUPS + g] * xsbuf[g];
        }

        xout[i] = val;
    }
}