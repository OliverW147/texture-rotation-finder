// height_match_gpu.cu -- CUDA surface-height pattern finder for MC 1.21 / 26.1.2.
//
// GPU counterpart of height_match.c, same CLI + same matches.txt format so the
// GUI drives both engines identically.
//
// FAST design (mirrors the CPU heightRegion lattice sharing, the thing that
// makes it ~26x faster than per-column): the expensive noise (base3d, 24
// octaves) is evaluated ONCE per 4-block lattice node, not per column. Three
// stages per Z-band tile:
//   (1) nodeKernel:   1 thread per lattice node -> fills that node's final_density
//                     Y-corner stack (49 corners) into global dens[].
//   (2) heightKernel: 1 thread per column -> reads its 4 surrounding node stacks,
//                     trilerps down for the exact surface Y (== heightRegion).
//   (3) matchKernel:  1 thread per candidate column -> reads precomputed heights
//                     for the anchor + every pattern offset (all orientations),
//                     appends matches. No noise work here.
// Bit-identical to the CPU finder (validated by diffing match sets).
//
// Build: build_gpu.bat height_match_gpu.exe height_match_gpu.cu

#include "height_gpu.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

extern "C" {
#include "height_exact.h"
}

#define CK(call) do { cudaError_t e=(call); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
    exit(2);} } while(0)

// ---- cell lattice (must match height_exact.c) ----
#define G_CELL_XZ 4
#define G_CELL_Y  8
#define G_TOPY    312
#define G_BOTY    (-64)
#define G_NYCORNER (((G_TOPY - G_BOTY)/G_CELL_Y) + 2)  // 49
#define G_SCAN_TOP 256

// ---- host packer (standalone; never linked with height_gpu.cu) ----
static int packSpline(DevHeight *H, const Spline *sp) {
    int idx = H->poolLen++;
    if (H->poolLen > SPL_POOL_MAX) { fprintf(stderr,"spline pool overflow\n"); exit(2); }
    DSplineNode *n = &H->pool[idx];
    if (sp->len == 1) { n->len=1; n->typ=0; n->fixedVal=((const FixSpline*)sp)->val; return idx; }
    n->len = sp->len; n->typ = sp->typ; n->fixedVal = 0.0f;
    int kids[SPL_MAX_CHILD]; float loc[SPL_MAX_CHILD], der[SPL_MAX_CHILD];
    for (int i=0;i<sp->len;i++){ loc[i]=sp->loc[i]; der[i]=sp->der[i]; kids[i]=packSpline(H,sp->val[i]); }
    n = &H->pool[idx];
    for (int i=0;i<sp->len;i++){ n->loc[i]=loc[i]; n->der[i]=der[i]; n->child[i]=kids[i]; }
    return idx;
}
static void packPerlin(DPerlin *d, const PerlinNoise *s) {
    for (int i=0;i<257;i++) d->d[i]=s->d[i];
    d->a=s->a; d->b=s->b; d->c=s->c; d->amplitude=s->amplitude; d->lacunarity=s->lacunarity;
}
static void packDoublePerlin(DDoublePerlin *d, const DoublePerlinNoise *s) {
    d->amplitude=s->amplitude; d->octcntA=s->octA.octcnt; d->octcntB=s->octB.octcnt;
    if (d->octcntA>DPN_MAX_OCT||d->octcntB>DPN_MAX_OCT){fprintf(stderr,"oct overflow\n");exit(2);}
    for (int i=0;i<d->octcntA;i++) packPerlin(&d->octA[i], &s->octA.octaves[i]);
    for (int i=0;i<d->octcntB;i++) packPerlin(&d->octB[i], &s->octB.octaves[i]);
}
static void hostPack(DevHeight *H, HeightGen *hg) {
    memset(H,0,sizeof(*H));
    for (int i=0;i<16;i++) packPerlin(&H->octmin[i], &hg->sn.octmin.octaves[i]);
    for (int i=0;i<16;i++) packPerlin(&H->octmax[i], &hg->sn.octmax.octaves[i]);
    for (int i=0;i<8 ;i++) packPerlin(&H->octmain[i],&hg->sn.octmain.octaves[i]);
    H->xzScale=hg->sn.xzScale; H->yScale=hg->sn.yScale; H->xzFactor=hg->sn.xzFactor; H->yFactor=hg->sn.yFactor;
    BiomeNoise *bn=&hg->g->bn;
    packDoublePerlin(&H->shift,&bn->climate[NP_SHIFT]);
    packDoublePerlin(&H->cont, &bn->climate[NP_CONTINENTALNESS]);
    packDoublePerlin(&H->ero,  &bn->climate[NP_EROSION]);
    packDoublePerlin(&H->weird,&bn->climate[NP_WEIRDNESS]);
    packDoublePerlin(&H->jagged,&hg->jaggedNoise);
    H->poolLen=0;
    H->rootOffset=packSpline(H,hg->spOffset);
    H->rootFactor=packSpline(H,hg->spFactor);
    H->rootJagged=packSpline(H,hg->spJaggedness);
}

// ---- pattern (device constant) ----
#define MAX_PATTERN 256
__constant__ int c_tpatdx[8][MAX_PATTERN];
__constant__ int c_tpatdz[8][MAX_PATTERN];
__constant__ int c_path[MAX_PATTERN];
__constant__ int c_patCount;
__constant__ int c_anchorH;

struct GMatch { int cx, cz, orient, anchorY; };

static inline int fdivh(int a,int b){int q=a/b;if((a%b)!=0&&((a<0)!=(b<0)))q--;return q;}

// ---- stage 1: one thread per lattice node -> fill its corner stack ----
// nodes laid out [zi*nNx + xi], each with G_NYCORNER doubles in dens[].
__global__ void nodeKernel(const DevHeight *H, int nx0, int nz0, int nNx, int nNz, double *dens) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)nNx * nNz) return;
    int xi = (int)(idx % nNx), zi = (int)(idx / nNx);
    int bx = nx0 + xi*G_CELL_XZ, bz = nz0 + zi*G_CELL_XZ;
    DColumnShape s; d_column_shape(H, bx, bz, &s);
    double *col = &dens[idx * G_NYCORNER];
    const int topYi = (G_SCAN_TOP - G_BOTY)/G_CELL_Y;
    for (int yi = topYi+1; yi < G_NYCORNER; yi++) col[yi] = -1.0;
    for (int yi = 0; yi <= topYi; yi++) col[yi] = d_final_density(H, &s, bx, G_BOTY + yi*G_CELL_Y, bz);
    if (col[topYi] >= 0.0)
        for (int yi = topYi+1; yi < G_NYCORNER; yi++) col[yi] = d_final_density(H, &s, bx, G_BOTY + yi*G_CELL_Y, bz);
}

// floor-div by 4 then *4 (lower node block coord), device helper.
__device__ __forceinline__ int fdivh_dev(int a){ int q=a/G_CELL_XZ; if((a%G_CELL_XZ)!=0 && (a<0)) q--; return q*G_CELL_XZ; }

// ---- stage 2: one thread per column -> trilerp height from node stacks ----
// columns cover [x0, x0+W) x [z0, z0+Hc); heights written row-major into heights[].
__global__ void heightKernel(const double *dens, int nx0, int nz0, int nNx,
        int x0, int z0, int W, int Hc, int *heights) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)W * Hc) return;
    int lx = (int)(idx % W), lz = (int)(idx / W);
    int x = x0 + lx, z = z0 + lz;
    int xi = (fdivh_dev(x) - nx0)/G_CELL_XZ;
    int zi = (fdivh_dev(z) - nz0)/G_CELL_XZ;
    double fx = (x - (nx0 + xi*G_CELL_XZ)) / (double)G_CELL_XZ;
    double fz = (z - (nz0 + zi*G_CELL_XZ)) / (double)G_CELL_XZ;
    const double *c00 = &dens[((long)zi*nNx + xi)*G_NYCORNER];
    const double *c01 = &dens[((long)(zi+1)*nNx + xi)*G_NYCORNER];
    const double *c10 = &dens[((long)zi*nNx + (xi+1))*G_NYCORNER];
    const double *c11 = &dens[((long)(zi+1)*nNx + (xi+1))*G_NYCORNER];
    int found = G_BOTY;
    for (int yi = G_NYCORNER-2; yi >= 0; yi--) {
        int yc0 = G_BOTY + yi*G_CELL_Y, yc1 = yc0 + G_CELL_Y;
        double d000=c00[yi], d010=c00[yi+1];
        double d001=c01[yi], d011=c01[yi+1];
        double d100=c10[yi], d110=c10[yi+1];
        double d101=c11[yi], d111=c11[yi+1];
        int done=0;
        for (int y = yc1-1; y >= yc0; y--) {
            if (y > 319 || y < -64) continue;
            double fy = (y - yc0)/(double)G_CELL_Y;
            double dx00=d_lerp(fy,d000,d010), dx01=d_lerp(fy,d001,d011);
            double dx10=d_lerp(fy,d100,d110), dx11=d_lerp(fy,d101,d111);
            double dz0=d_lerp(fz,dx00,dx01), dz1=d_lerp(fz,dx10,dx11);
            double d=d_lerp(fx,dz0,dz1);
            if (d >= 0.0) { found=y; done=1; break; }
        }
        if (done) break;
    }
    heights[idx] = found;
}

// ---- stage 3: one thread per candidate column -> test pattern ----
// heights[] covers the tile core + reach halo: width hW, origin (hx0,hz0).
__global__ void matchKernel(const int *heights, int hx0, int hz0, int hW,
        int coreX0, int coreZ0, int coreW, int coreH,
        int centX, int centZ, long long rSq,
        int tStart, int tEnd, int absolute, int tol,
        GMatch *out, int *outCount, int maxOut) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)coreW * coreH) return;
    int cx = coreX0 + (int)(idx % coreW);
    int cz = coreZ0 + (int)(idx / coreW);
    long long ddx=(long long)cx-centX, ddz=(long long)cz-centZ;
    if (ddx*ddx + ddz*ddz > rSq) return;
    for (int t = tStart; t <= tEnd; t++) {
        int a0x = cx + c_tpatdx[t][0], a0z = cz + c_tpatdz[t][0];
        int candAnchor = heights[(a0z - hz0)*(long)hW + (a0x - hx0)];
        int ok = 1;
        for (int p = 0; p < c_patCount; p++) {
            int hx = cx + c_tpatdx[t][p], hz = cz + c_tpatdz[t][p];
            int hv = heights[(hz - hz0)*(long)hW + (hx - hx0)];
            int want = c_path[p];
            int diff = absolute ? (hv - want) : ((hv - candAnchor) - (want - c_anchorH));
            if (diff < -tol || diff > tol) { ok = 0; break; }
        }
        if (ok) {
            int ri = atomicAdd(outCount, 1);
            if (ri < maxOut) { out[ri].cx=cx; out[ri].cz=cz; out[ri].orient=t; out[ri].anchorY=candAnchor; }
        }
    }
}

static void usage(const char*p){
    fprintf(stderr,
      "Usage: %s <seed> <specDir> <cx> <cz> <radiusBlocks> [options] <dx,dz:height>...\n"
      "  --absolute      match exact Y (default: relative relief)\n"
      "  --tol N         per-column tolerance (default 1)\n"
      "  --orient N      0..7 single, or -1 all 8 (default -1)\n"
      "  --max N         stop after N matches\n", p);
}

int main(int argc, char **argv) {
    if (argc < 7) { usage(argv[0]); return 1; }
    uint64_t seed = (uint64_t)strtoll(argv[1],NULL,10);
    const char *spec = argv[2];
    int centX=atoi(argv[3]), centZ=atoi(argv[4]);
    long long radius=(long long)strtoll(argv[5],NULL,10);
    int absolute=0, tol=1, orientArg=-1; long long maxMatches=-1;

    int PAT_DX[MAX_PATTERN],PAT_DZ[MAX_PATTERN],PAT_H[MAX_PATTERN],PCOUNT=0;
    for (int i=6;i<argc;i++){
        if(!strcmp(argv[i],"--absolute")) absolute=1;
        else if(!strcmp(argv[i],"--tol")) tol=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--orient")) orientArg=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--threads")) { ++i; }
        else if(!strcmp(argv[i],"--max")) maxMatches=strtoll(argv[++i],NULL,10);
        else {
            int dx,dz,h;
            if(sscanf(argv[i],"%d,%d:%d",&dx,&dz,&h)!=3){fprintf(stderr,"bad point '%s'\n",argv[i]);return 1;}
            if(PCOUNT>=MAX_PATTERN){fprintf(stderr,"too many points\n");return 1;}
            PAT_DX[PCOUNT]=dx;PAT_DZ[PCOUNT]=dz;PAT_H[PCOUNT]=h;PCOUNT++;
        }
    }
    if(PCOUNT<2){fprintf(stderr,"need >=2 points\n");return 1;}

    int tStart=(orientArg<0)?0:orientArg, tEnd=(orientArg<0)?7:orientArg;
    int anchorH=PAT_H[0];

    int tpatdx[8][MAX_PATTERN], tpatdz[8][MAX_PATTERN];
    auto xf=[&](int t,int dx,int dz,int*ox,int*oz){
        switch(t){case 0:*ox=dx;*oz=dz;break;case 1:*ox=-dz;*oz=dx;break;
        case 2:*ox=-dx;*oz=-dz;break;case 3:*ox=dz;*oz=-dx;break;
        case 4:*ox=-dx;*oz=dz;break;case 5:*ox=dx;*oz=-dz;break;
        case 6:*ox=dz;*oz=dx;break;default:*ox=-dz;*oz=-dx;break;}};
    for(int t=0;t<8;t++)for(int p=0;p<PCOUNT;p++) xf(t,PAT_DX[p],PAT_DZ[p],&tpatdx[t][p],&tpatdz[t][p]);

    int reach=0;
    for(int p=0;p<PCOUNT;p++){int a=abs(PAT_DX[p]),b=abs(PAT_DZ[p]); if(a>reach)reach=a; if(b>reach)reach=b;}

    printf("Surface-height pattern matcher (GPU)\n");
    printf("Seed: %lld  Center: (%d,%d)  Radius: %lld blocks\n",(long long)seed,centX,centZ,radius);
    printf("Pattern: %d points  mode: %s  tol: +-%d  orient: %d..%d  reach: %d\n",
           PCOUNT, absolute?"absolute":"relative", tol, tStart, tEnd, reach);
    fflush(stdout);

    Generator g; setupGenerator(&g,MC_1_21_3,0); applySeed(&g,DIM_OVERWORLD,seed);
    HeightGen hg;
    if(heightGenInit(&hg,&g,spec)!=0){fprintf(stderr,"heightGenInit failed\n");return 1;}
    DevHeight *Hhost=(DevHeight*)malloc(sizeof(DevHeight)); hostPack(Hhost,&hg);
    DevHeight *Hdev; CK(cudaMalloc(&Hdev,sizeof(DevHeight)));
    CK(cudaMemcpy(Hdev,Hhost,sizeof(DevHeight),cudaMemcpyHostToDevice));

    CK(cudaMemcpyToSymbol(c_tpatdx, tpatdx, sizeof(tpatdx)));
    CK(cudaMemcpyToSymbol(c_tpatdz, tpatdz, sizeof(tpatdz)));
    CK(cudaMemcpyToSymbol(c_path,   PAT_H,  sizeof(int)*PCOUNT));
    CK(cudaMemcpyToSymbol(c_patCount,&PCOUNT,sizeof(int)));
    CK(cudaMemcpyToSymbol(c_anchorH,&anchorH,sizeof(int)));

    const int MAXOUT = 4000000;
    GMatch *dOut; CK(cudaMalloc(&dOut,sizeof(GMatch)*MAXOUT));
    int *dCount; CK(cudaMalloc(&dCount,sizeof(int)));
    GMatch *hOut=(GMatch*)malloc(sizeof(GMatch)*MAXOUT);

    FILE *mf=fopen("matches.txt","w");
    if(mf){ fprintf(mf,"--- HEIGHT MATCH LOG ---\n");
        fprintf(mf,"Seed: %lld  Center: (%d,%d)  Radius: %lld\n",(long long)seed,centX,centZ,radius);
        fprintf(mf,"Mode: %s  tol: +-%d\n\n",absolute?"absolute":"relative",tol); fflush(mf); }

    long long total=0, rSq=radius*radius;
    const int BAND = 512;                 // Z-band core height (blocks)
    int coreX0 = (int)(centX - radius);
    int coreW  = (int)(2*radius + 1);
    int nBands = (int)((2*radius)/BAND + 1);

    // reusable device buffers sized to the largest tile (band + halo).
    int maxCoreH = BAND;
    int hWmax = coreW + 2*reach;
    int hHmax = maxCoreH + 2*reach;
    // node lattice for the halo region
    int nNxMax = (hWmax)/G_CELL_XZ + 2;
    int nNzMax = (hHmax)/G_CELL_XZ + 2;
    double *dDens; CK(cudaMalloc(&dDens, sizeof(double)*(long)nNxMax*nNzMax*G_NYCORNER));
    int *dHeights; CK(cudaMalloc(&dHeights, sizeof(int)*(long)hWmax*hHmax));

    cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);

    int bandIdx=0;
    for (long long bz = centZ - radius; bz <= centZ + radius; bz += BAND, bandIdx++) {
        int coreH = BAND;
        if (bz + coreH - 1 > centZ + radius) coreH = (int)(centZ + radius - bz + 1);
        int coreZ0 = (int)bz;

        // height grid covers core + reach halo
        int hx0 = coreX0 - reach, hz0 = coreZ0 - reach;
        int hW = coreW + 2*reach, hH = coreH + 2*reach;

        // node lattice snapped out from the height grid
        int nx0 = fdivh(hx0, G_CELL_XZ)*G_CELL_XZ;
        int nz0 = fdivh(hz0, G_CELL_XZ)*G_CELL_XZ;
        int nx1 = fdivh(hx0 + hW - 1, G_CELL_XZ)*G_CELL_XZ + G_CELL_XZ;
        int nz1 = fdivh(hz0 + hH - 1, G_CELL_XZ)*G_CELL_XZ + G_CELL_XZ;
        int nNx = (nx1 - nx0)/G_CELL_XZ + 1;
        int nNz = (nz1 - nz0)/G_CELL_XZ + 1;

        // stage 1: nodes
        long nNodes = (long)nNx*nNz;
        { int th=128, bl=(int)((nNodes+th-1)/th);
          nodeKernel<<<bl,th>>>(Hdev, nx0, nz0, nNx, nNz, dDens); CK(cudaGetLastError()); }
        // stage 2: heights
        long ncols=(long)hW*hH;
        { int th=128, bl=(int)((ncols+th-1)/th);
          heightKernel<<<bl,th>>>(dDens, nx0, nz0, nNx, hx0, hz0, hW, hH, dHeights); CK(cudaGetLastError()); }
        // stage 3: match
        int zero=0; CK(cudaMemcpy(dCount,&zero,sizeof(int),cudaMemcpyHostToDevice));
        long ncore=(long)coreW*coreH;
        { int th=128, bl=(int)((ncore+th-1)/th);
          matchKernel<<<bl,th>>>(dHeights, hx0, hz0, hW, coreX0, coreZ0, coreW, coreH,
              centX, centZ, rSq, tStart, tEnd, absolute, tol, dOut, dCount, MAXOUT); CK(cudaGetLastError()); }

        int nRes=0; CK(cudaMemcpy(&nRes,dCount,sizeof(int),cudaMemcpyDeviceToHost));
        if (nRes>0){
            int cap=nRes<MAXOUT?nRes:MAXOUT;
            CK(cudaMemcpy(hOut,dOut,sizeof(GMatch)*cap,cudaMemcpyDeviceToHost));
            for(int i=0;i<cap;i++){
                if(mf) fprintf(mf,"MATCH (%d,%d) orient=%d anchorY=%d\n",
                               hOut[i].cx,hOut[i].cz,hOut[i].orient,hOut[i].anchorY);
                total++;
                if(maxMatches>0 && total>=maxMatches) break;
            }
            if(mf) fflush(mf);
        }
        printf("\r  bands %d/%d  matches %lld   ", bandIdx+1, nBands, total); fflush(stdout);
        if(maxMatches>0 && total>=maxMatches) break;
    }
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms=0; cudaEventElapsedTime(&ms,e0,e1);
    printf("\n\nDone. %lld matches in %.1fs. Written to matches.txt\n", total, ms/1000.0);
    if(mf) fclose(mf);
    return 0;
}
