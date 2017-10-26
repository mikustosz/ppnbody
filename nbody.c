/*
 * Copyright (c) 2016, NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#ifndef FP32
#define FP64
#endif

#ifdef FP64
typedef double real;
const real SOFTENING_SQUARED = 0.01;
#define RSQRT(x) 1.0 / sqrt((x))
#define POW(x,y) pow((x),(y))
#else
typedef float real;
const real SOFTENING_SQUARED = 0.01f;
#define RSQRT(x) 1.0f / sqrtf((x))
#define POW(x,y) powf((x),(y))
#endif
#define forj for (j = i+1; j < n; j++)

typedef struct { real x, y, z; }    real3;

void integrate(real *Xin,  real *Yin,  real *Zin,  real *Win,  
			   real *Xout, real *Yout, real *Zout, real *Wout,
               real3 *vel, real *forceX, real *forceY, real *forceZ,
               real dt, int n,
               real *rxs, real *rys, real *rzs, real *distSqrs, real *ss)
{
  int i, j;
  for (i = 0; i < n; i++)
  {
  	#pragma omp simd
    forj { rxs[j] = Xin[j] - Xin[i]; }
  	#pragma omp simd
    forj { rys[j] = Yin[j] - Yin[i]; }
  	#pragma omp simd
    forj { rzs[j] = Zin[j] - Zin[i]; }
  	#pragma omp simd
    forj {
      distSqrs[j] = rxs[j]*rxs[j] + rys[j]*rys[j] + rzs[j]*rzs[j];
	  distSqrs[j] += SOFTENING_SQUARED;
	}
  	#pragma omp simd
    forj { ss[j] = Win[j] / POW(distSqrs[j], 3.0/2.0); }

  	#pragma omp simd
    forj
    {
	  // TODO optimize double multiplication?
      forceX[i] += rxs[j] * ss[j];
      forceY[i] += rys[j] * ss[j];
      forceZ[i] += rzs[j] * ss[j];

      forceX[j] -= rxs[j] * ss[j];
      forceY[j] -= rys[j] * ss[j];
      forceZ[j] -= rzs[j] * ss[j];
    }
  }

  for (i = 0; i < n; i++)
  {
    real fx = forceX[i],  fy = forceY[i],  fz = forceZ[i];
    real px = Xin[i],     py = Yin[i],     pz = Zin[i], invMass = Win[i];
    real vx = vel[i].x,   vy = vel[i].y,   vz = vel[i].z;

    // acceleration = force / mass; 
    // new velocity = old velocity + acceleration * deltaTime
    vx += (fx * invMass) * dt;
    vy += (fy * invMass) * dt;
    vz += (fz * invMass) * dt;

    // new position = old position + velocity * deltaTime
    px += vx * dt;
    py += vy * dt;
    pz += vz * dt;

    Xout[i] = px;
    Yout[i] = py;
    Zout[i] = pz;
    Wout[i] = invMass;

    vel[i].x = vx;
    vel[i].y = vy;
    vel[i].z = vz;
  }
}

real dot(real v0[3], real v1[3])
{
  return v0[0]*v1[0]+v0[1]*v1[1]+v0[2]*v1[2];
}

real normalize(real vector[3])
{
  float dist = sqrt(dot(vector, vector));
  if (dist > 1e-6)
  {
    vector[0] /= dist;
    vector[1] /= dist;
    vector[2] /= dist;
  }
  return dist;
}

void cross(real out[3], real v0[3], real v1[3])
{
  out[0] = v0[1]*v1[2]-v0[2]*v1[1];
  out[1] = v0[2]*v1[0]-v0[0]*v1[2];
  out[2] = v0[0]*v1[1]-v0[1]*v1[0];
}

void randomizeBodies(real *X, real *Y, real *Z, real *W, 
                     real3 *vel, 
                     float clusterScale, 
                     float velocityScale, 
                     int   n)
{
  srand(42);
  float scale = clusterScale;
  float vscale = scale * velocityScale;
  float inner = 2.5f * scale;
  float outer = 4.0f * scale;

  int i = 0;
  while (i < n)
  {
    real x, y, z;
    x = rand() / (float) RAND_MAX * 2 - 1;
    y = rand() / (float) RAND_MAX * 2 - 1;
    z = rand() / (float) RAND_MAX * 2 - 1;

    real point[3] = {x, y, z};
    real len = normalize(point);
    if (len > 1)
      continue;

    X[i]= point[0] * (inner + (outer - inner)*rand() / (real) RAND_MAX);
    Y[i]= point[1] * (inner + (outer - inner)*rand() / (real) RAND_MAX);
    Z[i]= point[2] * (inner + (outer - inner)*rand() / (real) RAND_MAX);
    W[i]= 1.0f;

    x = 0.0f; 
    y = 0.0f; 
    z = 1.0f; 
    real axis[3] = {x, y, z};
    normalize(axis);

    if (1 - dot(point, axis) < 1e-6)
    {
      axis[0] = point[1];
      axis[1] = point[0];
      normalize(axis);
    }
    real vv[3] = {X[i], Y[i], Z[i]};
    cross(vv, vv, axis);
    vel[i].x = vv[0] * vscale;
    vel[i].y = vv[1] * vscale;
    vel[i].z = vv[2] * vscale;

    i++;
  }
}

real3 average(real *X, real *Y, real *Z, int n)
{
  int i;
  real3 av= {0.0, 0.0, 0.0};
  for (i = 0; i < n; i++)
  {
    av.x += X[i];
    av.y += Y[i];
    av.z += Z[i];
  }
  av.x /= n;
  av.y /= n;
  av.z /= n;
  return av;
}

int main(int argc, char** argv)
{
  int i, n = 20000;
  int iterations = 10;
  real dt = 0.01667;

  if (argc >= 2) n = atoi(argv[1]);
  if (argc >= 3) iterations = atoi(argv[2]);

  real3 *v    = (real3*)malloc(n * sizeof(real3));

  real *fx = (real*)malloc(n * sizeof(real));
  real *fy = (real*)malloc(n * sizeof(real));
  real *fz = (real*)malloc(n * sizeof(real));

  real *Xin  = (real*)malloc(n * sizeof(real));
  real *Yin  = (real*)malloc(n * sizeof(real));
  real *Zin  = (real*)malloc(n * sizeof(real));
  real *Win  = (real*)malloc(n * sizeof(real));
  real *Xout = (real*)malloc(n * sizeof(real));
  real *Yout = (real*)malloc(n * sizeof(real));
  real *Zout = (real*)malloc(n * sizeof(real));
  real *Wout = (real*)malloc(n * sizeof(real));

  // For integrate function
  real *rxs = (real*)malloc(n * sizeof(real));
  real *rys = (real*)malloc(n * sizeof(real));
  real *rzs = (real*)malloc(n * sizeof(real));
  real *distSqrs = (real*)malloc(n * sizeof(real));
  real *ss = (real*)malloc(n * sizeof(real));

  randomizeBodies(Xin, Yin, Zin, Win, v,  1.54f, 8.0f, n);

  printf("n=%d bodies for %d iterations:\n", n, iterations);

  for (i = 0; i < iterations; i++)
  {
    integrate(Xin, Yin, Zin, Win, Xout, Yout, Zout, Wout, v, fx, fy, fz, dt, n, rxs, rys, rzs, distSqrs, ss);
    real *tx, *ty, *tz, *tw;
    tx = Xout;  ty = Yout;  tz = Zout;  tw = Wout;
    Xout = Xin; Yout = Yin; Zout = Zin; Wout = Win;
    Xin = tx;   Yin = ty;   Zin = tz;   Win = tw;
  }

  real3 p_av = average(Xout, Yout, Zout, n);
  printf("Average position: (%f,%f,%f)\n", p_av.x, p_av.y, p_av.z);

  free(v);    free(fx);   free(fy);   free(fz);
  free(Xin);  free(Yin);  free(Zin);  free(Win);
  free(Xout); free(Yout); free(Zout); free(Wout);
  free(rxs);
  free(rys);
  free(rzs);
  free(distSqrs);
  free(ss);

  return 0;
}

