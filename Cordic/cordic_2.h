// cordic.h - Header file for CORDIC algorithm
#ifndef CORDIC_H
#define CORDIC_H

#include <iostream>
#include <cmath>
#include <ap_fixed.h>

#define NUM_ITERATIONS 16
// cos(45) * cos(26) * ... * cos(~0)
#define COS_COMPENSATION (0.6072529351031395)

typedef ap_fixed<16,4> COS_SIN_TYPE;
typedef ap_fixed<16,4> THETA_TYPE;


void cordic(THETA_TYPE theta, COS_SIN_TYPE &s, COS_SIN_TYPE &c);

#endif // CORDIC_H

