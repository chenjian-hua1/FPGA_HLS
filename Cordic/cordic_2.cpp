#include "cordic_2.h"
#include <iostream>

// The cordic_phase array holds the angle for the current rotation (in radians)
// These are the arctangent values of 2^(-i)
// tan(theta) = 2^-i    ->   theta=arctan(2^-i)

// theta is radian
THETA_TYPE cordic_angle[NUM_ITERATIONS] = {
	0.7853981633974483,
	0.46364760900080615,
	0.24497866312686414,
	0.12435499454676144,
	0.06241880999595735,
	0.031239833430268277,
	0.015623728620476831,
	0.007812341060101111,
	0.0039062301319669718,
	0.0019531225164788188,
	0.0009765621895593195,
	0.0004882812111948983,
	0.00024414062014936177,
	0.00012207031189367021,
	6.103515617420877e-05,
	3.0517578115526096e-05
};


// theta is radian
void cordic(THETA_TYPE theta, COS_SIN_TYPE &s, COS_SIN_TYPE &c) {
	// current angle
	THETA_TYPE crt_angle;
	COS_SIN_TYPE x, y;
	COS_SIN_TYPE x_tan = 0, y_tan = 0;

	// Decide rotation start point
	if (theta >= M_PI*3/2 || theta <= M_PI/2) {
		// 1,4 Quadrant
		x = 1;
		y = 0;
		crt_angle = 0;
	}
	else {
		// 2,3 Quadrant
		x = -1;
		y = 0;
		crt_angle = M_PI;
	}

	// Rotation <NUM_ITERATIONS> times
	for (int i=0; i<NUM_ITERATIONS; i++) {
		// --------------- HLS -----------------------
		/* tan(theta[i]) = (2^-i) = /(2^i) -> right shift i bits */
		y_tan = y >> i;
		x_tan = x >> i;

		// -------------- C++ -------------------------
//		x_tan = x*pow(2, -i);
//		y_tan = y*pow(2, -i);

		// ------------- Share ------------------------
		 // +, -
		bool sign = (crt_angle<=theta);
		crt_angle = (sign)?(crt_angle+cordic_angle[i]):(crt_angle-cordic_angle[i]);

		// x' = x-y*(tan=2^-i)
//		x = ((sign)?(-y_tan):(y_tan)) + x;
		// x = y_tan + x;
		x = (sign)?(x-y_tan):(x+y_tan);
		// y' = x*(tan=2^-i)+y
		// y = ((sign)?(x_tan):(-x_tan)) + y;
		y = (sign)?(y+x_tan):(y-x_tan);


//		std::cout << x << " " << y_tan << " " << crt_angle << std::endl;
	}

	// sin = y/r (r=1)
//	s = y*COS_COMPENSATION;
//	c = x*COS_COMPENSATION;

	y = y*(COS_SIN_TYPE)COS_COMPENSATION;
	x = x*(COS_SIN_TYPE)COS_COMPENSATION;
	
	s = y;
	c = x;
}
