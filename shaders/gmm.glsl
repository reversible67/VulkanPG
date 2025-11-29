const float k_pi = 3.1415926535897932;
const float k_2pi = 6.2831853071795864;
const float k_numeric_limits_float_min = 1.0 / exp2(126);

/**
 * Box-Muller transform takes two uniform samples u0 and u1,
 * and transforms them into two Gaussian distributed samples n0 and n1.
 * @param uv 2 uniform samples in range [0, 1]
 * @return 2 standard Gaussian distributed samples.
 */
vec2 boxMuller(vec2 uv)
{
	float u = max(uv.x, k_numeric_limits_float_min); // clamp u to avoid log(0)
	float radius = sqrt(-2.0 * log(u)); // radius of the sample
	float theta = 2 * k_pi * uv.y; // angle of the sample
	return vec2(radius * cos(theta), radius * sin(theta));
}

/**
 * Box-Muller transform takes two uniform samples u0 and u1,
 * and transforms them into two Gaussian distributed samples n0 and n1.
 * @param uv 2 uniform samples in range [0, 1]
 * @param mean mean of the Gaussian distribution
 * @param std standard deviation of the Gaussian distribution
 * @return 2 Gaussian distributed samples.
 */
vec2 boxMuller(vec2 uv, vec2 mean, vec2 std)
{
	return boxMuller(uv) * std + mean;
}

struct MultivariateGaussian2D
{
	mat2 inverseCovariance;  // precision matrix (inverse of the covariance matrix)
	vec2 mean;         // mean of the distribution
	float normalization; // normalization constant
	float inv_det_sqrt;   // determinant of the covariance matrix
};

MultivariateGaussian2D createMultivariateGaussian2D(vec2 mean, mat2 covariance)
{
	MultivariateGaussian2D gaussian2D;
	gaussian2D.mean = mean;
	gaussian2D.inverseCovariance = inverse(covariance);
	float det_precision = determinant(gaussian2D.inverseCovariance);
	gaussian2D.inv_det_sqrt = 1.0 / sqrt(det_precision);
	gaussian2D.normalization = sqrt(abs(det_precision)) / k_2pi;
	return gaussian2D;
}

// Draw a sample from the distribution
vec2 drawSample(MultivariateGaussian2D gaussian2D, vec2 uv)
{
	// First sample x from two normal distributions using Box-Muller method
	vec2 n = boxMuller(uv);
	// Once I have vector x from N(0,1) I can transform it to v = A * x + mean,
	// where A is matrix from equation Covariance = A * A^TScalar
	float c2 = gaussian2D.inverseCovariance[0][1] * gaussian2D.inverseCovariance[1][0];
	vec2 dir;
	if (gaussian2D.inverseCovariance[0][0] > gaussian2D.inverseCovariance[1][1])
	{
		float a22 = sqrt(gaussian2D.inverseCovariance[0][0]);
		float a12 = -gaussian2D.inverseCovariance[0][1] / a22;
		float a11 = sqrt(gaussian2D.inverseCovariance[1][1] - c2 / gaussian2D.inverseCovariance[0][0]);
		dir.x = gaussian2D.inv_det_sqrt * (a11 * n.x + a12 * n.y) + gaussian2D.mean.x;
		dir.y = gaussian2D.inv_det_sqrt * a22 * n.y + gaussian2D.mean.y;
	}
	else
	{
		float a11 = sqrt(gaussian2D.inverseCovariance[1][1]);
		float a21 = -gaussian2D.inverseCovariance[0][1] / a11;
		float a22 = sqrt(gaussian2D.inverseCovariance[0][0] - c2 / gaussian2D.inverseCovariance[1][1]);
		dir.x = gaussian2D.inv_det_sqrt * a11 * n.x + gaussian2D.mean.x;
		dir.y = gaussian2D.inv_det_sqrt * (a21 * n.x + a22 * n.y) + gaussian2D.mean.y;
	}
	return dir;
}

float quadraticForm(mat2 mat, vec2 vec)
{
	return dot(vec, mat * vec);
}

// evaluate the probability density function at the given sample
float pdfGMM(MultivariateGaussian2D gaussian2D, vec2 s)
{
	vec2 x = s - gaussian2D.mean;
	return gaussian2D.normalization * exp(-0.5 * quadraticForm(gaussian2D.inverseCovariance, x));
}

/**
 * Compute the covariance matrix of a 2D variable pair.
 * @param ex The expected value of the first variable.
 * @param ey The expected value of the second variable.
 * @param ex2 The expected value of the first variable squared.
 * @param ey2 The expected value of the second variable squared.
 * @param exy The expected value of the product of the two variables.
 * @return The covariance matrix.
*/
mat2 covariance(float ex, float ey, float ex2, float ey2, float exy)
{
	float vx = ex2 - ex * ex;  // variance of x
	float vy = ey2 - ey * ey;  // variance of y
	float vxy = exy - ex * ey; // covariance of x and y
	return mat2(vx, vxy, vxy, vy);
}

struct GMMStatictics
{
	float ex;
	float ey;
	float ex2;
	float ey2;
	float exy;
};

GMMStatictics unpackStatistics(vec4 pack0, vec4 pack1)
{
	float weightSum = pack1.y;
	GMMStatictics stat;
	stat.ex = pack0.x / weightSum;
	stat.ey = pack0.y / weightSum;
	stat.ex2 = pack0.z / weightSum;
	stat.ey2 = pack0.w / weightSum;
	stat.exy = pack1.x / weightSum;
	return stat;
}

mat2 createCovariance(GMMStatictics statistics)
{
	return covariance(statistics.ex, statistics.ey, statistics.ex2, statistics.ey2, statistics.exy);
}

MultivariateGaussian2D createDistribution(GMMStatictics statistics)
{
	mat2 covariance = createCovariance(statistics);
	return createMultivariateGaussian2D(vec2(statistics.ex, statistics.ey), covariance);
}

struct GMM2D
{
	MultivariateGaussian2D lobes[4];
	vec4 sufficientStats0[4];
	vec4 sufficientStats1[4];
	int epoch_cap;
};

/** Compute the responsibility of a point for a given lobe.
 * @param h The index of the lobe.
 * @param point The point. */
float responsibility(GMM2D gmms, int h, vec2 point)
{
	return gmms.sufficientStats1[h].w * pdfGMM(gmms.lobes[h], point);
}

vec2 drawSample(GMM2D gmm, vec3 uv)
{
	float pdf = 0.0;
	for (int h = 0; h < 4; ++h)
	{
		pdf += gmm.sufficientStats1[h].w;
		if (uv.z < pdf || h == 4 - 1)
			return drawSample(gmm.lobes[h], uv.xy);
	}
	return vec2(0.0);
}

void buildGMMs(inout GMM2D gmms)
{
	for (int h = 0; h < 4; ++h)
	{
		GMMStatictics GMMstat = unpackStatistics(gmms.sufficientStats0[h], gmms.sufficientStats1[h]);
		gmms.lobes[h] = createMultivariateGaussian2D(vec2(GMMstat.ex, GMMstat.ey), createCovariance(GMMstat));
	}
}

float pdfGMMs(GMM2D gmms, vec2 point)
{
	float pdf = 0.0;
	for (int h = 0; h < 4; ++h)
		pdf += responsibility(gmms, h, point);
	return pdf;
}

void updateGMMs(inout GMM2D gmms, int h, vec2 square_coord, float sumWeight, float exponential_factor)
{
	if (sumWeight > 0)
	{
		// exponential smoothing vMF
		vec4 pack0 = gmms.sufficientStats0[h];
		vec4 pack1 = gmms.sufficientStats1[h];
		uint epoch_count = uint(pack1.z);
		float alpha = pow(exponential_factor, epoch_count);
		epoch_count = clamp(epoch_count, 0, gmms.epoch_cap);
		vec4 new_pack0 = sumWeight * vec4(square_coord.x, square_coord.y, square_coord.x * square_coord.x, square_coord.y * square_coord.y);
		vec2 new_pack1 = sumWeight * vec2(square_coord.x * square_coord.y, 1);
		vec4 update_pack0 = smoothstep(pack0, new_pack0, vec4(alpha));
		vec2 update_pack1 = smoothstep(pack1.xy, new_pack1, vec2(alpha));
		epoch_count += 1;
		gmms.sufficientStats0[h] = update_pack0;
		gmms.sufficientStats1[h] = vec4(update_pack1, epoch_count, pack1.w);
	}
}

void stepwiseEM(inout GMM2D gmms, float sumWeight, vec2 square_coord, float exponential_factor)
{
	if (sumWeight <= 0)
		return;
	// E-step: update sufficient statistics
	float pdf[4];
	float denom = 0.f;
	vec2 x = vec2(0.0);
	for (int h = 0; h < 4; ++h)
	{
		pdf[h] = responsibility(gmms, h, x);
		if (isnan(pdf[h]))
			pdf[h] = 0.001;
		if (isinf(pdf[h]))
			pdf[h] = 10000;
		if (pdf[h] <= 0)
			pdf[h] = 0.001;
		denom += pdf[h];
	}
	// M-step: update model
	float weights[4];
	float w_denom = 0.0;
	for (int h = 0; h < 4; ++h)
	{
		float posterior = pdf[h] / denom;
		updateGMMs(gmms, h, square_coord, sumWeight * posterior, exponential_factor);
		weights[h] = gmms.sufficientStats1[h].y;
		w_denom += weights[h];
	}
	// normalize weights
	for (int h = 0; h < 4; ++h)
		gmms.sufficientStats1[h].w = weights[h] / w_denom;

	// sufficientStats1[0].w = 1;
	// sufficientStats1[1].w = 0;
	// sufficientStats1[2].w = 0;
	// sufficientStats1[3].w = 0;
}