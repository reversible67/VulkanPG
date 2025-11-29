vec2 toConcentricMap(vec2 onSquare)
{
    float phi;
	float r;
    // (a,b) is now on [-1,1]^2
    float a = 2 * onSquare.x - 1;
    float b = 2 * onSquare.y - 1;
    if (a > -b)
	{ // region 1 or 2
        if (a > b)
		{ // region 1, also |a| > |b|
            r = a;
            phi = (k_pi / 4) * (b / a);
        }
        else
		{ // region 2, also |b| > |a|
            r = b;
            phi = (k_pi / 4) * (2 - (a / b));
        }
    }
    else
	{ // region 3 or 4
        if (a < b)
		{ // region 3, also |a| > |b|, a!= 0
            r = -a;
            phi = (k_pi / 4) * (4 + (b / a));
        }
        else
		{ // region 4, |b| >= |a|, but a==0 and b==0 could occur.
            r = -b;
            if (b != 0)
				phi = (k_pi / 4) * (6 - (a / b));
            else phi = 0;
        }
    }
    float u = r * cos(phi);
    float v = r * sin(phi);
    return vec2(u, v);
}

vec2 fromConcentricMap(vec2 onDisk)
{
    float r = sqrt(onDisk.x * onDisk.x + onDisk.y * onDisk.y);
    float phi = atan(onDisk.y, onDisk.x);
    if (phi < -k_pi / 4)
        phi += 2 * k_pi; // in range [-pi/4,7pi/4]
    float a;
	float b;
    if (phi < k_pi / 4)
	{ // region 1
        a = r;
        b = phi * a / (k_pi / 4);
    }
    else if (phi < 3 * k_pi / 4)
	{ // region 2
        b = r;
        a = -(phi - k_pi / 2) * b / (k_pi / 4);
    }
    else if (phi < 5 * k_pi / 4)
	{ // region 3
        a = -r;
        b = (phi - k_pi) * a / (k_pi / 4);
    }
    else
	{ // region 4
        b = -r;
        a = -(phi - 3 * k_pi / 2) * b / (k_pi / 4);
    }
    float x = (a + 1) / 2;
    float y = (b + 1) / 2;
    return vec2(x, y);
}

/**
 * Maps a point on the disk to a point on the unit hemisphere.
 * @ref: "A Low Distortion Map Between Disk and Square" - Peter Shirley & Kenneth Chiu
 */
vec3 concentricDiskToUniformHemisphere(vec2 onDisk)
{
    float r2 = onDisk.x * onDisk.x + onDisk.y * onDisk.y;
    float r = sqrt(r2);
    float z = 1.0 - r2;
    float z2 = z * z;
    float tmp = sqrt(1.0 - z2);
    float x = onDisk.x * tmp / r;
    float y = onDisk.y * tmp / r;
    return vec3(x, y, z);
}

/**
 * Maps a point on the unit hemisphere to a point on the disk.
 * @ref: "A Low Distortion Map Between Disk and Square" - Peter Shirley & Kenneth Chiu
 */
vec2 uniformHemisphereToConcentricDisk(vec3 onHemisphere)
{
    float r = sqrt(1.0 - onHemisphere.z);
    float tmp = sqrt(1.0 - onHemisphere.z * onHemisphere.z);
    float x = onHemisphere.x * r / tmp;
    float y = onHemisphere.y * r / tmp;
    return vec2(x, y);
}
